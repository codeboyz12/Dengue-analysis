"""ปลายทางแบบ DuckDB: ไฟล์ฐานข้อมูลเดียว (data/pipeline.duckdb) แทนโฟลเดอร์ validated/<run_id>/

โครงสร้าง (ตามที่ออกแบบไว้ ระยะ 2)
  mart.trends_weather       ตารางกว้างที่ผ่านตรวจ ทุกเวอร์ชันอยู่ในตารางเดียว แยกด้วยคอลัมน์ data_version
  core.weather_daily        อากาศรายวันที่ผ่านตรวจ (แยกเวอร์ชันแบบเดียวกัน)
  ops.data_version          ทะเบียนเวอร์ชัน: run_id, เวลา, metadata ทั้งก้อน (JSON)
  ops.audit_log             ผลทุกกฎ ทุกรอบ (สำเนาของ reports/audit_log.csv)
  <ตาราง>_latest (view)     เวอร์ชันล่าสุดของแต่ละตาราง สำหรับคนที่ใช้ SQL

กติกา: เขียนเพิ่มอย่างเดียว ไม่มี UPDATE/DELETE กับเวอร์ชันที่ผ่านแล้ว เขียนทั้งเวอร์ชันใน transaction เดียว
(ล้มกลางทาง = ไม่มีอะไรเข้าฐานข้อมูล) แถวถูกเก็บพร้อมลำดับเดิม (_ord) จึงอ่านกลับมาได้ตารางเดียวกับที่เขียน
DuckDB เขียนได้ทีละ process: pipeline เป็นผู้เขียนคนเดียว ส่วนคนอ่านเปิดแบบ read_only ได้พร้อมกันหลายคน
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.errors import StageError

TABLE_SCHEMA = {"trends_weather": "mart", "weather_daily": "core"}
DEFAULT_DB_FILE = "pipeline.duckdb"
META_COLS = ("data_version", "_ord")


def _q(name: str) -> str:
    """quote identifier ของ DuckDB (ชื่อคอลัมน์มีวงเล็บ ช่องว่าง และ ² )"""
    return '"' + name.replace('"', '""') + '"'


def _schema_of(table: str) -> str:
    return TABLE_SCHEMA.get(table, "mart")


class DuckDBSink:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.data_dir / cfg.data.get("load", {}).get("db_file", DEFAULT_DB_FILE)

    # --- การเชื่อมต่อ -----------------------------------------------------------
    def _connect(self, read_only: bool = False):
        import duckdb
        if read_only:
            if not self.path.exists():
                return None
            return duckdb.connect(str(self.path), read_only=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.path))

    @staticmethod
    def _ensure_ops(con) -> None:
        con.execute("CREATE SCHEMA IF NOT EXISTS ops")
        con.execute("""CREATE TABLE IF NOT EXISTS ops.data_version (
            data_version VARCHAR PRIMARY KEY, written_at VARCHAR, pipeline_version VARCHAR, metadata VARCHAR)""")
        con.execute("""CREATE TABLE IF NOT EXISTS ops.audit_log (
            run_id VARCHAR, stage VARCHAR, rule_id VARCHAR, severity VARCHAR, outcome VARCHAR,
            measured VARCHAR, detail VARCHAR, checked_at VARCHAR)""")

    @staticmethod
    def _has_table(con, schema: str, table: str) -> bool:
        return con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema=? AND table_name=?",
                           [schema, table]).fetchone()[0] > 0

    # --- Sink ----------------------------------------------------------------
    def write(self, run_id: str, tables: dict[str, pd.DataFrame], metadata: dict) -> str:
        con = self._connect()
        try:
            self._ensure_ops(con)
            if con.execute("SELECT count(*) FROM ops.data_version WHERE data_version=?", [run_id]).fetchone()[0]:
                raise FileExistsError(f"{self.path.name} มีเวอร์ชัน {run_id} อยู่แล้ว (ไม่เขียนทับเวอร์ชันที่ผ่านแล้ว)")
            con.execute("BEGIN TRANSACTION")
            try:
                for name, df in tables.items():
                    self._append(con, name, run_id, df)
                con.execute("INSERT INTO ops.data_version VALUES (?, ?, ?, ?)",
                            [run_id, str(metadata.get("written_at", "")), str(metadata.get("pipeline_version", "")),
                             json.dumps(metadata, ensure_ascii=False, default=str)])
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            # view ของเวอร์ชันล่าสุด (นอก transaction หลักเพราะเป็น DDL ซ้ำได้ ไม่มีผลต่อข้อมูล)
            for name in tables:
                s = _schema_of(name)
                con.execute(f"""CREATE OR REPLACE VIEW {s}.{name}_latest AS
                    SELECT * EXCLUDE (_ord) FROM {s}.{name}
                    WHERE data_version = (SELECT max(data_version) FROM ops.data_version)
                    ORDER BY _ord""")
        finally:
            con.close()
        return f"{self.path.name}#{run_id}"

    def _append(self, con, name: str, run_id: str, df: pd.DataFrame) -> None:
        s = _schema_of(name)
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
        body = df.reset_index(drop=True).copy()
        body.insert(0, "_ord", range(len(body)))
        body.insert(0, "data_version", run_id)
        con.register("_incoming", body)
        try:
            if not self._has_table(con, s, name):
                con.execute(f"CREATE TABLE {s}.{name} AS SELECT * FROM _incoming LIMIT 0")
            have = [r[0] for r in con.execute(f"DESCRIBE {s}.{name}").fetchall()]
            if have != list(body.columns):
                raise StageError(
                    f"คอลัมน์ของตาราง {s}.{name} ในฐานข้อมูลไม่ตรงกับข้อมูลใหม่: "
                    f"ขาด {[c for c in have if c not in body.columns]}, เกิน {[c for c in body.columns if c not in have]} "
                    "(เปลี่ยนโครงสร้างแล้วต้องสร้างฐานข้อมูลใหม่ หรือย้ายตารางเดิมก่อน)")
            con.execute(f"INSERT INTO {s}.{name} SELECT * FROM _incoming")
        finally:
            con.unregister("_incoming")

    def versions(self) -> list[str]:
        con = self._connect(read_only=True)
        if con is None:
            return []
        try:
            if not self._has_table(con, "ops", "data_version"):
                return []
            return [r[0] for r in con.execute("SELECT data_version FROM ops.data_version ORDER BY 1").fetchall()]
        finally:
            con.close()

    def latest_version(self) -> str | None:
        v = self.versions()
        return v[-1] if v else None

    def latest_metadata(self) -> dict | None:
        v = self.latest_version()
        if v is None:
            return None
        con = self._connect(read_only=True)
        try:
            raw = con.execute("SELECT metadata FROM ops.data_version WHERE data_version=?", [v]).fetchone()[0]
        finally:
            con.close()
        return json.loads(raw)

    def read(self, version: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """อ่านตารางของเวอร์ชันหนึ่ง คืนลำดับแถวและคอลัมน์เหมือนตอนเขียน
        columns = เลือกเฉพาะคอลัมน์ที่ต้องใช้ (ฐานข้อมูลแบบ columnar อ่านเฉพาะส่วนนั้นจริง)"""
        s = _schema_of(table)
        con = self._connect(read_only=True)
        if con is None or version not in self.versions():
            if con is not None:
                con.close()
            raise FileNotFoundError(f"ไม่พบเวอร์ชันในฐานข้อมูล: {version}")
        try:
            if not self._has_table(con, s, table):
                raise FileNotFoundError(f"ไม่พบตาราง {s}.{table}")
            have = [r[0] for r in con.execute(f"DESCRIBE {s}.{table}").fetchall() if r[0] not in META_COLS]
            if columns is None:
                columns = have
            missing = [c for c in columns if c not in have]
            if missing:
                raise KeyError(f"ตาราง {s}.{table} ไม่มีคอลัมน์ {missing}")
            sel = ", ".join(_q(c) for c in columns)
            return con.execute(f"SELECT {sel} FROM {s}.{table} WHERE data_version = ? ORDER BY _ord",
                               [version]).df()
        finally:
            con.close()

    # --- audit log -----------------------------------------------------------
    def record_audit(self, rows: list[tuple]) -> None:
        con = self._connect()
        try:
            self._ensure_ops(con)
            con.executemany("INSERT INTO ops.audit_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        finally:
            con.close()


def mirror_audit(cfg: Config, rows: list[tuple]) -> None:
    """สำเนา audit log ลงฐานข้อมูล (ถ้าใช้ sink แบบ duckdb) ล้มเหลวแล้วเตือนเฉย ๆ เพราะ CSV ยังเป็นต้นฉบับ"""
    if cfg.data.get("load", {}).get("sink", "file") != "duckdb":
        return
    try:
        DuckDBSink(cfg).record_audit(rows)
    except Exception as ex:                      # noqa: BLE001 — ฐานข้อมูลถูกล็อกโดย process อื่น ฯลฯ
        print(f"⚠ บันทึก audit ลงฐานข้อมูลไม่สำเร็จ ({ex}) — ยังอยู่ใน reports/audit_log.csv", file=sys.stderr)
