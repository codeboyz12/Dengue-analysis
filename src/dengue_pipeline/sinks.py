"""ช่องเสียบปลายทางของข้อมูลที่ผ่านตรวจ (ขั้น load)

มี 2 แบบ เลือกด้วย [load] sink ใน config:
  file    FileSink    ไฟล์ในโฟลเดอร์ data/validated/<run_id>/
  duckdb  DuckDBSink  ไฟล์ฐานข้อมูลเดียว data/pipeline.duckdb (sink_duckdb.py) ค่าตั้งต้นของ pipeline.toml
ทั้งสองมีเมธอดเดียวกัน ขั้นอื่นเรียกผ่าน get_sink() จึงไม่ต้องแก้
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.io import read_json, read_table, write_json, write_table


class Sink(Protocol):
    def write(self, run_id: str, tables: dict[str, pd.DataFrame], metadata: dict) -> str: ...
    def latest_metadata(self) -> dict | None: ...
    def latest_version(self) -> str | None: ...
    def read(self, version: str, table: str, columns: list[str] | None = None) -> pd.DataFrame: ...


class FileSink:
    """เขียนลง data/validated/<run_id>/ แบบ atomic (เขียนโฟลเดอร์ชั่วคราวก่อนแล้วค่อยเปลี่ยนชื่อ)
    เวอร์ชันเดิมที่ผ่านแล้วไม่ถูกแตะ"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fmt = cfg["load"]["format"]

    def write(self, run_id: str, tables: dict[str, pd.DataFrame], metadata: dict) -> str:
        final = self.cfg.validated_dir / run_id
        if final.exists():
            raise FileExistsError(f"validated/{run_id} มีอยู่แล้ว (ไม่เขียนทับเวอร์ชันที่ผ่านแล้ว)")
        tmp = self.cfg.validated_dir / f".tmp-{run_id}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            for name, df in tables.items():
                write_table(df, tmp / name, self.fmt)
            write_json(tmp / "metadata.json", metadata)
            tmp.rename(final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return str(final)

    def _versions(self) -> list[Path]:
        d = self.cfg.validated_dir
        if not d.exists():
            return []
        runs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
        return [p for p in runs if (p / "metadata.json").exists()]

    def read_metadata(self, version: str) -> dict:
        return read_json(self.cfg.validated_dir / version / "metadata.json")

    def latest_metadata(self) -> dict | None:
        runs = self._versions()
        return read_json(runs[-1] / "metadata.json") if runs else None

    def latest_version(self) -> str | None:
        runs = self._versions()
        return runs[-1].name if runs else None

    def read(self, version: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """อ่านตารางจากเวอร์ชันที่ผ่านตรวจแล้ว (version = run_id ของข้อมูล) columns = เลือกเฉพาะคอลัมน์"""
        vdir = self.cfg.validated_dir / version
        if not vdir.is_dir():
            raise FileNotFoundError(f"ไม่พบเวอร์ชันใน validated: {version}")
        df = read_table(vdir / table)
        if columns is None:
            return df
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"ตาราง {table} ไม่มีคอลัมน์ {missing}")
        return df[columns]

    def versions(self) -> list[str]:
        return [p.name for p in self._versions()]


def get_sink(cfg: Config) -> Sink:
    kind = cfg.data.get("load", {}).get("sink", "file")
    if kind == "file":
        return FileSink(cfg)
    if kind == "duckdb":
        from dengue_pipeline.sink_duckdb import DuckDBSink      # import ตอนใช้ ไม่บังคับติดตั้งเมื่อใช้แบบ file
        return DuckDBSink(cfg)
    raise ValueError(f"[load] sink ต้องเป็น file หรือ duckdb (ได้ {kind!r})")
