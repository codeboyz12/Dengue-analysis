"""ขั้น 6 export: ส่งออกตารางฐาน 44 คอลัมน์ให้ notebook (อ่านจากชั้น validated ผ่าน sink)

ไม่อยู่ในสายหลัก (STAGES) เพราะทำเมื่อต้องการข้อมูลเทรน: python -m dengue_pipeline export
ผลลัพธ์: data/exports/contract_v<N>/<run_id ของข้อมูล>/trends_weather_merged.csv + metadata.json

  - ไฟล์นี้ใช้แทน trends_weather_merged.csv ใน notebook ได้ทันที (คอลัมน์ ลำดับ ชนิดข้อมูลเหมือนกัน)
  - ป้าย D ฟีเจอร์ และ anomaly ต่อ fold สร้างใน notebook ที่เดียว pipeline ไม่นิยามซ้ำ
  - ต้องผ่านกฎ EXP-01..03 และ LKG-02 ก่อนเขียนไฟล์ ถ้าไม่ผ่านจะไม่เขียนอะไรเลย
"""
from __future__ import annotations

from pathlib import Path

from dengue_pipeline import __version__
from dengue_pipeline.config import Config
from dengue_pipeline.contract import Contract
from dengue_pipeline.errors import StageError
from dengue_pipeline.io import sha256_of, write_json
from dengue_pipeline.rules import raise_if_blocked, summarize, write_audit
from dengue_pipeline.runstate import new_run_id, utc_now
from dengue_pipeline.sinks import get_sink
from dengue_pipeline.validation.checks_export import check_export

DEFAULT_CONTRACT = "config/contract_v2.toml"


def resolve_contract_path(cfg: Config, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    p = Path(cfg.data.get("export", {}).get("contract", DEFAULT_CONTRACT)).expanduser()
    return p if p.is_absolute() else cfg.root / p


def export_csv(cfg: Config, version: str | None = None, contract_path: str | None = None) -> dict:
    sink = get_sink(cfg)
    version = version or sink.latest_version()
    if version is None:
        raise StageError("ยังไม่มีข้อมูลในชั้น validated: รัน `run` ให้สำเร็จก่อน")
    contract = Contract.load(resolve_contract_path(cfg, contract_path))

    df = sink.read(version, "trends_weather")
    absent = [c for c in contract.columns if c not in df.columns]
    if absent:
        raise StageError(f"ตาราง validated ({version}) ขาดคอลัมน์ที่ contract v{contract.version} ต้องการ: "
                         f"{absent} (จำนวน lag ใน config น้อยกว่าที่ contract ระบุหรือไม่?)")

    table = df[contract.output_columns].sort_values(contract.keys, kind="mergesort").reset_index(drop=True)
    results = check_export(table, contract)
    write_audit(cfg, f"export-{new_run_id()}", "export", results)
    raise_if_blocked("export", results)

    out_dir = cfg.exports_dir / f"contract_v{contract.version}" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trends_weather_merged.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8")

    meta = {
        "contract_version": contract.version, "data_version": version,
        "exported_at": utc_now(), "pipeline_version": __version__,
        "rows": len(table), "columns": list(table.columns),
        "sha256": sha256_of(csv_path), "rules": summarize(results),
    }
    write_json(out_dir / "metadata.json", meta)
    return {**meta, "path": str(csv_path)}
