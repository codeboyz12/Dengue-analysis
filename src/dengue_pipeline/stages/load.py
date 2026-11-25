"""ขั้น 5 load: ส่งชุดที่ผ่านตรวจทั้งสองจุดเข้าปลายทาง (DuckDB หรือไฟล์ในชั้น validated ตาม [load] sink)

ขั้นนี้ถูกเรียกได้เฉพาะเมื่อ validate-merged สำเร็จ (ตัวรันบังคับ) ข้อมูลที่ผ่านก่อนหน้าไม่ถูกแตะ
ปลายทางเลือกด้วย [load] sink ใน config (sinks.get_sink)
"""
from __future__ import annotations

import pandas as pd

from dengue_pipeline import __version__
from dengue_pipeline.config import Config
from dengue_pipeline.io import read_json, read_table
from dengue_pipeline.runstate import RunState, utc_now
from dengue_pipeline.sinks import get_sink


def run(cfg: Config, state: RunState, args) -> str:
    sdir = cfg.abs(state.staging)
    wdir, tdir = cfg.abs(state.raw["weather"]), cfg.abs(state.raw["trends"])
    wman, tman = read_json(wdir / "manifest.json"), read_json(tdir / "manifest.json")

    merged = read_table(sdir / "trends_weather")
    daily = pd.read_csv(wdir / wman["file"], parse_dates=["date"])
    meta = read_json(sdir / "meta.json")

    metadata = {
        "run_id": state.run_id, "written_at": utc_now(), "pipeline_version": __version__,
        "source_rows": {"trends": tman["rows"], "weather": wman["rows"]},
        "rows": {"trends_weather": len(merged), "weather_daily": len(daily)},
        "lags": meta["lags"],
        "raw": {"trends": tman, "weather": wman},
        "rules": state.rules,
    }
    path = get_sink(cfg).write(state.run_id, {"trends_weather": merged, "weather_daily": daily}, metadata)
    state.validated = path if "#" in path else cfg.rel(path)      # duckdb: ชื่อไฟล์#run_id ส่วน file: path ของโฟลเดอร์
    state.save(cfg)
    return f"เขียนที่ {state.validated} ({len(merged):,} แถว)"
