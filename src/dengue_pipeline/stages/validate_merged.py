"""ขั้น 4 validate-merged: ตรวจความสอดคล้องภายในหลัง merge ด้วยกฎชั้น 2 (MRG)"""
from __future__ import annotations

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.io import read_json, read_table
from dengue_pipeline.rules import raise_if_blocked, summarize, write_audit
from dengue_pipeline.runstate import RunState
from dengue_pipeline.validation.checks_merged import check_merged


def run(cfg: Config, state: RunState, args) -> str:
    sdir = cfg.abs(state.staging)
    merged = read_table(sdir / "trends_weather")
    meta = read_json(sdir / "meta.json")
    wdir = cfg.abs(state.raw["weather"])
    wman = read_json(wdir / "manifest.json")
    daily = pd.read_csv(wdir / wman["file"], parse_dates=["date"])

    v = cfg["validation"]
    results = check_merged(merged, meta, daily, meta["lags"], v["seed"], v["lag_check_weeks_per_province"])
    write_audit(cfg, state.run_id, "validate-merged", results)
    state.rules["validate-merged"] = summarize(results)
    state.save(cfg)
    raise_if_blocked("validate-merged", results)
    s = state.rules["validate-merged"]
    return f"ผ่าน {s['pass']} ข้อ, ข้าม {s['skip']} ข้อ"
