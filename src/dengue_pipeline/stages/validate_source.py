"""ขั้น 2 validate-source: ตรวจแต่ละแหล่งแยกกันด้วยกฎชั้น 1 (SRC, TRD, WX)"""
from __future__ import annotations

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.io import read_json
from dengue_pipeline.rules import raise_if_blocked, summarize, write_audit
from dengue_pipeline.runstate import RunState
from dengue_pipeline.sinks import get_sink
from dengue_pipeline.trends_io import load_weekly
from dengue_pipeline.validation.checks_source import check_trends, check_weather


def run(cfg: Config, state: RunState, args) -> str:
    tdir = cfg.abs(state.raw["trends"])
    wdir = cfg.abs(state.raw["weather"])
    tman, wman = read_json(tdir / "manifest.json"), read_json(wdir / "manifest.json")

    weekly = load_weekly(tdir / tman["filename"], cfg)
    daily = pd.read_csv(wdir / wman["file"], parse_dates=["date"])

    prev = get_sink(cfg).latest_metadata()
    prev_rows = (prev or {}).get("source_rows", {})
    v = cfg["validation"]

    results = check_trends(weekly, v["expected_provinces"], v["rate_abs_tol"], prev_rows.get("trends"))
    results += check_weather(daily, v["expected_provinces"], wman.get("window"), prev_rows.get("weather"))

    write_audit(cfg, state.run_id, "validate-source", results)
    state.rules["validate-source"] = summarize(results)
    state.save(cfg)
    raise_if_blocked("validate-source", results)
    s = state.rules["validate-source"]
    return f"ผ่าน {s['pass']} ข้อ, ข้าม {s['skip']} ข้อ"
