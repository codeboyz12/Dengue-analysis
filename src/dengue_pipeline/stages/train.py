"""stages/train.py — CLI stage wrapper สำหรับ train subcommand

flow:
1. คำนวณ config_hash จาก analysis.toml
2. เปิด DuckDB read_only → รัน run_experiment → ปิด
3. เปิด DuckDB write → เขียน mart.results (append-only) → ปิด
4. คืน summary string
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.sink_duckdb import DuckDBSink


def run(cfg: Config, cfg_analysis: dict, run_id: str | None = None) -> str:
    """stage: อ่าน DuckDB → experiment → เขียน mart.results

    cfg          = Config จาก pipeline.toml
    cfg_analysis = dict จาก config/analysis.toml
    run_id       = data_version ที่จะใช้ (None = ล่าสุด)
    คืน summary string
    """
    import duckdb
    from dengue_pipeline.modelling import run_experiment

    # --- 1. config_hash จาก analysis.toml ---
    analysis_path: Path | None = None
    # หา path ของ analysis.toml จาก cfg.root
    candidate = cfg.root / "config" / "analysis.toml"
    if candidate.exists():
        analysis_path = candidate
    if analysis_path is None:
        # fallback: หาใน cwd
        fallback = Path("config/analysis.toml")
        if fallback.exists():
            analysis_path = fallback
    config_hash = ""
    if analysis_path and analysis_path.exists():
        config_hash = hashlib.sha256(analysis_path.read_bytes()).hexdigest()[:12]

    # --- 2. เปิด DuckDB read_only ---
    sink = DuckDBSink(cfg)
    if not sink.path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ DuckDB: {sink.path}\n"
            "รัน 'python -m dengue_pipeline run' ก่อนเพื่อโหลดข้อมูล")

    con_ro = duckdb.connect(str(sink.path), read_only=True)
    try:
        # ตรวจว่า mart.trends_weather_latest มีข้อมูล
        try:
            n_rows = con_ro.execute(
                "SELECT count(*) FROM mart.trends_weather_latest").fetchone()[0]
        except Exception as ex:
            raise RuntimeError(
                f"mart.trends_weather_latest ไม่พบหรือยังไม่มีข้อมูล: {ex}\n"
                "รัน 'python -m dengue_pipeline run' ก่อน") from ex

        if n_rows == 0:
            raise RuntimeError("mart.trends_weather_latest ว่างเปล่า")

        # อ่าน data_version ล่าสุด
        try:
            dv = con_ro.execute(
                "SELECT max(data_version) FROM ops.data_version").fetchone()[0]
        except Exception:
            dv = ""

        print(f"[train] data_version={dv!r}  config_hash={config_hash!r}  rows={n_rows:,}")

        # --- 3. รัน experiment ---
        results_df = run_experiment(con_ro, cfg_analysis)
    finally:
        con_ro.close()

    if len(results_df) == 0:
        return "run_experiment คืนผลว่าง ไม่มีอะไรบันทึก"

    # ใส่ run_id และ config_hash
    results_df["run_id"] = dv
    results_df["config_hash"] = config_hash

    # --- 4. เปิด DuckDB write → เขียน mart.results ---
    con_rw = duckdb.connect(str(sink.path))
    try:
        con_rw.execute("CREATE SCHEMA IF NOT EXISTS mart")
        con_rw.execute("""
            CREATE TABLE IF NOT EXISTS mart.results (
                run_id VARCHAR,
                config_hash VARCHAR,
                data_version VARCHAR,
                fold VARCHAR,
                model VARCHAR,
                feature_set VARCHAR,
                mae DOUBLE,
                delta_mae_pct DOUBLE,
                ci_lower DOUBLE,
                ci_upper DOUBLE,
                ci_method VARCHAR,
                n_train INTEGER,
                n_test INTEGER,
                created_at VARCHAR
            )
        """)
        con_rw.register("_results_df", results_df)
        con_rw.execute("INSERT INTO mart.results SELECT * FROM _results_df")
        con_rw.unregister("_results_df")
    finally:
        con_rw.close()

    n_rows_written = len(results_df)
    folds_done = results_df["fold"].nunique()
    models_done = results_df["model"].nunique()
    return (
        f"เขียน mart.results {n_rows_written} แถว "
        f"({folds_done} folds × {models_done} models) "
        f"data_version={dv!r} config_hash={config_hash!r}"
    )
