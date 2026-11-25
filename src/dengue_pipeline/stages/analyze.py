"""stages/analyze.py — CLI wrapper สำหรับ H1/H2/H3 hypothesis testing

เขียนผลลง mart.h_results (rho + placebo) และ mart.h_decisions (decisions)
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import duckdb
import pandas as pd

from dengue_pipeline.config import Config


def run(cfg: Config, cfg_analysis: dict, run_id: str | None = None) -> str:
    from dengue_pipeline.sink_duckdb import DuckDBSink
    db_path = DuckDBSink(cfg).path

    con_r = duckdb.connect(str(db_path), read_only=True)
    try:
        from dengue_pipeline.hypotheses import run_h_analysis
        rho_df, placebo_df, decisions = run_h_analysis(con_r, cfg_analysis)
    finally:
        con_r.close()

    created_at = datetime.datetime.now().isoformat(timespec="seconds")

    rho_df["created_at"] = created_at
    placebo_df["created_at"] = created_at if not placebo_df.empty else None

    con_w = duckdb.connect(str(db_path))
    try:
        con_w.execute("""
            CREATE TABLE IF NOT EXISTS mart.h_results (
                fold        VARCHAR,
                var         VARCHAR,
                lag         INTEGER,
                analysis    VARCHAR,
                median_rho  DOUBLE,
                median_abs_rho DOUBLE,
                n_prov      INTEGER,
                created_at  VARCHAR
            )
        """)
        con_w.execute("""
            CREATE TABLE IF NOT EXISTS mart.h_placebo (
                fold            VARCHAR,
                var             VARCHAR,
                lag             INTEGER,
                rho_real        DOUBLE,
                placebo_median  DOUBLE,
                placebo_p95     DOUBLE,
                n_placebo       INTEGER,
                created_at      VARCHAR
            )
        """)
        con_w.execute("""
            CREATE TABLE IF NOT EXISTS mart.h_decisions (
                created_at  VARCHAR,
                decisions_json VARCHAR
            )
        """)

        if not rho_df.empty:
            con_w.execute("""
                INSERT INTO mart.h_results
                SELECT fold, var, lag, analysis, median_rho, median_abs_rho, n_prov, created_at
                FROM rho_df
            """)

        if not placebo_df.empty:
            con_w.execute("""
                INSERT INTO mart.h_placebo
                SELECT fold, var, lag, rho_real, placebo_median, placebo_p95, n_placebo, created_at
                FROM placebo_df
            """)

        con_w.execute(
            "INSERT INTO mart.h_decisions VALUES (?, ?)",
            [created_at, json.dumps(decisions, ensure_ascii=False)]
        )
        con_w.commit()
    finally:
        con_w.close()

    n_rho = len(rho_df)
    h1 = {v: decisions.get(v, {}) for v in cfg_analysis["hypotheses"]["H1"]["vars"]}
    h2 = {v: decisions.get(v, {}) for v in cfg_analysis["hypotheses"]["H2"]["vars"]}
    h3 = decisions.get("H3", {})

    summary = (
        f"เขียน mart.h_results {n_rho} แถว, mart.h_placebo {len(placebo_df)} แถว\n"
        f"  H1 (rain):  {'✅ ผ่าน' if h1.get('rain', {}).get('passed') else '❌ ไม่ผ่าน'}"
        f" (lag ผ่าน: {h1.get('rain', {}).get('pass_lags', [])})\n"
        f"  H2 (tmax):  {'✅ ผ่าน' if h2.get('tmax', {}).get('passed') else '❌ ไม่ผ่าน'}"
        f" (lag ผ่าน: {h2.get('tmax', {}).get('pass_lags', [])})\n"
        f"  H2 (tmin):  {'✅ ผ่าน' if h2.get('tmin', {}).get('passed') else '❌ ไม่ผ่าน'}"
        f" (lag ผ่าน: {h2.get('tmin', {}).get('pass_lags', [])})\n"
        f"  H3: eligible={h3.get('eligible')} "
        + (f"best_lag rain={h3.get('rain', {}).get('best_lag')} "
           f"tmax={h3.get('tmax', {}).get('best_lag')} "
           f"tmin={h3.get('tmin', {}).get('best_lag')}"
           if h3.get("eligible") else "ตัดสินไม่ได้")
    )
    return summary
