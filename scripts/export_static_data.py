"""
export_static_data.py
ส่งออกผลวิเคราะห์ทั้งหมดจาก DuckDB ไปเป็น JSON ที่ reports/static/data.json
สำหรับ dashboard HTML แบบ static (ไม่ต้องการ DuckDB ตอน render)

ใช้: source .venv/bin/activate && python scripts/export_static_data.py
"""

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "pipeline.duckdb"
OUT_DIR = ROOT / "reports" / "static"
OUT_PATH = OUT_DIR / "data.json"

MAIN_FOLDS = ["2019", "2022", "2023", "2024", "2025"]

# ---------------------------------------------------------------------------
# helper: แปลง NaN/Inf → None ก่อน json.dumps
# ---------------------------------------------------------------------------
def _clean(val):
    if val is None:
        return None
    if isinstance(val, float):
        if val != val:   # NaN
            return None
        if val == float("inf") or val == float("-inf"):
            return None
    return val


def _row(d: dict) -> dict:
    return {k: _clean(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# 1. meta
# ---------------------------------------------------------------------------
def get_meta(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute(
        "SELECT data_version, written_at FROM ops.data_version ORDER BY written_at DESC LIMIT 1"
    ).fetchone()
    data_version = row[0] if row else "unknown"

    cnt = con.execute(
        "SELECT count(*), min(\"Week start\"), max(\"Week start\") FROM mart.trends_weather_latest"
    ).fetchone()
    n_rows = cnt[0]
    date_min = cnt[1].strftime("%Y-%m-%d") if hasattr(cnt[1], "strftime") else str(cnt[1])
    date_max = cnt[2].strftime("%Y-%m-%d") if hasattr(cnt[2], "strftime") else str(cnt[2])

    n_prov = con.execute(
        'SELECT count(DISTINCT "P-code") FROM mart.trends_weather_latest'
    ).fetchone()[0]
    n_weeks = con.execute(
        'SELECT count(DISTINCT "Week start") FROM mart.trends_weather_latest'
    ).fetchone()[0]

    return {
        "data_version": data_version,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_rows": n_rows,
        "n_provinces": n_prov,
        "n_weeks": n_weeks,
        "date_range": [date_min, date_max],
    }


# ---------------------------------------------------------------------------
# 2. data_sources — สถิติดิบจาก DuckDB สำหรับ section "แหล่งข้อมูล"
# ---------------------------------------------------------------------------
def get_data_sources(con: duckdb.DuckDBPyConnection) -> dict:
    r = con.execute(
        """
        SELECT
          strftime(MIN("Week start"), '%Y-%m-%d')  AS date_min,
          strftime(MAX("Week start"), '%Y-%m-%d')  AS date_max,
          COUNT(*)                                  AS n_rows,
          COUNT(DISTINCT "P-code")                  AS n_prov,
          COUNT(DISTINCT "Year")                    AS n_years,
          SUM("Dengue total")                       AS total_dengue,
          ROUND(AVG("Dengue rate per 100,000"), 3)  AS avg_rate,
          COUNT(CASE WHEN "Dengue total" = 0 THEN 1 END) AS zero_rows,
          ROUND(MIN(rain_wk_mm),   1)  AS rain_min,
          ROUND(MAX(rain_wk_mm),   1)  AS rain_max,
          ROUND(AVG(rain_wk_mm),   1)  AS rain_mean,
          ROUND(MIN(tmax_mean_c),  1)  AS tmax_min,
          ROUND(MAX(tmax_mean_c),  1)  AS tmax_max,
          ROUND(AVG(tmax_mean_c),  1)  AS tmax_mean,
          ROUND(MIN(tmin_mean_c),  1)  AS tmin_min,
          ROUND(MAX(tmin_mean_c),  1)  AS tmin_max,
          ROUND(AVG(tmin_mean_c),  1)  AS tmin_mean,
          MIN("Population")            AS pop_min,
          MAX("Population")            AS pop_max,
          COUNT(DISTINCT "Month")      AS n_months,
          COUNT(DISTINCT "Epi week")   AS n_epi_weeks
        FROM mart.trends_weather_latest
        """
    ).fetchone()

    keys = [
        "date_min", "date_max", "n_rows", "n_prov", "n_years",
        "total_dengue", "avg_rate", "zero_rows",
        "rain_min", "rain_max", "rain_mean",
        "tmax_min", "tmax_max", "tmax_mean",
        "tmin_min", "tmin_max", "tmin_mean",
        "pop_min", "pop_max", "n_months", "n_epi_weeks",
    ]
    merged = {k: _clean(v) for k, v in zip(keys, r)}

    # weather raw rows = n_prov × n_days
    n_weather_rows = con.execute(
        "SELECT COUNT(*) FROM core.weather_daily_latest"
    ).fetchone()[0] if _table_exists(con, "core", "weather_daily_latest") else None

    return {
        "trends": {
            "source":        "ระบบรายงานโรค 506 กรมควบคุมโรค กระทรวงสาธารณสุข",
            "file":          "TRENDS.xlsx (2 ชีท: รายสัปดาห์ ISO + รายปี)",
            "diseases":      ["ไข้เลือดออก (Dengue)", "ชิคุนกุนยา", "โรคมือเท้าปาก"],
            "unit_raw":      "ผู้ป่วย/สัปดาห์/จังหวัด (absolute count)",
            "unit_analysis": "อัตราป่วยต่อ 100,000 ประชากร",
            "pop_source":    "POPC_YYYY ในไฟล์ TRENDS.xlsx (อัปเดตรายปี)",
            "pop_min":       merged["pop_min"],
            "pop_max":       merged["pop_max"],
        },
        "weather": {
            "source":    "ERA5 reanalysis ผ่าน Open-Meteo API",
            "fetch_script": "scripts/weather_fetch.py",
            "location":  "Centroid พิกัดของแต่ละจังหวัด (77 จุด) — ไม่ใช่สถานีตรวจวัดจริง",
            "variables": ["rain_wk_mm (ฝนสะสมรายสัปดาห์)", "tmax_mean_c (°C)", "tmin_mean_c (°C)"],
            "unit_rain":  "mm/day (ดิบ) → mm/week หลัง sum 7 วัน",
            "unit_temp":  "°C (mean 7 วัน)",
            "n_rows_raw": n_weather_rows,
            "note":       "ERA5 เป็น reanalysis ไม่ได้ตรวจสอบกับข้อมูล TMD",
            "rain_min":   merged["rain_min"],
            "rain_max":   merged["rain_max"],
            "rain_mean":  merged["rain_mean"],
            "tmax_min":   merged["tmax_min"],
            "tmax_max":   merged["tmax_max"],
            "tmax_mean":  merged["tmax_mean"],
            "tmin_min":   merged["tmin_min"],
            "tmin_max":   merged["tmin_max"],
            "tmin_mean":  merged["tmin_mean"],
        },
        "merged": {
            "join_keys":    ["P-code (รหัสจังหวัด 2 หลัก)", "Week start (วันจันทร์ ISO)"],
            "filter":       "week_days = 7 (สัปดาห์เต็มเท่านั้น)",
            "n_rows":       merged["n_rows"],
            "n_prov":       merged["n_prov"],
            "n_years":      merged["n_years"],
            "date_min":     merged["date_min"],
            "date_max":     merged["date_max"],
            "total_dengue": merged["total_dengue"],
            "avg_rate":     merged["avg_rate"],
            "zero_rows":    merged["zero_rows"],
        },
    }


def _table_exists(con, schema: str, table: str) -> bool:
    r = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema=? AND table_name=?",
        [schema, table],
    ).fetchone()
    return r[0] > 0


# ---------------------------------------------------------------------------
# 3. EDA by year (pandas-side aggregation เพื่อหลีกเลี่ยง INT128 bug ใน DuckDB)
# ---------------------------------------------------------------------------
def get_eda_by_year(con: duckdb.DuckDBPyConnection) -> list:
    df = con.execute(
        'SELECT "Year", "Dengue total", "Dengue rate per 100,000", rain_wk_mm, tmax_mean_c '
        'FROM mart.trends_weather_latest WHERE week_days = 7'
    ).df()

    agg = (
        df.groupby("Year")
        .agg(
            n_rows=("Year", "count"),
            total_cases=("Dengue total", "sum"),
            avg_rate=("Dengue rate per 100,000", "mean"),
            avg_rain=("rain_wk_mm", "mean"),
            avg_tmax=("tmax_mean_c", "mean"),
        )
        .reset_index()
    )

    rows = []
    for _, r in agg.iterrows():
        rows.append(
            _row(
                {
                    "year": int(r["Year"]),
                    "n_rows": int(r["n_rows"]),
                    "total_cases": int(r["total_cases"]),
                    "avg_rate": round(float(r["avg_rate"]), 4),
                    "avg_rain": round(float(r["avg_rain"]), 4) if pd.notna(r["avg_rain"]) else None,
                    "avg_tmax": round(float(r["avg_tmax"]), 4) if pd.notna(r["avg_tmax"]) else None,
                }
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 3. H1/H2 results (จาก mart.h_results)
# ---------------------------------------------------------------------------
def get_h_corr_data(con: duckdb.DuckDBPyConnection, var: str) -> list:
    df = con.execute(
        "SELECT fold, lag, median_rho, median_abs_rho FROM mart.h_results "
        "WHERE var = ? ORDER BY fold, lag",
        [var],
    ).df()

    rows = []
    for _, r in df.iterrows():
        rows.append(
            _row(
                {
                    "fold": str(r["fold"]),
                    "lag": int(r["lag"]),
                    "median_rho": round(float(r["median_rho"]), 6) if pd.notna(r["median_rho"]) else None,
                    "median_abs_rho": round(float(r["median_abs_rho"]), 6) if pd.notna(r["median_abs_rho"]) else None,
                }
            )
        )
    return rows


def get_h_decision(decisions_json: dict, var: str) -> bool:
    return bool(decisions_json.get(var, {}).get("passed", False))


# ---------------------------------------------------------------------------
# 4. H4 results (จาก mart.results)  — latest run per (fold, model, feature_set)
# ---------------------------------------------------------------------------
def get_h4_results(con: duckdb.DuckDBPyConnection) -> dict:
    df = con.execute(
        "SELECT * FROM mart.results ORDER BY created_at DESC"
    ).df()

    # dedup: เอา row แรกต่อ (fold, model, feature_set) = latest
    df_dedup = df.drop_duplicates(subset=["fold", "model", "feature_set"], keep="first")

    main_df = df_dedup[df_dedup["fold"].astype(str).isin(MAIN_FOLDS)].copy()

    # ----------------------------------------------------------------
    # สร้าง results list: จับคู่ M0 กับ M1 ต่อ (fold, model)
    # ต้องการเฉพาะ row ที่มี delta_mae_pct (แถว M1) และดึง mae_m0 จากแถว M0
    # ----------------------------------------------------------------
    m0 = main_df[main_df["feature_set"] == "M0"].set_index(["fold", "model"])["mae"]
    m1_rows = main_df[main_df["feature_set"] == "M1"].copy()

    results = []
    for _, r in m1_rows.iterrows():
        key = (r["fold"], r["model"])
        mae_m0 = float(m0.get(key, float("nan")))
        mae_m1 = float(r["mae"])
        delta_pct = (
            round(float(r["delta_mae_pct"]) * 100, 4) if pd.notna(r["delta_mae_pct"]) else None
        )
        ci_lower = (
            round(float(r["ci_lower"]) * 100, 4) if pd.notna(r["ci_lower"]) else None
        )
        ci_upper = (
            round(float(r["ci_upper"]) * 100, 4) if pd.notna(r["ci_upper"]) else None
        )
        ci_method = r["ci_method"] if pd.notna(r["ci_method"]) else None

        # H4 pass: delta_pct >= 5 (อากาศช่วยลด MAE ≥ 5%) และ CI ล่างไม่คร่อมศูนย์
        passed = (
            delta_pct is not None
            and ci_lower is not None
            and delta_pct >= 5.0
            and ci_lower > 0.0
        )

        results.append(
            _row(
                {
                    "fold": str(r["fold"]),
                    "model": str(r["model"]),
                    "feature_set": str(r["feature_set"]),
                    "mae_m0": round(mae_m0, 6) if mae_m0 == mae_m0 else None,
                    "mae_m1": round(mae_m1, 6),
                    "delta_pct": delta_pct,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_method": ci_method,
                    "pass": passed,
                }
            )
        )

    # sensitivity: โมเดลอื่น (ทุก fold รวม supp) สำหรับ M1 เทียบ M0
    sensitivity = []
    for _, r in df_dedup[df_dedup["feature_set"] == "M1"].iterrows():
        key = (r["fold"], r["model"])
        mae_m0_s = float(df_dedup[
            (df_dedup["fold"] == r["fold"]) &
            (df_dedup["model"] == r["model"]) &
            (df_dedup["feature_set"] == "M0")
        ]["mae"].values[0]) if len(df_dedup[
            (df_dedup["fold"] == r["fold"]) &
            (df_dedup["model"] == r["model"]) &
            (df_dedup["feature_set"] == "M0")
        ]) > 0 else float("nan")

        delta_pct_s = (
            round(float(r["delta_mae_pct"]) * 100, 4) if pd.notna(r["delta_mae_pct"]) else None
        )
        sensitivity.append(
            _row(
                {
                    "fold": str(r["fold"]),
                    "model": str(r["model"]),
                    "mae_m0": round(mae_m0_s, 6) if mae_m0_s == mae_m0_s else None,
                    "mae_m1": round(float(r["mae"]), 6),
                    "delta_pct": delta_pct_s,
                }
            )
        )

    # H4 decision: HuberRegressor ผ่าน ≥ 4/5 main folds
    huber_results = [r for r in results if r["model"] == "HuberRegressor"]
    n_pass = sum(1 for r in huber_results if r.get("pass", False))
    h4_decision = n_pass >= 4

    return {
        "decision": h4_decision,
        "primary_model": "HuberRegressor",
        "threshold_pct": 5.0,
        "pass_folds_min": 4,
        "n_pass_folds": n_pass,
        "results": results,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"[export_static_data] เชื่อมต่อ DuckDB ที่ {DB_PATH}")
    sys.stdout.flush()

    if not DB_PATH.exists():
        print(f"ERROR: ไม่พบไฟล์ {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # --- meta ---
    print("  → meta")
    sys.stdout.flush()
    meta = get_meta(con)

    # --- data sources ---
    print("  → data sources")
    sys.stdout.flush()
    data_sources = get_data_sources(con)

    # --- EDA ---
    print("  → EDA by year")
    sys.stdout.flush()
    eda_by_year = get_eda_by_year(con)

    # --- H decisions ---
    print("  → H decisions")
    sys.stdout.flush()
    dec_row = con.execute(
        "SELECT decisions_json FROM mart.h_decisions ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    decisions_json = json.loads(dec_row[0]) if dec_row else {}

    # --- H1/H2 ---
    print("  → H1/H2 results (rain, tmax, tmin)")
    sys.stdout.flush()

    h1_rain = {
        "decision": get_h_decision(decisions_json, "rain"),
        "threshold": 0.10,
        "data": get_h_corr_data(con, "rain"),
    }
    h2_tmax = {
        "decision": get_h_decision(decisions_json, "tmax"),
        "threshold": 0.10,
        "data": get_h_corr_data(con, "tmax"),
    }
    h2_tmin = {
        "decision": get_h_decision(decisions_json, "tmin"),
        "threshold": 0.10,
        "data": get_h_corr_data(con, "tmin"),
    }

    # --- H3 ---
    h3_raw = decisions_json.get("H3", {})
    h3_eligible = bool(h3_raw.get("eligible", False))
    h3 = {
        "eligible": h3_eligible,
        "reason": "H1/H2 ไม่ผ่านทั้งคู่" if not h3_eligible else "H1 หรือ H2 ผ่าน",
    }

    # --- H4 ---
    print("  → H4 results")
    sys.stdout.flush()
    h4 = get_h4_results(con)

    con.close()

    # --- รวม output ---
    output = {
        "meta": meta,
        "data_sources": data_sources,
        "eda_by_year": eda_by_year,
        "h1_rain": h1_rain,
        "h2_tmax": h2_tmax,
        "h2_tmin": h2_tmin,
        "h3": h3,
        "h4": h4,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    # --- สรุป ---
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\n[export_static_data] เขียนสำเร็จ → {OUT_PATH}")
    print(f"  ขนาดไฟล์       : {size_kb:.1f} KB")
    print(f"  data_version   : {meta['data_version']}")
    print(f"  generated_at   : {meta['generated_at']}")
    print(f"  n_rows         : {meta['n_rows']:,}")
    print(f"  EDA years      : {len(eda_by_year)}")
    print(f"  H1 rain data   : {len(h1_rain['data'])} rows | decision={h1_rain['decision']}")
    print(f"  H2 tmax data   : {len(h2_tmax['data'])} rows | decision={h2_tmax['decision']}")
    print(f"  H2 tmin data   : {len(h2_tmin['data'])} rows | decision={h2_tmin['decision']}")
    print(f"  H3 eligible    : {h3['eligible']}")
    print(f"  H4 decision    : {h4['decision']} | pass_folds={h4['n_pass_folds']}/5")
    print(f"  H4 results     : {len(h4['results'])} rows (main folds M1)")
    print(f"  H4 sensitivity : {len(h4['sensitivity'])} rows (all folds M1)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
