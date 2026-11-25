"""ฟีเจอร์จากประวัติของโรคและป้าย D สำหรับ CSV ที่ส่งออก

คำนวณเหมือน _province_block ใน notebook: อัตรา = ผลรวมจำนวนผู้ป่วยในหน้าต่าง / ประชากร ณ สัปดาห์ t * per
นับเฉพาะสัปดาห์เต็ม (week_days == 7) และหน้าต่างต้องครอบคลุมช่วงวันที่ต่อเนื่องสัปดาห์ละ 7 วันพอดี
ไม่เช่นนั้นเป็น NaN

ส่วนล่าง: build_features() (SQL / DuckDB version) สำหรับ train subcommand
SQL logic เหมือน docs/notebook_snippets/build_features_sql.py ทุกจุด
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_pipeline.contract import Contract

# ชื่อคอลัมน์อากาศดิบ → key ชื่อสั้น (เหมือนใน notebook snippets)
WX_BASE = {"rain": "rain_wk_mm", "tmax": "tmax_mean_c", "tmin": "tmin_mean_c"}

WEEK_DAYS = 7


def _sorted(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    return df.sort_values(contract.keys, kind="mergesort")


def window_sum(df: pd.DataFrame, col: str, lo: int, hi: int) -> pd.Series:
    """ผลรวมของ col ที่สัปดาห์ t+lo .. t+hi ของแต่ละจังหวัด (df ต้องเรียงตามจังหวัด แล้วสัปดาห์)
    เป็น NaN ถ้าหน้าต่างหลุดขอบของจังหวัดหรือมี NaN ในหน้าต่าง (ไม่เติม 0 ให้)"""
    g = df.groupby("P-code")[col]
    parts = [g.shift(-k) for k in range(lo, hi + 1)]      # shift(-k) = ค่าของสัปดาห์ t+k
    return pd.concat(parts, axis=1).sum(axis=1, skipna=False)


def window_rate(df: pd.DataFrame, source: str, lo: int, hi: int, rate: dict) -> pd.Series:
    """อัตราของหน้าต่าง t+lo..t+hi: ผลรวมผู้ป่วย (เฉพาะสัปดาห์เต็ม) / ประชากรที่ t * per

    NaN เมื่อ: หน้าต่างหลุดขอบ, มีสัปดาห์สั้นในหน้าต่าง, หรือ Week start ไม่ห่างกันสัปดาห์ละ 7 วันพอดี
    ระหว่างขอบหน้าต่างกับสัปดาห์ t (เทียบกับ ok_b3 / ok_f4 ใน notebook)
    """
    lo_b, hi_b = min(lo, 0), max(hi, 0)                   # ช่วงที่ต้องต่อเนื่อง: ครอบทั้งหน้าต่างและสัปดาห์ t
    full = df[rate["full_week_col"]].eq(rate["full_week_value"])
    work = pd.DataFrame({"P-code": df["P-code"], "c": df[source].where(full).astype(float)}, index=df.index)
    total = window_sum(work, "c", lo, hi)
    ws = df.groupby("P-code")["Week start"]
    span = ws.shift(-hi_b) - ws.shift(-lo_b)
    ok = span.dt.days.eq(WEEK_DAYS * (hi_b - lo_b))
    return total.where(ok) / df[rate["population"]].astype(float) * rate["per"]


def add_history_features(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """เพิ่มฟีเจอร์จากประวัติ (ใช้เฉพาะสัปดาห์ <= t) คืนตารางที่เรียงตามคีย์ index เดิมคงไว้"""
    out = _sorted(df, contract).copy()
    for name, spec in contract.history_features.items():
        lo, hi = spec["window"]
        out[name] = window_rate(out, spec["source"], lo, hi, contract.rate)
    return out


def add_label(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """ป้าย D = log1p(R_next) - log1p(R_prev) ตามหน้าต่างใน contract"""
    lab = contract.label
    out = _sorted(df, contract).copy()
    past = window_rate(out, lab["source"], *lab["past_window"], contract.rate)
    future = window_rate(out, lab["source"], *lab["future_window"], contract.rate)
    out[lab["name"]] = np.log1p(future) - np.log1p(past)
    return out


def build_table(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """ตารางที่ส่งออก: คอลัมน์ตาม contract + ฟีเจอร์จากประวัติ + ป้าย (คอลัมน์สุดท้าย)"""
    return add_label(add_history_features(df, contract), contract)[contract.output_columns]


# ---------------------------------------------------------------------------
# SQL / DuckDB version สำหรับ train subcommand
# SQL logic เหมือน docs/notebook_snippets/build_features_sql.py ทุกจุด
# ---------------------------------------------------------------------------

def build_features(con, cfg_analysis: dict) -> pd.DataFrame:
    """สร้าง feature table จาก mart.trends_weather_latest (SQL/DuckDB version)

    con          = duckdb connection ที่เปิดแล้ว (read_only หรือ write ก็ได้)
    cfg_analysis = dict จาก config/analysis.toml
    คืน DataFrame พร้อมใช้สำหรับ get_fold
    """
    import duckdb as _duckdb

    MAX_WX_LAG = max(cfg_analysis["features"]["M1"]["wx_lags"])
    FOURIER_K = cfg_analysis["features"]["M0"]["seas_fourier_k"]
    TARGET = cfg_analysis["target"]["name"]

    source = "mart.trends_weather_latest"
    cols = con.execute(f"SELECT * FROM {source} LIMIT 0").df().columns
    drop = [c for c in ("data_version", "_ord") if c in cols]
    base = f"SELECT * EXCLUDE ({', '.join(drop)}) FROM {source}" if drop else f"SELECT * FROM {source}"

    # ---- ตรวจเหมือน _national_block: ทุกสัปดาห์ต้องมีจังหวัดครบ ----
    n_prov, bad = con.execute(f'''
        WITH s AS ({base}), t AS (SELECT count(DISTINCT "P-code") AS n FROM s)
        SELECT (SELECT n FROM t),
               (SELECT count(*) FROM (SELECT "Week start" FROM s GROUP BY 1
                                      HAVING count(DISTINCT "P-code") <> (SELECT n FROM t)))''').fetchone()
    assert bad == 0, "บางสัปดาห์มีจังหวัดไม่ครบ"
    regions = [r[0] for r in con.execute(
        f'SELECT DISTINCT CAST("Health region" AS INTEGER) FROM ({base}) ORDER BY 1').fetchall()]

    W = 'OVER (PARTITION BY "P-code" ORDER BY "Week start")'
    W4 = ('OVER (PARTITION BY "P-code" ORDER BY "Week start" '
          'ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)')
    dd = lambda a, b: f"date_diff('day', CAST({a} AS DATE), CAST({b} AS DATE))"
    wx = list(WX_BASE.items())

    # s1: ค่า c / w รายสัปดาห์เต็ม, ผลรวมหน้าต่าง 4 สัปดาห์, วันที่ของแถวข้างเคียง
    s1 = ['*',
          'CAST("Population" AS DOUBLE) AS pop',
          'CASE WHEN week_days = 7 THEN CAST("Dengue total" AS DOUBLE) END AS c']
    s1 += [f'CASE WHEN week_days = 7 THEN CAST("{col}" AS DOUBLE) END AS w_{key}' for key, col in wx]
    # ws_l3 และ ws_l7 จำเป็นสำหรับ s4_b3/s4_b7 เสมอ ไม่ขึ้นกับ MAX_WX_LAG
    _ws_needed = sorted(set(range(1, MAX_WX_LAG + 1)) | {3, 7})
    s1 += [f'lag("Week start", {k}) {W} AS ws_l{k}' for k in _ws_needed]
    s1 += [f'lead("Week start", 4) {W} AS ws_f4']
    s1 += [f'CASE WHEN count(CASE WHEN week_days = 7 THEN "Dengue total" END) {W4} = 4 '
           f'THEN sum(CASE WHEN week_days = 7 THEN CAST("Dengue total" AS DOUBLE) END) {W4} END AS s4']

    # s2: lag/lead ของ c, w, s4 พร้อมเงื่อนไขความต่อเนื่องของวันที่
    s2 = ['*']
    s2 += [f'CASE WHEN {dd(f"ws_l{k}", chr(34)+"Week start"+chr(34))} = {7*k} THEN lag(c, {k}) {W} END AS c_l{k}'
           for k in (1, 2, 3)]
    for key, _ in wx:
        s2 += [f'CASE WHEN {dd(f"ws_l{k}", chr(34)+"Week start"+chr(34))} = {7*k} '
               f'THEN lag(w_{key}, {k}) {W} END AS w_{key}_l{k}' for k in range(1, MAX_WX_LAG + 1)]
    s2 += [f'CASE WHEN {dd("ws_l3", chr(34)+"Week start"+chr(34))} = 21 THEN s4 END AS s4_b3',
           f'CASE WHEN {dd("ws_l7", chr(34)+"Week start"+chr(34))} = 49 THEN lag(s4, 4) {W} END AS s4_b7',
           f'CASE WHEN {dd(chr(34)+"Week start"+chr(34), "ws_f4")} = 28 THEN lead(s4, 4) {W} END AS s4_f4',
           f'lead("Week end", 4) {W} AS label_end_']

    # s3: อัตรา R และ target
    s3 = ['*', 's4_b3 / pop * 1e5 AS R_prev', 's4_b7 / pop * 1e5 AS R_prev2',
          's4_f4 / pop * 1e5 AS R_next']

    out = ['p.*', f'ln(1 + R_next) - ln(1 + R_prev) AS "{TARGET}"',
           'label_end_ AS label_end', 'R_prev AS diag_R_prev', 'R_next AS diag_R_next',
           's4_b3 + s4_f4 AS diag_cases_8w',
           'ln(1 + c / pop * 1e5) AS hist_lograte_lag0']
    out += [f'ln(1 + c_l{k} / pop * 1e5) AS hist_lograte_lag{k}' for k in (1, 2, 3)]
    out += ['ln(1 + R_prev) AS hist_logR_prev', 'ln(1 + R_prev2) AS hist_logR_prev2',
            'ln(1 + R_prev) - ln(1 + R_prev2) AS hist_growth_prev']
    for key, _ in wx:
        out += [f'w_{key} AS wk_{key}', f'w_{key} AS wx_{key}_lag0']
        out += [f'w_{key}_l{k} AS wx_{key}_lag{k}' for k in range(1, MAX_WX_LAG + 1)]
    out += ['n.nat_lograte_lag0', 'n.nat_logR_prev', 'n.nat_growth_prev']
    ang = 'pi() * 2 * dayofyear(p."Week start") / 365.25'
    for k in range(1, FOURIER_K + 1):
        out += [f'sin({k} * {ang}) AS seas_sin{k}', f'cos({k} * {ang}) AS seas_cos{k}']
    out += [f'CAST(CAST(p."Health region" AS INTEGER) = {r} AS BIGINT) AS reg_{r}' for r in regions]

    # แยก p.* ออกจากคอลัมน์ช่วย (pop, c, w_*, ws_*, s4*, ...) ให้เหลือเฉพาะคอลัมน์เดิมของ raw
    helper = (['pop', 'c', 's4', 'ws_f4', 'label_end_', 's4_b3', 's4_b7', 's4_f4', 'R_prev', 'R_prev2', 'R_next']
              + [f'ws_l{k}' for k in _ws_needed] + [f'c_l{k}' for k in (1, 2, 3)]
              + [f'w_{key}' for key, _ in wx]
              + [f'w_{key}_l{k}' for key, _ in wx for k in range(1, MAX_WX_LAG + 1)])
    out[0] = f'p.* EXCLUDE ({", ".join(helper)})'

    sql = f'''
    WITH src AS ({base}),
    s1 AS (SELECT {", ".join(s1)} FROM src),
    s2 AS (SELECT {", ".join(s2)} FROM s1),
    p  AS (SELECT {", ".join(s3)} FROM s2),
    n1 AS (SELECT "Week start" AS ws, sum(CAST("Dengue total" AS DOUBLE)) AS cases,
                  sum(CAST("Population" AS DOUBLE)) AS pop, min(week_days) AS wd
           FROM src GROUP BY 1),
    n2 AS (SELECT ws, pop, CASE WHEN wd = 7 THEN cases END AS c,
                  lag(ws, 3) OVER o AS ws_l3, lag(ws, 7) OVER o AS ws_l7 FROM n1
           WINDOW o AS (ORDER BY ws)),
    n3 AS (SELECT *, CASE WHEN count(c) OVER (ORDER BY ws ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                          THEN sum(c) OVER (ORDER BY ws ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) END AS s4
           FROM n2),
    n4 AS (SELECT ws,
                  CASE WHEN {dd("ws_l3", "ws")} = 21 THEN s4 END / pop * 1e5 AS R_prev,
                  CASE WHEN {dd("ws_l7", "ws")} = 49 THEN lag(s4, 4) OVER (ORDER BY ws) END / pop * 1e5 AS R_prev2,
                  c, pop FROM n3),
    n AS (SELECT ws AS "Week start", ln(1 + c / pop * 1e5) AS nat_lograte_lag0,
                 ln(1 + R_prev) AS nat_logR_prev, ln(1 + R_prev) - ln(1 + R_prev2) AS nat_growth_prev FROM n4)
    SELECT {", ".join(out)}
    FROM p LEFT JOIN n ON p."Week start" = n."Week start"
    ORDER BY p."P-code", p."Week start"
    '''
    return con.execute(sql).df()
