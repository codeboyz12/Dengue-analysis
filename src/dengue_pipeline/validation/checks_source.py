"""กฎชั้น 1: ตรวจระดับแหล่ง (SRC, TRD, WX)

ชื่อคอลัมน์ของ TRENDS อ้างอิงจาก trends_weather_merged.csv (ยืนยันกับไฟล์จริง)
กฎที่ต้องใช้คอลัมน์ที่ไม่มีในตาราง จะบันทึกเป็น skip ไม่ใช่ผ่าน
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from dengue_pipeline.rules import RuleResult, check, info, skip

# ---- สัญญาของแหล่ง (SRC-01) ------------------------------------------------
TRENDS_REQUIRED = {
    "Year": "int", "Epi week": "int", "Week start": "date", "Week end": "date",
    "P-code": "str", "Province": "str",
    "Dengue fever (DF)": "int", "Dengue haemorrhagic fever (DHF)": "int",
    "Dengue shock syndrome (DSS)": "int", "Dengue total": "int",
    "Chikungunya": "int", "HFMD": "int",
}
WEATHER_REQUIRED = {
    "P-code": "str", "Province": "str", "date": "date",
    "precip_mm": "float", "t_max_c": "float", "t_min_c": "float",
}
COUNT_COLS = ["Dengue fever (DF)", "Dengue haemorrhagic fever (DHF)",
              "Dengue shock syndrome (DSS)", "Dengue total", "Chikungunya", "HFMD"]
RATE_PAIRS = [("Dengue rate per 100,000", "Dengue total"),
              ("Chikungunya rate per 100,000", "Chikungunya"),
              ("HFMD rate per 100,000", "HFMD")]
WX_VARS = ["precip_mm", "t_max_c", "t_min_c"]


def _kind_ok(s: pd.Series, kind: str) -> bool:
    if kind == "int":
        return pdt.is_integer_dtype(s)
    if kind == "float":
        return pdt.is_float_dtype(s) or pdt.is_integer_dtype(s)
    if kind == "date":
        return pdt.is_datetime64_any_dtype(s)
    if kind == "str":
        return pdt.is_string_dtype(s) or s.dtype == object
    return False


def schema_rule(df: pd.DataFrame, required: dict, label: str) -> RuleResult:
    missing = [c for c in required if c not in df.columns]
    bad = [c for c, k in required.items() if c in df.columns and not _kind_ok(df[c], k)]
    ok = not missing and not bad
    return check("SRC-01", ok, f"{label}: ขาด {len(missing)} คอลัมน์, ชนิดผิด {len(bad)}",
                 f"missing={missing} wrong_type={bad}" if not ok else "")


def key_rule(df: pd.DataFrame, keys: list[str], label: str) -> RuleResult:
    if any(k not in df.columns for k in keys):
        return skip("SRC-02", f"{label}: ไม่มีคอลัมน์คีย์ {keys}")
    dup = int(df.duplicated(keys).sum())
    return check("SRC-02", dup == 0, f"{label} {keys}: ซ้ำ {dup} แถว")


def rowcount_rule(n: int, prev: int | None, label: str) -> RuleResult:
    if prev is None:
        return check("SRC-03", True, f"{label}: {n} แถว", "ยังไม่มีรอบก่อนให้เทียบ")
    return check("SRC-03", n >= prev, f"{label}: {n} แถว (รอบก่อน {prev})")


# ---- TRENDS ------------------------------------------------------------------
def _week_days(df: pd.DataFrame) -> pd.Series:
    return (df["Week end"] - df["Week start"]).dt.days + 1


def check_trends(df: pd.DataFrame, expected_provinces: int, rate_abs_tol: float,
                 prev_rows: int | None) -> list[RuleResult]:
    out = [schema_rule(df, TRENDS_REQUIRED, "TRENDS"),
           key_rule(df, ["P-code", "Year", "Epi week"], "TRENDS"),
           key_rule(df, ["P-code", "Week start"], "TRENDS"),
           rowcount_rule(len(df), prev_rows, "TRENDS")]
    if out[0].outcome == "fail":
        # ไม่มีคอลัมน์พื้นฐาน กฎที่เหลือรันไม่ได้อย่างมีความหมาย
        return out + [skip(r, "สัญญาคอลัมน์ไม่ผ่าน (SRC-01)")
                      for r in ["TRD-01", "TRD-02", "TRD-03", "TRD-04", "TRD-05", "TRD-06"]]

    # TRD-01 ครบ 77 จังหวัด และทุกจังหวัดมีจำนวนสัปดาห์เท่ากัน
    per = df.groupby("P-code").size()
    n_prov = len(per)
    equal = per.nunique() == 1
    out.append(check("TRD-01", n_prov == expected_provinces and equal,
                     f"{len(per)} จังหวัด (คาด {expected_provinces}), สัปดาห์ต่อจังหวัด "
                     f"{sorted(set(per.tolist()))[:5]}"))

    # TRD-02 จำนวนผู้ป่วย >= 0
    neg = {c: int((df[c] < 0).sum()) for c in COUNT_COLS if (df[c] < 0).any()}
    out.append(check("TRD-02", not neg, f"ค่าติดลบ: {neg or 'ไม่มี'}"))

    # TRD-03 total = DF + DHF + DSS
    calc = (df["Dengue fever (DF)"] + df["Dengue haemorrhagic fever (DHF)"]
            + df["Dengue shock syndrome (DSS)"])
    bad3 = int((calc != df["Dengue total"]).sum())
    out.append(check("TRD-03", bad3 == 0, f"แถวที่ผลรวมไม่ตรง {bad3}"))

    # TRD-04 อัตราต่อแสนคำนวณซ้ำแล้วตรง
    if "Population" not in df.columns:
        out.append(skip("TRD-04", "ไม่มีคอลัมน์ Population (ตาราง weekly sheet ยังไม่รวมกับ province sheet)"))
    else:
        worst, used = 0.0, []
        for rate_col, cnt_col in RATE_PAIRS:
            if rate_col not in df.columns:
                continue
            used.append(rate_col)
            diff = (df[rate_col] - df[cnt_col] / df["Population"] * 1e5).abs()
            worst = max(worst, float(np.nanmax(diff.to_numpy())) if len(diff) else 0.0)
        if not used:
            out.append(skip("TRD-04", "ไม่มีคอลัมน์อัตราต่อแสนในตาราง"))
        else:
            out.append(check("TRD-04", worst <= rate_abs_tol,
                             f"คลาดเคลื่อนสูงสุด {worst:.5f} (เกณฑ์ {rate_abs_tol}) จาก {len(used)} อัตรา"))

    # TRD-05 week_days 1-7 และสัปดาห์ไม่ซ้อนกัน
    wd = _week_days(df)
    in_range = bool(wd.between(1, 7).all())
    weeks = df[["Week start", "Week end"]].drop_duplicates().sort_values("Week start")
    overlap = int((weeks["Week start"].iloc[1:].to_numpy() <= weeks["Week end"].iloc[:-1].to_numpy()).sum())
    out.append(check("TRD-05", in_range and overlap == 0,
                     f"week_days นอก 1-7: {int((~wd.between(1, 7)).sum())} แถว, สัปดาห์ซ้อนกัน {overlap} คู่"))

    # TRD-06 สัปดาห์สั้นเกิดเฉพาะที่ขอบปีปฏิทิน (เริ่ม 1 ม.ค. หรือจบ 31 ธ.ค.)
    short = weeks[(weeks["Week end"] - weeks["Week start"]).dt.days + 1 < 7]
    edge = ((short["Week start"].dt.month == 1) & (short["Week start"].dt.day == 1)) | \
           ((short["Week end"].dt.month == 12) & (short["Week end"].dt.day == 31))
    off = short[~edge]
    out.append(check("TRD-06", len(off) == 0,
                     f"สัปดาห์สั้น {len(short)} สัปดาห์, อยู่นอกขอบปี {len(off)}",
                     f"ตัวอย่าง: {off['Week start'].dt.date.astype(str).tolist()[:3]}" if len(off) else ""))
    return out


# ---- อากาศรายวัน ------------------------------------------------------------
def check_weather(df: pd.DataFrame, expected_provinces: int,
                  window: dict | None, prev_rows: int | None) -> list[RuleResult]:
    out = [schema_rule(df, WEATHER_REQUIRED, "weather"),
           key_rule(df, ["P-code", "date"], "weather"),
           rowcount_rule(len(df), prev_rows, "weather")]
    if out[0].outcome == "fail":
        return out + [skip(r, "สัญญาคอลัมน์ไม่ผ่าน (SRC-01)") for r in ["WX-01", "WX-02", "WX-03", "WX-04"]]

    # WX-01 ฝน >= 0 และ t_max >= t_min รายวัน
    neg_rain = int((df["precip_mm"] < 0).sum())
    inverted = int((df["t_max_c"] < df["t_min_c"]).sum())
    out.append(check("WX-01", neg_rain == 0 and inverted == 0,
                     f"ฝนติดลบ {neg_rain} แถว, t_max<t_min {inverted} แถว"))

    # WX-02 จำนวนแถว = จังหวัด x จำนวนวัน และไม่มีวันขาด
    if window is None:
        out.append(skip("WX-02", "ไม่มีขอบเขตวันที่ใน manifest"))
    else:
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        n_days = (end - start).days + 1
        per = df.groupby("P-code")["date"].agg(["count", "min", "max"])
        full = ((per["count"] == n_days) & (per["min"] == start) & (per["max"] == end)).all()
        n_prov = len(per)
        out.append(check("WX-02", bool(full) and n_prov == expected_provinces and len(df) == n_prov * n_days,
                         f"{len(df)} แถว (คาด {expected_provinces}x{n_days}={expected_provinces * n_days}), "
                         f"{n_prov} จังหวัด, ช่วง {start.date()}..{end.date()}",
                         f"จังหวัดที่ช่วงวันไม่ครบ: {int((~((per['count'] == n_days) & (per['min'] == start) & (per['max'] == end))).sum())}"))

    # WX-03 ไม่มีค่าว่างรายจุด
    nan = {c: int(df[c].isna().sum()) for c in WX_VARS if df[c].isna().any()}
    out.append(check("WX-03", not nan, f"ค่าว่าง: {nan or 'ไม่มี'}"))

    # WX-04 ช่วงค่า (INFO: ยังไม่มีเกณฑ์ บันทึกค่าไว้เพื่อตั้งเกณฑ์จากข้อมูลจริง)
    rng = {c: (round(float(df[c].min()), 2), round(float(df[c].max()), 2)) for c in WX_VARS}
    out.append(info("WX-04", rng, "ยังไม่ล็อกเกณฑ์ บันทึกช่วงค่าเพื่อใช้ตั้งเกณฑ์"))
    return out


# ---- ขอบเขตวันที่ของอากาศ (ใช้ร่วมกับ ingest) -------------------------------
def weather_window(weekly: pd.DataFrame, buffer_days: int, end_lag_days: int,
                   today: dt.date | None = None) -> dict:
    """วันเริ่ม = Week start แรก - buffer, วันจบ = min(Week end สุดท้าย, วันนี้ - end_lag_days)"""
    today = today or dt.date.today()
    start = weekly["Week start"].min().date() - dt.timedelta(days=buffer_days)
    end = min(weekly["Week end"].max().date(), today - dt.timedelta(days=end_lag_days))
    return {"start": start.isoformat(), "end": end.isoformat()}
