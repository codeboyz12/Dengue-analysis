"""ขั้น 3 transform: ได้ตารางจังหวัด x สัปดาห์ที่มีอากาศและ lag ในชั้น staging

  3a รวมสองชีทของ TRENDS ตาม P-code (ประชากร/พื้นที่/centroid ตรงกับปีของแถว)
  3b รวมอากาศรายวันเป็นรายสัปดาห์ตามช่วง Week start..Week end และสร้าง lag
"""
from __future__ import annotations

import re

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.errors import StageError
from dengue_pipeline.io import read_json, write_json, write_table
from dengue_pipeline.runstate import RunState
from dengue_pipeline.trends_io import kind_of, load_province_sheet, load_weekly
from dengue_pipeline.weather_merge import attach_weather


POP_YEAR_COL = re.compile(r"^Population\s+(\d{4})$")


def province_long(prov: pd.DataFrame) -> pd.DataFrame:
    """แปลงชีท province data แบบคอลัมน์ต่อปี (Population 2016 ... 2025) เป็นหนึ่งแถวต่อจังหวัด-ปี

    ได้คอลัมน์ Year, Population และ Population density (per km²) = Population / Area (km²)
    (ตรวจแล้วตรงกับ trends_weather_merged.csv) ถ้าชีทมี Year หรือไม่มีคอลัมน์ Population YYYY ส่งคืนตามเดิม
    """
    pop_cols = {c: int(m.group(1)) for c in prov.columns if (m := POP_YEAR_COL.match(str(c)))}
    if "Year" in prov.columns or not pop_cols:
        return prov
    base = [c for c in prov.columns if c not in pop_cols]
    long = prov.melt(id_vars=base, value_vars=list(pop_cols), var_name="_col", value_name="Population")
    long["Year"] = long["_col"].map(pop_cols).astype("int64")
    long = long.drop(columns="_col")
    if "Area (km²)" in long.columns and "Population density (per km²)" not in long.columns:
        long["Population density (per km²)"] = long["Population"] / long["Area (km²)"]
    return long


def merge_trends_sheets(xlsx_path, cfg: Config) -> pd.DataFrame:
    """3a: รวมชีท weekly data กับ province data เป็นตารางเดียว (ผลเหมือน trends_merged.csv 23 คอลัมน์)

    คีย์รวม: P-code และ Year ถ้าชีท province data มีคอลัมน์ Year (ประชากรรายปี) ไม่เช่นนั้นใช้ P-code อย่างเดียว
    คอลัมน์ของชีทจังหวัดที่ซ้ำกับ weekly data (เช่น Province) ไม่ถูกเพิ่มซ้ำ
    รองรับ: หนึ่งแถวต่อจังหวัด-ปี, หรือคอลัมน์ต่อปี (Population 2016, 2017, ... แปลงด้วย province_long)
    รูปแบบอื่นหยุดพร้อมข้อความชัดเจน ไม่เดา ตรวจโครงสร้างจริงของไฟล์ด้วย: python -m dengue_pipeline inspect-trends
    จำนวนแถวหลังรวมต้องเท่ากับ weekly data และไม่มีค่าว่างในคอลัมน์จังหวัด (MRG-07 ตรวจต่อ)
    """
    weekly = load_weekly(xlsx_path, cfg)
    prov = province_long(load_province_sheet(xlsx_path, cfg))
    for name, df in (("weekly data", weekly), ("province data", prov)):
        if "P-code" not in df.columns:
            raise StageError(f"ชีท {name} ไม่มีคอลัมน์ P-code (พบ: {list(df.columns)})")
    keys = ["P-code", "Year"] if ("Year" in prov.columns and "Year" in weekly.columns) else ["P-code"]
    dup = int(prov.duplicated(keys).sum())
    if dup:
        raise StageError(
            f"ชีท province data มีคีย์ {keys} ซ้ำ {dup} แถว: ถ้าเก็บประชากรรายปีต้องมีคอลัมน์ Year "
            "ในรูปแบบหนึ่งแถวต่อจังหวัด-ปี (รูปแบบอื่นยังไม่รองรับ)")
    extra = [c for c in prov.columns if c in keys or c not in weekly.columns]
    merged = weekly.merge(prov[extra], on=keys, how="left", validate="many_to_one")
    if "Population" not in merged.columns:
        raise StageError("หลังรวมสองชีทแล้วไม่มีคอลัมน์ Population: ถ้าชีท province data เก็บประชากรแบบ "
                         "คอลัมน์ต่อปี (Population 2016 ...) ต้องแปลงเป็นหนึ่งแถวต่อจังหวัด-ปีก่อน")
    return merged


def run(cfg: Config, state: RunState, args) -> str:
    tdir = cfg.abs(state.raw["trends"])
    wdir = cfg.abs(state.raw["weather"])
    tman, wman = read_json(tdir / "manifest.json"), read_json(wdir / "manifest.json")
    tpath = tdir / tman["filename"]

    weekly_rows = None
    if kind_of(tpath) == "xlsx":
        weekly_rows = len(load_weekly(tpath, cfg))
        trends = merge_trends_sheets(tpath, cfg)
    else:
        trends = load_weekly(tpath, cfg)   # csv ที่รวมสองชีทแล้ว

    daily = pd.read_csv(wdir / wman["file"], parse_dates=["date"])
    lags = cfg["weather"]["lags"]
    merged = attach_weather(trends, daily, lags)

    out_dir = cfg.staging_dir / state.run_id
    write_table(merged, out_dir / "trends_weather", cfg["load"]["format"])
    write_json(out_dir / "meta.json", {
        "rows_before": len(trends), "weekly_rows": weekly_rows, "lags": lags,
        "columns": list(merged.columns),
    })
    state.staging = cfg.rel(out_dir)
    state.save(cfg)
    return f"รวมแล้ว {len(merged):,} แถว x {merged.shape[1]} คอลัมน์ (lag 1-{lags})"
