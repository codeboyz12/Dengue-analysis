"""อ่านข้อมูล TRENDS จากอินพุตได้สองแบบ: TRENDS.xlsx (สองชีท) หรือ csv ที่รวมแล้ว"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dengue_pipeline.config import Config
from dengue_pipeline.errors import StageError

PROVINCE_COLS = ["P-code", "Province", "Centroid latitude", "Centroid longitude"]


def kind_of(path: Path) -> str:
    s = path.suffix.lower()
    if s in (".xlsx", ".xlsm"):
        return "xlsx"
    if s == ".csv":
        return "csv"
    raise StageError(f"นามสกุลไฟล์ TRENDS ไม่รองรับ: {path.name} (รองรับ .xlsx, .csv)")


def load_weekly(path: Path, cfg: Config) -> pd.DataFrame:
    """ตารางรายสัปดาห์ (xlsx: ชีท weekly data ; csv: ทั้งไฟล์)"""
    if kind_of(path) == "xlsx":
        df = pd.read_excel(path, sheet_name=cfg["trends"]["weekly_sheet"])
    else:
        df = pd.read_csv(path)
    for c in ("Week start", "Week end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load_province_sheet(path: Path, cfg: Config) -> pd.DataFrame:
    """ชีท province data ทั้งชีท (ทุกคอลัมน์) สำหรับรวมกับ weekly data"""
    if kind_of(path) != "xlsx":
        raise StageError("ชีท province data มีเฉพาะในไฟล์ .xlsx")
    return pd.read_excel(path, sheet_name=cfg["trends"]["province_sheet"])


def load_provinces(path: Path, cfg: Config) -> pd.DataFrame:
    """ตารางจังหวัด (รหัส ชื่อ centroid) สำหรับดึงอากาศ"""
    if kind_of(path) == "xlsx":
        df = pd.read_excel(path, sheet_name=cfg["trends"]["province_sheet"], usecols=PROVINCE_COLS)
    else:
        df = pd.read_csv(path, usecols=PROVINCE_COLS).drop_duplicates("P-code")
    return df.drop_duplicates("P-code").reset_index(drop=True)
