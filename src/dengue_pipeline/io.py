"""อ่าน/เขียนตารางตามฟอร์แมตที่ config กำหนด (parquet | csv) และ helper ไฟล์"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path_no_ext: Path, fmt: str) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    elif fmt == "csv":
        p = path_no_ext.with_suffix(".csv")
        df.to_csv(p, index=False, encoding="utf-8")
    else:
        raise ValueError(f"ฟอร์แมตไม่รองรับ: {fmt}")
    return p


def read_table(path_no_ext: Path) -> pd.DataFrame:
    pq = path_no_ext.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    cs = path_no_ext.with_suffix(".csv")
    if cs.exists():
        return pd.read_csv(cs, parse_dates=_date_cols(cs))
    raise FileNotFoundError(f"ไม่พบตาราง {path_no_ext}.(parquet|csv)")


def _date_cols(csv_path: Path) -> list[str]:
    head = pd.read_csv(csv_path, nrows=0).columns
    return [c for c in ("Week start", "Week end", "date") if c in head]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
