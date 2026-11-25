"""ดึงอากาศรายวัน (ฝน, t_max, t_min) จาก ERA5 ผ่าน Open-Meteo ที่จุด centroid ของแต่ละจังหวัด

พอร์ตจาก fetch_weather.py เดิม (ตรรกะเหมือนเดิม) ต่างกันที่:
  - เหลือเฉพาะ Open-Meteo/ERA5 (ตัด NASA POWER ออกตามขอบเขตที่ตกลง)
  - โควตารายวันหมด -> ยกข้อยกเว้น IncompleteRun (สถานะ "ยังไม่ครบ") แทน sys.exit(2)
  - cache อยู่ใต้ data/cache/weather ดึงต่อได้ด้วยการรันซ้ำ
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

from dengue_pipeline.config import Config
from dengue_pipeline.errors import IncompleteRun, StageError

OM_URL = "https://archive-api.open-meteo.com/v1/archive"
COLS = ["date", "precip_mm", "t_max_c", "t_min_c"]


class DailyLimit(Exception):
    """โควตารายวันของ Open-Meteo หมด"""


def get_json(session, url, params, max_tries: int = 6):
    for attempt in range(max_tries):
        try:
            r = session.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            wait = 5 * 2 ** attempt
            print(f"  network error ({e.__class__.__name__}); retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()
        low = r.text[:400].lower()
        if "limit" in low and "daily" in low:
            raise DailyLimit(r.text[:200])
        if "limit" in low and "hourly" in low:
            now = dt.datetime.now()
            wait = 3600 - (now.minute * 60 + now.second) + 10
            print(f"  hourly limit reached; sleeping {wait // 60} min")
            time.sleep(wait)
            continue
        if r.status_code == 429 or ("limit" in low and "minut" in low):
            print("  minute limit reached; sleeping 65s")
            time.sleep(65)
            continue
        if r.status_code >= 500:
            wait = 10 * 2 ** attempt
            print(f"  server error {r.status_code}; retry in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("too many retries for " + str(params))


def fetch_chunk(session, lat: float, lon: float, start: dt.date, end: dt.date,
                model: str, tz: str) -> pd.DataFrame:
    params = {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": tz,
    }
    if model != "best_match":
        params["models"] = model
    d = get_json(session, OM_URL, params)["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "precip_mm": d["precipitation_sum"],
        "t_max_c": d["temperature_2m_max"],
        "t_min_c": d["temperature_2m_min"],
    })


def check_complete(df: pd.DataFrame, label: str) -> None:
    """หยุดทันทีถ้าตัวแปรใดว่างทั้งคอลัมน์ (โมเดล/พารามิเตอร์ผิด)"""
    empty = [c for c in COLS[1:] if df[c].isna().all()]
    if empty:
        raise StageError(f"{label}: ตัวแปร {empty} ว่างทั้งคอลัมน์ ตรวจ model/พารามิเตอร์ "
                         "(หมายเหตุ: era5_land ไม่มีข้อมูลฝน)")


def year_chunks(start: dt.date, end: dt.date):
    for y in range(start.year, end.year + 1):
        yield max(start, dt.date(y, 1, 1)), min(end, dt.date(y, 12, 31))


def fetch_all(cfg: Config, provinces: pd.DataFrame, window: dict,
              session=None, fetch=fetch_chunk, sleep_s: float | None = None) -> pd.DataFrame:
    """ดึงทุกจังหวัดทุกปีตาม window แล้วรวมเป็นตารางเดียว

    chunk ที่ดึงแล้วอยู่ใน cache ข้ามได้ ถ้าโควตาหมดจะยกข้อยกเว้น IncompleteRun
    (ของที่ดึงได้แล้วยังอยู่ใน cache รันซ้ำคำสั่งเดิมเพื่อต่อ)
    """
    wcfg = cfg["weather"]
    model, tz = wcfg["model"], wcfg["timezone"]
    pause = wcfg["request_sleep_s"] if sleep_s is None else sleep_s
    start = dt.date.fromisoformat(window["start"])
    end = dt.date.fromisoformat(window["end"])
    session = session or requests.Session()
    cache = cfg.cache_dir / "weather"
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"openmeteo_{model}"

    def chunk_file(code, s, e) -> Path:
        return cache / f"{tag}_{code}_{s:%Y%m%d}_{e:%Y%m%d}.csv"

    n = len(provinces)
    try:
        for i, r in provinces.iterrows():
            for s, e in year_chunks(start, end):
                f = chunk_file(r["P-code"], s, e)
                if f.exists():
                    continue
                print(f"[{i + 1}/{n}] {r['P-code']} {r['Province']} {s} -> {e}")
                df = fetch(session, r["Centroid latitude"], r["Centroid longitude"], s, e, model, tz)
                check_complete(df, f"{r['P-code']} {s}")
                df.to_csv(f, index=False)
                if pause:
                    time.sleep(pause)
    except DailyLimit as ex:
        raise IncompleteRun(
            f"โควตารายวันของ Open-Meteo หมด ({ex}) ของที่ดึงแล้วอยู่ใน {cache} "
            "รันคำสั่งเดิมซ้ำพรุ่งนี้เพื่อต่อ") from ex

    parts = []
    for _, r in provinces.iterrows():
        for s, e in year_chunks(start, end):
            d = pd.read_csv(chunk_file(r["P-code"], s, e), parse_dates=["date"])
            d.insert(0, "Province", r["Province"])
            d.insert(0, "P-code", r["P-code"])
            parts.append(d)
    return pd.concat(parts, ignore_index=True)[["P-code", "Province"] + COLS]
