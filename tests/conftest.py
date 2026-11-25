"""ข้อมูลสังเคราะห์ที่จำลองโครงสร้างจริงของ TRENDS: สัปดาห์สั้นที่ขอบปี, 23 คอลัมน์พื้นฐานเหมือน trends_merged.csv"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dengue_pipeline.config import Config

BASE_COLUMNS = [
    "Country", "Year", "Month", "Epi week", "Week start", "Week end", "Health region", "P-code",
    "Province", "Dengue fever (DF)", "Dengue haemorrhagic fever (DHF)", "Dengue shock syndrome (DSS)",
    "Dengue total", "Chikungunya", "HFMD", "Dengue rate per 100,000", "Chikungunya rate per 100,000",
    "HFMD rate per 100,000", "Centroid latitude", "Centroid longitude", "Area (km²)", "Population",
    "Population density (per km²)",
]
PROVINCES = [("TH10", "Bangkok", 13.75, 100.5), ("TH50", "Chiang Mai", 18.8, 98.9),
             ("TH90", "Songkhla", 7.2, 100.6)]
POP = 1_000_000
N_FULL_WEEKS = 14


def make_weeks(n_full: int = N_FULL_WEEKS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """สัปดาห์สั้นที่ปลายปี 2018 (2 วัน) และต้นปี 2019 (5 วัน) แล้วสัปดาห์เต็มวันอาทิตย์-เสาร์"""
    weeks = [(pd.Timestamp("2018-12-30"), pd.Timestamp("2018-12-31")),
             (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-05"))]
    s = pd.Timestamp("2019-01-06")
    for _ in range(n_full):
        weeks.append((s, s + pd.Timedelta(days=6)))
        s += pd.Timedelta(days=7)
    return weeks


def make_trends(seed: int = 1, n_full: int = N_FULL_WEEKS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for region, (code, name, lat, lon) in enumerate(PROVINCES, start=1):
        epi = {2018: 51, 2019: 0}
        for ws, we in make_weeks(n_full):
            epi[ws.year] += 1
            df_, dhf, dss = (int(x) for x in rng.integers(0, 40, 3))
            chik, hfmd = int(rng.integers(0, 5)), int(rng.integers(0, 60))
            tot = df_ + dhf + dss
            rows.append({
                "Country": "Thailand", "Year": ws.year, "Month": ws.month, "Epi week": epi[ws.year],
                "Week start": ws, "Week end": we, "Health region": region, "P-code": code, "Province": name,
                "Dengue fever (DF)": df_, "Dengue haemorrhagic fever (DHF)": dhf,
                "Dengue shock syndrome (DSS)": dss, "Dengue total": tot, "Chikungunya": chik, "HFMD": hfmd,
                "Dengue rate per 100,000": round(tot / POP * 1e5, 3),
                "Chikungunya rate per 100,000": round(chik / POP * 1e5, 3),
                "HFMD rate per 100,000": round(hfmd / POP * 1e5, 3),
                "Centroid latitude": lat, "Centroid longitude": lon, "Area (km²)": 1000.0 + region,
                "Population": POP, "Population density (per km²)": POP / (1000.0 + region),
            })
    return pd.DataFrame(rows)[BASE_COLUMNS]


def make_weather(trends: pd.DataFrame, buffer_days: int = 35, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = trends["Week start"].min() - pd.Timedelta(days=buffer_days)
    end = trends["Week end"].max()
    days = pd.date_range(start, end)
    rows = []
    for code, name, *_ in PROVINCES:
        tmax = 30 + rng.normal(0, 2, len(days))
        rows.append(pd.DataFrame({
            "P-code": code, "Province": name, "date": days,
            "precip_mm": np.round(np.clip(rng.gamma(0.6, 6, len(days)), 0, None), 2),
            "t_max_c": np.round(tmax, 2), "t_min_c": np.round(tmax - 8 - rng.random(len(days)), 2),
        }))
    return pd.concat(rows, ignore_index=True)


def write_config(root: Path, sink: str = "file", **validation_overrides) -> Config:
    (root / "config").mkdir(parents=True, exist_ok=True)
    v = {"expected_provinces": 3, "rate_abs_tol": 0.001, "lag_check_weeks_per_province": 3, "seed": 342}
    v.update(validation_overrides)
    (root / "config" / "pipeline.toml").write_text(f"""
[paths]
data_dir = "data"
reports_dir = "reports"
[trends]
input = "TRENDS.xlsx"
weekly_sheet = "weekly data"
province_sheet = "province data"
[weather]
provider = "openmeteo"
model = "era5"
timezone = "Asia/Bangkok"
buffer_days = 35
lags = 4
end_lag_days = 6
request_sleep_s = 0
input = ""
[validation]
{chr(10).join(f"{k} = {v}" for k, v in v.items())}
[load]
format = "parquet"
sink = "{sink}"
""", encoding="utf-8")
    return Config.load(root / "config" / "pipeline.toml")


@pytest.fixture
def dcfg(tmp_path) -> Config:
    """config ที่ใช้ปลายทางแบบ DuckDB"""
    return write_config(tmp_path, sink="duckdb")


@pytest.fixture
def trends() -> pd.DataFrame:
    return make_trends()


@pytest.fixture
def weather(trends) -> pd.DataFrame:
    return make_weather(trends)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return write_config(tmp_path)


@pytest.fixture
def input_files(tmp_path, trends, weather):
    """เขียน trends_merged.csv และ weather_daily.csv สังเคราะห์ลงดิสก์"""
    tp, wp = tmp_path / "trends_merged.csv", tmp_path / "weather_daily.csv"
    trends.to_csv(tp, index=False)
    weather.to_csv(wp, index=False)
    return tp, wp


def fake_fetch_factory(fail_after: int | None = None, calls: list | None = None):
    """ตัวจำลองการดึง Open-Meteo: คืนข้อมูลสังเคราะห์ตามช่วงวัน; fail_after = ล้มด้วยโควตาหมดหลังกี่ครั้ง"""
    from dengue_pipeline.weather_fetch import DailyLimit
    state = {"n": 0}

    def fetch(session, lat, lon, s, e, model, tz):
        if fail_after is not None and state["n"] >= fail_after:
            raise DailyLimit("Daily API request limit exceeded")
        state["n"] += 1
        if calls is not None:
            calls.append((lat, lon, s, e))
        days = pd.date_range(s, e)
        return pd.DataFrame({"date": days, "precip_mm": 1.0, "t_max_c": 33.0, "t_min_c": 25.0})
    return fetch
