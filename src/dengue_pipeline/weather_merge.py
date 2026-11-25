"""รวมอากาศรายวันเข้ากับตาราง TRENDS รายสัปดาห์ (จังหวัด x สัปดาห์)

พอร์ตจาก merge_weather.py เดิม ตรรกะการคำนวณเหมือนเดิมทุกประการ (ลำดับคอลัมน์ผลลัพธ์เท่าเดิม)
ต่างกันที่ การตรวจอินพุต (คีย์ซ้ำ สัปดาห์ซ้อน ค่าติดลบ ฯลฯ) ย้ายไปเป็นกฎ BLOCK ในชั้น validate
แล้ว ฟังก์ชันนี้เหลือเฉพาะการคำนวณ

TRENDS ตัดสัปดาห์ที่ขอบปีปฏิทิน จึงมีสัปดาห์ 1-6 วัน อากาศจึงรวมตามช่วง [Week start, Week end]
ของแต่ละแถวจริง ค่าจะเป็น NaN ถ้าวันที่ต้องใช้ขาดแม้วันเดียว (ไม่ใช้ผลรวม/ค่าเฉลี่ยบางส่วน)

คอลัมน์ที่เพิ่ม: week_days, weather_days, rain_wk_mm, tmax_mean_c, tmax_max_c, tmin_mean_c,
tmin_min_c, tmean_c, rain_mean_mm_day และ lag K = 1..lags ของ rain_wk_mm / tmax_mean_c / tmin_mean_c
(lagK = บล็อก 7 วันที่ K ก่อน Week start)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_pipeline.errors import StageError

LAG_VARS = ["rain_wk_mm", "tmax_mean_c", "tmin_mean_c"]


def attach_weather(trends: pd.DataFrame, weather: pd.DataFrame, lags: int) -> pd.DataFrame:
    t = trends.copy()
    w = weather.copy()
    need_t = {"P-code", "Week start", "Week end"}
    need_w = {"P-code", "date", "precip_mm", "t_max_c", "t_min_c"}
    if need_t - set(t.columns) or need_w - set(w.columns):
        raise StageError(f"คอลัมน์ไม่ครบ: trends ขาด {sorted(need_t - set(t.columns))}, "
                         f"weather ขาด {sorted(need_w - set(w.columns))}")

    t["week_days"] = (t["Week end"] - t["Week start"]).dt.days + 1

    # ---- สัปดาห์ปัจจุบัน: รวมตามช่วงวันของแต่ละแถว ----------------------------------
    weeks = t[["Week start", "Week end"]].drop_duplicates()
    cal = pd.concat([pd.DataFrame({"Week start": s, "date": pd.date_range(s, e)})
                     for s, e in weeks.itertuples(index=False)], ignore_index=True)
    if cal["date"].duplicated().any():
        raise StageError("สัปดาห์ของ TRENDS ซ้อนกันในเวลา")
    w["t_mean_c"] = (w["t_max_c"] + w["t_min_c"]) / 2
    w["ok"] = w[["precip_mm", "t_max_c", "t_min_c"]].notna().all(axis=1)
    wk = (w.merge(cal, on="date")
            .groupby(["P-code", "Week start"])
            .agg(weather_days=("ok", "sum"),
                 rain_wk_mm=("precip_mm", "sum"), rain_n=("precip_mm", "count"),
                 tmax_mean_c=("t_max_c", "mean"), tmax_max_c=("t_max_c", "max"), tmax_n=("t_max_c", "count"),
                 tmin_mean_c=("t_min_c", "mean"), tmin_min_c=("t_min_c", "min"), tmin_n=("t_min_c", "count"),
                 tmean_c=("t_mean_c", "mean"))
            .reset_index())
    out = t.merge(wk, on=["P-code", "Week start"], how="left", validate="many_to_one")
    wd = out["week_days"]
    out.loc[out["rain_n"] < wd, "rain_wk_mm"] = np.nan
    out.loc[out["tmax_n"] < wd, ["tmax_mean_c", "tmax_max_c"]] = np.nan
    out.loc[out["tmin_n"] < wd, ["tmin_mean_c", "tmin_min_c"]] = np.nan
    out.loc[out["weather_days"] < wd, "tmean_c"] = np.nan
    out["rain_mean_mm_day"] = out["rain_wk_mm"] / wd
    out = out.drop(columns=["rain_n", "tmax_n", "tmin_n"])

    # ---- lag: บล็อก 7 วันก่อน Week start (rolling บนตารางรายวันที่ต่อเนื่อง) -------------
    codes = sorted(w["P-code"].unique())
    days = pd.date_range(w["date"].min(), w["date"].max())
    grid = pd.MultiIndex.from_product([codes, days], names=["P-code", "date"])
    d = w.set_index(["P-code", "date"])[["precip_mm", "t_max_c", "t_min_c"]].reindex(grid)
    g = d.groupby(level=0)
    roll = pd.DataFrame({
        "rain_wk_mm": g["precip_mm"].transform(lambda s: s.rolling(7, min_periods=7).sum()),
        "tmax_mean_c": g["t_max_c"].transform(lambda s: s.rolling(7, min_periods=7).mean()),
        "tmin_mean_c": g["t_min_c"].transform(lambda s: s.rolling(7, min_periods=7).mean()),
    }).reset_index()
    for k in range(1, lags + 1):
        lag = roll.copy()
        # บล็อกที่จบวัน x เป็น lag k ของแถวที่ Week start = x + 1 + 7*(k-1)
        lag["Week start"] = lag["date"] + pd.Timedelta(days=1 + 7 * (k - 1))
        lag = lag.drop(columns="date").rename(columns={c: f"{c}_lag{k}" for c in LAG_VARS})
        out = out.merge(lag, on=["P-code", "Week start"], how="left", validate="many_to_one")
    return out
