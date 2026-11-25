"""กฎชั้น 2: ตรวจหลัง merge (MRG)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_pipeline.rules import RuleResult, check, skip

WEATHER_COLS = ["weather_days", "rain_wk_mm", "tmax_mean_c", "tmax_max_c",
                "tmin_mean_c", "tmin_min_c", "tmean_c", "rain_mean_mm_day"]
LAG_VARS = ["rain_wk_mm", "tmax_mean_c", "tmin_mean_c"]
BASE_NEEDED_MRG07 = ["Population", "Area (km²)", "Centroid latitude", "Centroid longitude"]


def lag_cols(lags: int) -> list[str]:
    return [f"{v}_lag{k}" for k in range(1, lags + 1) for v in LAG_VARS]


def check_merged(df: pd.DataFrame, meta: dict, daily: pd.DataFrame | None,
                 lags: int, seed: int, weeks_per_province: int) -> list[RuleResult]:
    out: list[RuleResult] = []

    # MRG-01 คีย์ไม่ซ้ำ และครบทุกจังหวัดทุกสัปดาห์
    dup = int(df.duplicated(["P-code", "Week start"]).sum())
    n_prov, n_weeks = df["P-code"].nunique(), df["Week start"].nunique()
    out.append(check("MRG-01", dup == 0 and len(df) == n_prov * n_weeks,
                     f"ซ้ำ {dup} แถว; {len(df)} แถว = {n_prov} จังหวัด x {n_weeks} สัปดาห์ (ต้องเท่ากัน)"))

    # MRG-02 จำนวนแถวหลัง join เท่าก่อน join
    before = meta.get("rows_before")
    out.append(skip("MRG-02", "ไม่มี meta.rows_before") if before is None else
               check("MRG-02", len(df) == before, f"ก่อน {before} -> หลัง {len(df)}"))

    # MRG-03 weather_days = week_days
    bad3 = int((df["weather_days"] != df["week_days"]).sum())
    out.append(check("MRG-03", bad3 == 0, f"แถวที่ weather_days != week_days: {bad3}"))

    # MRG-04 ไม่มีค่าว่างในคอลัมน์อากาศและ lag
    cols = WEATHER_COLS + lag_cols(lags)
    nan = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
    absent = [c for c in cols if c not in df.columns]
    out.append(check("MRG-04", not nan and not absent,
                     f"คอลัมน์ที่มีค่าว่าง {len(nan)}, คอลัมน์ที่หาย {len(absent)}",
                     f"nan={nan} absent={absent}" if nan or absent else ""))

    # MRG-05 lagK ตรงกับบล็อก 7 วันก่อน Week start (สุ่มตรวจเทียบตารางรายวัน)
    out.append(_lag_spot_check(df, daily, lags, seed, weeks_per_province))

    # MRG-06 ฝน >= 0 และ tmax_mean >= tmin_mean
    neg = int((df["rain_wk_mm"] < 0).sum())
    inv = int((df["tmax_mean_c"] < df["tmin_mean_c"]).sum())
    out.append(check("MRG-06", neg == 0 and inv == 0, f"ฝนติดลบ {neg} แถว, tmax_mean<tmin_mean {inv} แถว"))

    # MRG-07 หลังรวมสองชีทของ TRENDS: จำนวนแถวเท่ากับ weekly sheet และไม่มีค่าว่างในคอลัมน์จังหวัด
    needed = [c for c in BASE_NEEDED_MRG07 if c in df.columns]
    absent7 = [c for c in BASE_NEEDED_MRG07 if c not in df.columns]
    nan7 = {c: int(df[c].isna().sum()) for c in needed if df[c].isna().any()}
    weekly_rows = meta.get("weekly_rows")
    rows_ok = True if weekly_rows is None else len(df) == weekly_rows
    note = "ตัดการเทียบจำนวนแถวกับ weekly sheet (อินพุตเป็น csv ที่รวมแล้ว)" if weekly_rows is None else ""
    out.append(check("MRG-07", rows_ok and not nan7 and not absent7,
                     f"แถว {len(df)} (weekly sheet {weekly_rows}), ค่าว่างในคอลัมน์จังหวัด {nan7 or 'ไม่มี'}, "
                     f"คอลัมน์ที่หาย {absent7 or 'ไม่มี'}", note))
    return out


def _lag_spot_check(df: pd.DataFrame, daily: pd.DataFrame | None, lags: int,
                    seed: int, weeks_per_province: int) -> RuleResult:
    if daily is None:
        return skip("MRG-05", "ไม่มีตารางอากาศรายวันให้เทียบ")
    daily = daily.set_index(["P-code", "date"]).sort_index()
    rng = np.random.default_rng(seed)
    bad, checked = [], 0
    for code, g in df.groupby("P-code"):
        n = min(weeks_per_province, len(g))
        for idx in rng.choice(g.index.to_numpy(), size=n, replace=False):
            row = df.loc[idx]
            for k in range(1, lags + 1):
                block_start = row["Week start"] - pd.Timedelta(days=7 * k)
                days = pd.date_range(block_start, periods=7)
                try:
                    blk = daily.loc[(code, days), :]
                except KeyError:
                    bad.append((code, str(row["Week start"].date()), k, "วันขาด"))
                    continue
                exp = {"rain_wk_mm": blk["precip_mm"].sum(),
                       "tmax_mean_c": blk["t_max_c"].mean(),
                       "tmin_mean_c": blk["t_min_c"].mean()}
                for v, e in exp.items():
                    got = row[f"{v}_lag{k}"]
                    checked += 1
                    if not (np.isclose(got, e, rtol=1e-9, atol=1e-9) or (np.isnan(got) and np.isnan(e))):
                        bad.append((code, str(row["Week start"].date()), k, v))
    return check("MRG-05", not bad, f"ตรวจ {checked} ค่า ผิด {len(bad)}",
                 f"ตัวอย่าง: {bad[:3]}" if bad else "")
