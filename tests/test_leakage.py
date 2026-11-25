"""LKG-01 (ขั้น transform): ตัดข้อมูลทุกอย่างหลังสัปดาห์ t แล้วสร้างฟีเจอร์ใหม่ ค่าของแถว t ต้องเหมือนเดิมทุกคอลัมน์

ถ้าค่าเปลี่ยน แปลว่าฟีเจอร์ของสัปดาห์ t ใช้ข้อมูลหลังเวลา t (อนาคตรั่วเข้ามา)
ครอบคลุมคอลัมน์อากาศของสัปดาห์ t เอง (lag 0 ซึ่งรู้ครบเมื่อสัปดาห์ t จบ) และ lag 1-8

LKG-ANOM: anomaly (climatology ต่อ fold) ต้องคำนวณจาก train เท่านั้น เปลี่ยนอากาศปีทดสอบ
แล้ว climatology และ anomaly ของแถว train ต้องไม่เปลี่ยน
"""
import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dengue_pipeline.folds import get_fold
from dengue_pipeline.weather_merge import attach_weather
from conftest import make_weather


# ---------------------------------------------------------------------------
# helpers สำหรับ get_fold integration test
# ---------------------------------------------------------------------------

def _make_minimal_feat(n_train_weeks: int = 52, n_test_weeks: int = 20) -> pd.DataFrame:
    """สร้าง feature table ขั้นต่ำ (มีทุกคอลัมน์ที่ get_fold ต้องการ) สำหรับทดสอบ anomaly"""
    rng = np.random.default_rng(42)
    provinces = ["TH10", "TH50"]
    rows = []
    start = pd.Timestamp("2017-01-01")
    for pcode in provinces:
        for i in range(n_train_weeks + n_test_weeks):
            ws = start + pd.Timedelta(weeks=i)
            we = ws + pd.Timedelta(days=6)
            rows.append({
                "P-code": pcode, "Month": ws.month, "Year": ws.year,
                "Week start": ws, "Week end": we, "week_days": 7,
                "wk_rain": float(rng.gamma(0.6, 6)),
                "wk_tmax": 30.0 + rng.normal(0, 2),
                "wk_tmin": 22.0 + rng.normal(0, 2),
                # M0 features (dummy numeric values ≠ null)
                "hist_lograte_lag0": rng.random(), "hist_lograte_lag1": rng.random(),
                "hist_lograte_lag2": rng.random(), "hist_lograte_lag3": rng.random(),
                "hist_logR_prev": rng.random(), "hist_logR_prev2": rng.random(),
                "hist_growth_prev": rng.random(),
                "nat_lograte_lag0": rng.random(), "nat_logR_prev": rng.random(),
                "nat_growth_prev": rng.random(),
                "seas_sin1": rng.random(), "seas_cos1": rng.random(),
                "seas_sin2": rng.random(), "seas_cos2": rng.random(),
                "seas_sin3": rng.random(), "seas_cos3": rng.random(),
                "D": rng.normal(0, 0.5),
                # label_end = ws (< test_start=2018-01-01 สำหรับทุกแถวปี 2017)
                "label_end": ws,
            })
    return pd.DataFrame(rows)


_MIN_CFG = {
    "target": {"name": "D"},
    "features": {
        "M0": {
            "hist": ["hist_lograte_lag0", "hist_lograte_lag1",
                     "hist_lograte_lag2", "hist_lograte_lag3",
                     "hist_logR_prev", "hist_logR_prev2", "hist_growth_prev"],
            "nat": ["nat_lograte_lag0", "nat_logR_prev", "nat_growth_prev"],
            "seas_fourier_k": 3,
        },
        "M1": {
            "wx_vars": ["rain", "tmax", "tmin"],
            "wx_lags": [0, 1, 2, 3, 4],
        },
    },
    "folds": {"seed": 42},
}


def _weather_for_lags(trends: pd.DataFrame, lags: int) -> pd.DataFrame:
    """สร้างข้อมูลอากาศที่มี buffer เพียงพอสำหรับ lag k"""
    return make_weather(trends, buffer_days=(lags + 1) * 7)


@pytest.mark.parametrize("lags", [4, 8])
@pytest.mark.parametrize("cut_week_start", ["2019-01-06", "2019-02-10", "2019-03-17"])
def test_weather_features_use_only_data_up_to_week_end(trends, lags, cut_week_start):
    weather = _weather_for_lags(trends, lags)
    full = attach_weather(trends, weather, lags=lags)

    cut = pd.Timestamp(cut_week_start)
    cut_end = trends.loc[trends["Week start"] == cut, "Week end"].iloc[0]
    t_cut = trends[trends["Week start"] <= cut]
    w_cut = weather[weather["date"] <= cut_end]
    partial = attach_weather(t_cut, w_cut, lags=lags)

    keep = full[full["Week start"] <= cut].reset_index(drop=True)
    assert len(partial) == len(keep) > 0
    assert_frame_equal(keep, partial.reset_index(drop=True), check_exact=False)


@pytest.mark.parametrize("lags", [4, 8])
def test_future_weather_change_does_not_alter_past_rows(trends, lags):
    weather = _weather_for_lags(trends, lags)
    base = attach_weather(trends, weather, lags=lags)
    changed = weather.copy()
    late = changed["date"] > pd.Timestamp("2019-03-01")
    changed.loc[late, ["precip_mm", "t_max_c", "t_min_c"]] = [999.0, 99.0, 90.0]
    after = attach_weather(trends, changed, lags=lags)
    past = base["Week end"] <= pd.Timestamp("2019-03-01")
    assert past.sum() > 0
    assert_frame_equal(base[past].reset_index(drop=True), after[past].reset_index(drop=True))


# ---------------------------------------------------------------------------
# LKG-ANOM: climatology ต้องคำนวณจาก train (Week end < test_start) เท่านั้น
# ---------------------------------------------------------------------------

def _make_feature_df(n_train_weeks: int = 52, n_test_weeks: int = 20) -> pd.DataFrame:
    """สร้าง feature table สังเคราะห์ที่มี wk_* columns สำหรับทดสอบ anomaly leakage"""
    rng = np.random.default_rng(42)
    provinces = ["TH10", "TH50"]
    rows = []
    start = pd.Timestamp("2018-01-07")  # สัปดาห์เต็มแรก
    for pcode in provinces:
        for i in range(n_train_weeks + n_test_weeks):
            ws = start + pd.Timedelta(weeks=i)
            we = ws + pd.Timedelta(days=6)
            rows.append({
                "P-code": pcode,
                "Week start": ws,
                "Week end": we,
                "Month": ws.month,
                "Year": ws.year,
                "week_days": 7,
                "wk_rain": float(rng.gamma(0.6, 6)),
                "wk_tmax": 30.0 + rng.normal(0, 2),
                "wk_tmin": 22.0 + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


def _clim_sql(test_start: pd.Timestamp, keys: list[str]) -> tuple[str, list]:
    """SQL คำนวณ climatology จาก train เท่านั้น (Week end < test_start)"""
    avgs = ", ".join(f"avg(wk_{k}) AS wk_{k}_clim" for k in keys)
    return (
        f'SELECT "P-code", "Month", {avgs} FROM feat '
        f'WHERE week_days = 7 AND "Week end" < $1 GROUP BY 1, 2 ORDER BY 1, 2',
        [test_start.to_pydatetime()],
    )


def test_anomaly_climatology_ignores_test_year_data():
    """เปลี่ยน wk_* ในปีทดสอบ → climatology ของ train ต้องไม่เปลี่ยน"""
    feat = _make_feature_df()
    test_start = pd.Timestamp("2019-01-07")
    keys = ["rain", "tmax", "tmin"]

    con = duckdb.connect()
    con.register("feat", feat)
    sql, params = _clim_sql(test_start, keys)
    clim_before = con.execute(sql, params).df()

    # เปลี่ยนค่า wk_* ในปีทดสอบให้ผิดปกติมาก
    modified = feat.copy()
    is_test = modified["Week start"] >= test_start
    for k in keys:
        modified.loc[is_test, f"wk_{k}"] = 9999.0

    con.unregister("feat")
    con.register("feat", modified)
    clim_after = con.execute(sql, params).df()
    con.close()

    assert_frame_equal(clim_before, clim_after, check_exact=False,
                       obj="climatology ต้องไม่เปลี่ยนเมื่ออากาศปีทดสอบเปลี่ยน")


def test_anomaly_train_rows_unchanged_when_test_weather_changes():
    """เปลี่ยน wk_* ในปีทดสอบ → anomaly (wk - clim) ของแถว train ต้องเหมือนเดิม"""
    feat = _make_feature_df()
    test_start = pd.Timestamp("2019-01-07")
    keys = ["rain", "tmax", "tmin"]

    anom_cols = ", ".join(f"wk_{k} - wk_{k}_clim AS anom_{k}" for k in keys)
    anom_sql = f"""
    WITH clim AS (
        SELECT "P-code", "Month",
               {", ".join(f"avg(wk_{k}) AS wk_{k}_clim" for k in keys)}
        FROM feat WHERE week_days = 7 AND "Week end" < $1 GROUP BY 1, 2
    ),
    joined AS (SELECT f.*, {", ".join(f"c.wk_{k}_clim" for k in keys)}
               FROM feat f LEFT JOIN clim c USING ("P-code", "Month"))
    SELECT "P-code", "Week start", {anom_cols} FROM joined
    WHERE "Week start" < $1 AND week_days = 7 ORDER BY 1, 2
    """

    con = duckdb.connect()
    con.register("feat", feat)
    anom_before = con.execute(anom_sql, [test_start.to_pydatetime()]).df()

    modified = feat.copy()
    is_test = modified["Week start"] >= test_start
    for k in keys:
        modified.loc[is_test, f"wk_{k}"] = 9999.0

    con.unregister("feat")
    con.register("feat", modified)
    anom_after = con.execute(anom_sql, [test_start.to_pydatetime()]).df()
    con.close()

    assert len(anom_before) > 0
    assert_frame_equal(anom_before, anom_after, check_exact=False,
                       obj="anomaly ของแถว train ต้องไม่เปลี่ยนเมื่ออากาศปีทดสอบเปลี่ยน")


# ---------------------------------------------------------------------------
# LKG-FOLD: ทดสอบผ่าน get_fold() จริง (integration)
# ---------------------------------------------------------------------------

def test_get_fold_wxa_not_leaked_by_test_weather():
    """wxa_* (anomaly lag 0-4) ใน train จาก get_fold() ต้องไม่เปลี่ยนเมื่อเปลี่ยน wk_* ปีทดสอบ"""
    feat = _make_minimal_feat()
    fold = {"name": "2018", "train_years": (2017, 2017), "test_year": 2018}

    before = get_fold(feat, fold, _MIN_CFG)
    tr_before = before["train"]
    wxa_cols = sorted(c for c in tr_before.columns if c.startswith("wxa_"))
    assert len(wxa_cols) > 0, "ไม่พบ wxa_* ใน train"

    # เปลี่ยน wk_* ในปีทดสอบให้ผิดปกติมาก
    modified = feat.copy()
    is_test = modified["Year"] == 2018
    for k in ["rain", "tmax", "tmin"]:
        modified.loc[is_test, f"wk_{k}"] = 9999.0

    after = get_fold(modified, fold, _MIN_CFG)
    tr_after = after["train"]

    assert_frame_equal(
        tr_before[wxa_cols].reset_index(drop=True),
        tr_after[wxa_cols].reset_index(drop=True),
        check_exact=False,
        obj="wxa_* ของ train ต้องไม่เปลี่ยนเมื่อ wk_* ปีทดสอบเปลี่ยน",
    )


def test_get_fold_wxa_lag4_not_null_for_sufficient_history():
    """wxa_*_lag4 ต้องไม่เป็น null ในแถว train ที่มีประวัติ 4+ สัปดาห์ก่อนหน้า"""
    feat = _make_minimal_feat()
    fold = {"name": "2018", "train_years": (2017, 2017), "test_year": 2018}

    result = get_fold(feat, fold, _MIN_CFG)
    tr = result["train"]
    lag4_cols = [c for c in tr.columns if c.endswith("_lag4") and c.startswith("wxa_")]
    assert len(lag4_cols) > 0
    # แถวที่มีประวัติครบ (ไม่ใช่แถวแรกๆ ของจังหวัด) ต้องไม่ null
    for pcode, grp in tr.groupby("P-code"):
        # ข้าม 4 แถวแรกของแต่ละจังหวัด (อาจยังไม่มี lag)
        later = grp.iloc[4:]
        if len(later) > 0:
            null_rate = later[lag4_cols].isna().mean().mean()
            assert null_rate < 0.5, f"{pcode}: wxa_lag4 null rate {null_rate:.0%} สูงผิดปกติ"
