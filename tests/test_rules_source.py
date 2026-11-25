import numpy as np
import pandas as pd

from dengue_pipeline.validation.checks_source import check_trends, check_weather, weather_window

EXPECTED = 3
TOL = 0.001


def by_id(results):
    out = {}
    for r in results:
        out.setdefault(r.rule_id, []).append(r)
    return out


def trends_results(df, prev=None):
    return by_id(check_trends(df, EXPECTED, TOL, prev))


def window_of(trends):
    return weather_window(trends, 35, 6)


def weather_results(w, trends, prev=None):
    return by_id(check_weather(w, EXPECTED, window_of(trends), prev))


def assert_only_failing(results, expected_ids):
    failed = {k for k, v in results.items() if any(r.blocking_failure for r in v)}
    assert failed == set(expected_ids), failed


def test_valid_trends_passes_everything(trends):
    res = trends_results(trends)
    assert_only_failing(res, [])
    assert all(r.outcome == "pass" for rs in res.values() for r in rs)


def test_valid_weather_passes_everything(trends, weather):
    res = weather_results(weather, trends)
    assert_only_failing(res, [])
    assert res["WX-04"][0].severity == "INFO"


def test_duplicate_key_blocks(trends):
    dup = pd.concat([trends, trends.iloc[[0]]], ignore_index=True)
    res = trends_results(dup)
    assert any(r.blocking_failure for r in res["SRC-02"])


def test_total_mismatch_blocks_trd03(trends):
    bad = trends.copy()
    bad.loc[5, "Dengue total"] += 1
    assert_only_failing(trends_results(bad), ["TRD-03", "TRD-04"])


def test_negative_count_blocks_trd02(trends):
    bad = trends.copy()
    bad.loc[3, "HFMD"] = -1
    assert "TRD-02" in {k for k, v in trends_results(bad).items() if any(r.blocking_failure for r in v)}


def test_rate_beyond_rounding_blocks_trd04(trends):
    bad = trends.copy()
    bad.loc[7, "Dengue rate per 100,000"] += 0.01   # เกินการปัดเศษที่ 3 ตำแหน่ง
    assert_only_failing(trends_results(bad), ["TRD-04"])


def test_rate_within_rounding_passes(trends):
    ok = trends.copy()
    ok.loc[7, "Dengue rate per 100,000"] += 0.0004
    assert_only_failing(trends_results(ok), [])


def test_missing_population_is_skip_not_pass(trends):
    res = trends_results(trends.drop(columns=["Population"]))
    assert res["TRD-04"][0].outcome == "skip"
    assert_only_failing(res, [])


def test_missing_province_blocks_trd01(trends):
    bad = trends[trends["P-code"] != "TH90"]
    assert_only_failing(trends_results(bad), ["TRD-01"])


def test_unequal_weeks_per_province_blocks_trd01(trends):
    bad = trends.drop(index=trends.index[trends["P-code"] == "TH50"][-1])
    assert_only_failing(trends_results(bad), ["TRD-01"])


def test_short_week_mid_year_blocks_trd06(trends):
    bad = trends.copy()
    # ตัดสัปดาห์กลางปีให้เหลือ 3 วัน (ไม่ใช่ขอบปี)
    mask = bad["Week start"] == pd.Timestamp("2019-02-03")
    bad.loc[mask, "Week end"] = pd.Timestamp("2019-02-05")
    assert_only_failing(trends_results(bad), ["TRD-06"])


def test_overlapping_weeks_block_trd05(trends):
    bad = trends.copy()
    mask = bad["Week start"] == pd.Timestamp("2019-02-03")
    bad.loc[mask, "Week end"] = pd.Timestamp("2019-02-10")   # ยาว 8 วัน ทับสัปดาห์ถัดไป
    assert "TRD-05" in {k for k, v in trends_results(bad).items() if any(r.blocking_failure for r in v)}


def test_row_count_drop_blocks_src03(trends):
    res = trends_results(trends, prev=len(trends) + 10)
    assert any(r.blocking_failure for r in res["SRC-03"])


def test_missing_required_column_blocks_and_skips_rest(trends):
    res = trends_results(trends.drop(columns=["Dengue total"]))
    assert any(r.blocking_failure for r in res["SRC-01"])
    assert res["TRD-03"][0].outcome == "skip"


def test_negative_rain_blocks_wx01(trends, weather):
    bad = weather.copy()
    bad.loc[10, "precip_mm"] = -0.5
    assert_only_failing(weather_results(bad, trends), ["WX-01"])


def test_tmax_below_tmin_blocks_wx01(trends, weather):
    bad = weather.copy()
    bad.loc[10, ["t_max_c", "t_min_c"]] = [20.0, 25.0]
    assert_only_failing(weather_results(bad, trends), ["WX-01"])


def test_missing_day_blocks_wx02(trends, weather):
    bad = weather.drop(index=20)
    assert_only_failing(weather_results(bad, trends), ["WX-02"])


def test_nan_blocks_wx03(trends, weather):
    bad = weather.copy()
    bad.loc[30, "t_min_c"] = np.nan
    assert "WX-03" in {k for k, v in weather_results(bad, trends).items() if any(r.blocking_failure for r in v)}


def test_weather_window_buffer_and_end_cap(trends):
    w = weather_window(trends, 35, 6)
    assert w["start"] == "2018-11-25"          # 2018-12-30 ลบ 35 วัน
    assert w["end"] == str(trends["Week end"].max().date())
    import datetime as dt
    capped = weather_window(trends, 35, 6, today=dt.date(2019, 2, 1))
    assert capped["end"] == "2019-01-26"       # วันนี้ลบ 6 วัน
