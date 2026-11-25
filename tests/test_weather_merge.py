import numpy as np
import pandas as pd

from dengue_pipeline.validation.checks_merged import check_merged, lag_cols
from dengue_pipeline.weather_merge import attach_weather

EXPECTED_NEW_COLUMNS = (
    ["week_days", "weather_days", "rain_wk_mm", "tmax_mean_c", "tmax_max_c", "tmin_mean_c",
     "tmin_min_c", "tmean_c", "rain_mean_mm_day"]
    + [f"{v}_lag{k}" for k in range(1, 5) for v in ["rain_wk_mm", "tmax_mean_c", "tmin_mean_c"]]
)


def test_output_columns_match_existing_merged_file_layout(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    assert list(out.columns) == list(trends.columns) + EXPECTED_NEW_COLUMNS
    assert out.shape[1] == 23 + 21 == 44          # เท่ากับ trends_weather_merged.csv จริง (44 คอลัมน์)
    assert len(out) == len(trends)


def test_current_week_values_for_a_full_week(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    row = out[(out["P-code"] == "TH50") & (out["Week start"] == pd.Timestamp("2019-02-03"))].iloc[0]
    days = pd.date_range("2019-02-03", "2019-02-09")
    w = weather[(weather["P-code"] == "TH50") & weather["date"].isin(days)]
    assert row["week_days"] == 7 and row["weather_days"] == 7
    assert np.isclose(row["rain_wk_mm"], w["precip_mm"].sum())
    assert np.isclose(row["tmax_mean_c"], w["t_max_c"].mean())
    assert np.isclose(row["tmax_max_c"], w["t_max_c"].max())
    assert np.isclose(row["tmin_min_c"], w["t_min_c"].min())
    assert np.isclose(row["rain_mean_mm_day"], row["rain_wk_mm"] / 7)


def test_partial_weeks_aggregate_over_their_own_days(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    short = out[(out["P-code"] == "TH10") & (out["Week start"] == pd.Timestamp("2018-12-30"))].iloc[0]
    assert short["week_days"] == 2 and short["weather_days"] == 2
    w = weather[(weather["P-code"] == "TH10") & weather["date"].isin(pd.date_range("2018-12-30", "2018-12-31"))]
    assert np.isclose(short["rain_wk_mm"], w["precip_mm"].sum())
    assert np.isclose(short["rain_mean_mm_day"], short["rain_wk_mm"] / 2)


def test_lag_k_is_the_k_th_seven_day_block_before_week_start(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    ws = pd.Timestamp("2019-03-03")
    row = out[(out["P-code"] == "TH90") & (out["Week start"] == ws)].iloc[0]
    for k in range(1, 5):
        days = pd.date_range(ws - pd.Timedelta(days=7 * k), periods=7)
        w = weather[(weather["P-code"] == "TH90") & weather["date"].isin(days)]
        assert np.isclose(row[f"rain_wk_mm_lag{k}"], w["precip_mm"].sum())
        assert np.isclose(row[f"tmax_mean_c_lag{k}"], w["t_max_c"].mean())
        assert np.isclose(row[f"tmin_mean_c_lag{k}"], w["t_min_c"].mean())


def test_missing_day_gives_nan_never_a_partial_value(trends, weather):
    day = pd.Timestamp("2019-02-05")
    bad = weather[~((weather["P-code"] == "TH50") & (weather["date"] == day))]
    out = attach_weather(trends, bad, lags=4)
    row = out[(out["P-code"] == "TH50") & (out["Week start"] == pd.Timestamp("2019-02-03"))].iloc[0]
    assert row["weather_days"] == 6
    assert np.isnan(row["rain_wk_mm"]) and np.isnan(row["tmax_mean_c"]) and np.isnan(row["tmean_c"])
    # สัปดาห์ถัดไปใช้วันนั้นเป็น lag1 -> ต้องเป็น NaN เช่นกัน
    nxt = out[(out["P-code"] == "TH50") & (out["Week start"] == pd.Timestamp("2019-02-10"))].iloc[0]
    assert np.isnan(nxt["rain_wk_mm_lag1"])
    # และ MRG-03/04 ต้องจับได้
    res = {r.rule_id: r for r in check_merged(out, {"rows_before": len(trends)}, bad, 4, 342, 3)}
    assert res["MRG-03"].blocking_failure and res["MRG-04"].blocking_failure


def test_province_without_weather_is_caught(trends, weather):
    out = attach_weather(trends, weather[weather["P-code"] != "TH90"], lags=4)
    res = {r.rule_id: r for r in check_merged(out, {"rows_before": len(trends)}, None, 4, 342, 3)}
    assert res["MRG-03"].blocking_failure and res["MRG-04"].blocking_failure


def test_valid_merge_passes_all_merged_rules(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    res = check_merged(out, {"rows_before": len(trends), "weekly_rows": None}, weather, 4, 342, 3)
    assert not [r for r in res if r.blocking_failure], [(r.rule_id, r.measured) for r in res if r.blocking_failure]
    assert {r.rule_id: r.outcome for r in res}["MRG-05"] == "pass"


def test_corrupted_lag_is_caught_by_mrg05(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    out[lag_cols(4)] = out[lag_cols(4)] + 1.0       # เลื่อนค่า lag ทุกแถวผิด
    res = {r.rule_id: r for r in check_merged(out, {"rows_before": len(trends)}, weather, 4, 342, 3)}
    assert res["MRG-05"].blocking_failure


def test_row_count_change_is_caught_by_mrg02(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    res = {r.rule_id: r for r in check_merged(out, {"rows_before": len(trends) - 1}, weather, 4, 342, 3)}
    assert res["MRG-02"].blocking_failure


def test_mrg07_row_count_against_weekly_sheet_and_nan_in_province_columns(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    res = {r.rule_id: r for r in check_merged(out, {"rows_before": len(out), "weekly_rows": len(out) + 3},
                                              weather, 4, 342, 3)}
    assert res["MRG-07"].blocking_failure
    out.loc[0, "Population"] = np.nan
    res = {r.rule_id: r for r in check_merged(out, {"rows_before": len(out), "weekly_rows": len(out)},
                                              weather, 4, 342, 3)}
    assert res["MRG-07"].blocking_failure


def test_duplicate_key_after_merge_is_caught_by_mrg01(trends, weather):
    out = attach_weather(trends, weather, lags=4)
    dup = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    res = {r.rule_id: r for r in check_merged(dup, {"rows_before": len(trends)}, weather, 4, 342, 3)}
    assert res["MRG-01"].blocking_failure and res["MRG-02"].blocking_failure
