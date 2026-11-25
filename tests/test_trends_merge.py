"""ขั้น 3a: รวมสองชีทของ TRENDS (weekly data + province data) ในสาย pipeline"""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from conftest import make_trends
from dengue_pipeline.errors import StageError
from dengue_pipeline.pipeline import RunArgs, run_pipeline
from dengue_pipeline.stages.transform import merge_trends_sheets

PROV_EXTRA = ["Centroid latitude", "Centroid longitude", "Area (km²)", "Population",
              "Population density (per km²)"]


def split_sheets(trends: pd.DataFrame, with_year: bool = True):
    """แยกตารางรวมสังเคราะห์กลับเป็นสองชีทแบบที่ TRENDS.xlsx น่าจะเป็น"""
    weekly = trends.drop(columns=PROV_EXTRA)
    keys = ["P-code", "Year"] if with_year else ["P-code"]
    prov = trends[["P-code", "Province", *(["Year"] if with_year else []), *PROV_EXTRA]].drop_duplicates(keys)
    return weekly, prov


def write_xlsx(path, weekly: pd.DataFrame, prov: pd.DataFrame):
    with pd.ExcelWriter(path) as xw:
        weekly.to_excel(xw, sheet_name="weekly data", index=False)
        prov.to_excel(xw, sheet_name="province data", index=False)
    return path


@pytest.mark.parametrize("with_year", [True, False])
def test_merge_reproduces_the_merged_table(tmp_path, cfg, trends, with_year):
    xlsx = write_xlsx(tmp_path / "TRENDS.xlsx", *split_sheets(trends, with_year))
    merged = merge_trends_sheets(xlsx, cfg)
    assert_frame_equal(merged, trends, check_dtype=False)   # คอลัมน์/ลำดับ/ค่า เหมือน trends_merged.csv


def test_duplicate_province_keys_stop_with_clear_message(tmp_path, cfg, trends):
    weekly, prov = split_sheets(trends, with_year=False)
    xlsx = write_xlsx(tmp_path / "TRENDS.xlsx", weekly, pd.concat([prov, prov.iloc[:1]]))
    with pytest.raises(StageError, match="ซ้ำ"):
        merge_trends_sheets(xlsx, cfg)


def test_missing_population_stops_instead_of_guessing(tmp_path, cfg, trends):
    weekly, prov = split_sheets(trends)
    xlsx = write_xlsx(tmp_path / "TRENDS.xlsx", weekly, prov.drop(columns=["Population"]))
    with pytest.raises(StageError, match="Population"):
        merge_trends_sheets(xlsx, cfg)


def test_full_run_from_xlsx_compares_row_count_with_weekly_sheet(tmp_path, cfg, trends, weather):
    xlsx = write_xlsx(tmp_path / "TRENDS.xlsx", *split_sheets(trends))
    wp = tmp_path / "weather_daily.csv"
    weather.to_csv(wp, index=False)
    assert run_pipeline(cfg, "all", args=RunArgs(trends_input=str(xlsx), weather_input=str(wp))) == 0
    log = pd.read_csv(cfg.reports_dir / "audit_log.csv")
    assert (log["outcome"] == "fail").sum() == 0
    mrg07 = log[log["rule_id"] == "MRG-07"].iloc[0]
    assert mrg07["outcome"] == "pass" and f"weekly sheet {len(trends)}" in mrg07["measured"]
    assert "csv" not in str(mrg07["detail"])


def wide_province_sheet(trends: pd.DataFrame) -> pd.DataFrame:
    """ชีท province data รูปแบบจริงของ TRENDS: หนึ่งแถวต่อจังหวัด คอลัมน์ Population ต่อปี"""
    base = trends[["P-code", "Province", "Centroid latitude", "Centroid longitude", "Area (km²)"]] \
        .drop_duplicates("P-code").reset_index(drop=True)
    pops = trends.pivot_table(index="P-code", columns="Year", values="Population", aggfunc="first")
    pops.columns = [f"Population {y}" for y in pops.columns]
    return base.merge(pops.reset_index(), on="P-code")


def test_wide_population_columns_merge_to_the_same_table(tmp_path, cfg):
    trends = make_trends()
    # ในไฟล์จริง density = Population / Area ตรวจกับ trends_weather_merged.csv แล้ว ข้อมูลสังเคราะห์ต้องเป็นแบบเดียวกัน
    trends["Population density (per km²)"] = trends["Population"] / trends["Area (km²)"]
    weekly, _ = split_sheets(trends)
    xlsx = write_xlsx(tmp_path / "TRENDS.xlsx", weekly, wide_province_sheet(trends))
    got = merge_trends_sheets(xlsx, cfg)
    assert_frame_equal(got.reset_index(drop=True), trends.reset_index(drop=True), check_dtype=False)
    assert list(got.columns[-5:]) == PROV_EXTRA


def test_inspect_trends_reports_wide_layout_as_supported(tmp_path, cfg, capsys):
    from dengue_pipeline.cli import inspect_trends
    trends = make_trends()
    weekly, _ = split_sheets(trends)
    xlsx = write_xlsx(tmp_path / "TRENDS.xlsx", weekly, wide_province_sheet(trends))
    inspect_trends(cfg, str(xlsx))
    out = capsys.readouterr().out
    assert "แปลงคอลัมน์ Population YYYY" in out and "-> รองรับ" in out
