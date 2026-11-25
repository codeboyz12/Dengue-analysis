"""ปลายทางแบบ DuckDB: เขียน/อ่านกลับตรงเดิม, ไม่เขียนทับ, rollback, และใช้ในสายจริงได้เหมือน FileSink"""
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dengue_pipeline.cli import main
from dengue_pipeline.errors import StageError
from dengue_pipeline.pipeline import RunArgs, run_pipeline
from dengue_pipeline.sink_duckdb import DuckDBSink
from dengue_pipeline.sinks import FileSink, get_sink
from dengue_pipeline.stages.export import export_csv

from conftest import make_trends, write_config
from test_export import CONTRACT


@pytest.fixture
def sink(dcfg) -> DuckDBSink:
    return DuckDBSink(dcfg)


def tables(seed=1):
    t = make_trends(seed=seed)
    d = pd.DataFrame({"P-code": ["TH10"] * 3, "date": pd.date_range("2019-01-01", periods=3),
                      "precip_mm": [0.0, 1.5, 2.25], "t_max_c": [33.1, 34.0, 35.5], "t_min_c": [25.0, 24.5, 26.0]})
    return {"trends_weather": t, "weather_daily": d}


def test_get_sink_picks_the_configured_backend(tmp_path):
    assert isinstance(get_sink(write_config(tmp_path / "a", sink="file")), FileSink)
    assert isinstance(get_sink(write_config(tmp_path / "b", sink="duckdb")), DuckDBSink)
    with pytest.raises(ValueError, match="sink"):
        get_sink(write_config(tmp_path / "c", sink="oracle"))


def test_empty_database_has_no_versions(sink):
    assert sink.versions() == [] and sink.latest_version() is None and sink.latest_metadata() is None
    with pytest.raises(FileNotFoundError):
        sink.read("20260101T000000Z", "trends_weather")


def test_write_then_read_returns_the_same_tables_in_the_same_order(sink):
    t = tables()
    shuffled = t["trends_weather"].sample(frac=1, random_state=3)          # ลำดับแถวไม่ใช่ลำดับตามคีย์
    sink.write("20260101T000000Z", {**t, "trends_weather": shuffled}, {"rows": {"trends_weather": len(shuffled)}})
    got = sink.read("20260101T000000Z", "trends_weather")
    assert_frame_equal(got, shuffled.reset_index(drop=True), check_dtype=False)
    assert_frame_equal(sink.read("20260101T000000Z", "weather_daily"), t["weather_daily"], check_dtype=False)


def test_values_are_stored_exactly_not_as_float32(sink):
    df = pd.DataFrame({"P-code": ["A"], "Week start": pd.to_datetime(["2019-01-06"]), "x": [30.123456789012]})
    sink.write("v1", {"trends_weather": df}, {})
    assert sink.read("v1", "trends_weather")["x"].iloc[0] == 30.123456789012


def test_column_selection_reads_only_those_columns(sink):
    sink.write("v1", tables(), {})
    got = sink.read("v1", "trends_weather", columns=["P-code", "Area (km²)", "Dengue total"])
    assert list(got.columns) == ["P-code", "Area (km²)", "Dengue total"] and len(got) == len(make_trends())
    with pytest.raises(KeyError, match="Nope"):
        sink.read("v1", "trends_weather", columns=["Nope"])


def test_existing_version_is_never_overwritten(sink):
    sink.write("v1", tables(), {})
    with pytest.raises(FileExistsError):
        sink.write("v1", tables(seed=2), {})
    assert_frame_equal(sink.read("v1", "trends_weather"), tables()["trends_weather"], check_dtype=False)


def test_failed_write_leaves_nothing_behind(sink):
    sink.write("v1", tables(), {})
    bad = tables(seed=2)
    bad["weather_daily"] = bad["weather_daily"].rename(columns={"precip_mm": "rain"})   # ตารางที่สองไม่ตรงโครงสร้าง
    with pytest.raises(StageError, match="weather_daily"):
        sink.write("v2", bad, {})
    assert sink.versions() == ["v1"]
    import duckdb
    con = duckdb.connect(str(sink.path), read_only=True)
    assert con.execute("select count(*) from mart.trends_weather where data_version='v2'").fetchone()[0] == 0
    con.close()
    sink.write("v3", tables(seed=3), {})                                   # ยังเขียนต่อได้ตามปกติ
    assert sink.versions() == ["v1", "v3"]


def test_latest_version_metadata_and_views(sink):
    sink.write("20260101T000000Z", tables(1), {"run_id": "a", "rows": {"trends_weather": 1}})
    sink.write("20260102T000000Z", tables(2), {"run_id": "b", "note": "ภาษาไทย"})
    assert sink.latest_version() == "20260102T000000Z"
    assert sink.latest_metadata() == {"run_id": "b", "note": "ภาษาไทย"}
    import duckdb
    con = duckdb.connect(str(sink.path), read_only=True)
    latest = con.execute("select * from mart.trends_weather_latest").df()
    con.close()
    assert "_ord" not in latest.columns and "data_version" in latest.columns
    assert set(latest["data_version"]) == {"20260102T000000Z"}
    assert_frame_equal(latest.drop(columns="data_version"), sink.read("20260102T000000Z", "trends_weather"), check_dtype=False)


def test_full_pipeline_with_duckdb_sink_then_export_and_audit(dcfg, input_files):
    dcfg.data["export"] = {"contract": str(CONTRACT)}
    tp, wp = input_files
    assert run_pipeline(dcfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp))) == 0
    sink = get_sink(dcfg)
    v = sink.latest_version()
    assert v and not dcfg.validated_dir.exists()                           # ไม่มีโฟลเดอร์ validated/ เมื่อใช้ฐานข้อมูล
    assert sink.latest_metadata()["rows"]["trends_weather"] == len(make_trends())
    meta = export_csv(dcfg)
    assert meta["data_version"] == v and meta["rows"] == len(make_trends())
    import duckdb
    con = duckdb.connect(str(sink.path), read_only=True)
    n_db = con.execute("select count(*) from ops.audit_log").fetchone()[0]
    con.close()
    assert n_db == len(pd.read_csv(dcfg.reports_dir / "audit_log.csv"))    # สำเนาตรงกับ CSV ทุกแถว


def test_second_run_still_checks_row_count_against_the_database_version(dcfg, tmp_path, input_files, trends, weather):
    tp, wp = input_files
    assert run_pipeline(dcfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp))) == 0
    sp = tmp_path / "smaller.csv"
    trends[trends["Week start"] < trends["Week start"].max()].to_csv(sp, index=False)
    import time; time.sleep(1.1)
    assert run_pipeline(dcfg, "all", args=RunArgs(trends_input=str(sp), weather_input=str(wp))) == 1
    assert len(get_sink(dcfg).versions()) == 1


def test_migrate_sink_moves_file_versions_into_the_database_once(tmp_path, input_files, capsys):
    cfg_file = write_config(tmp_path, sink="file")
    tp, wp = input_files
    assert run_pipeline(cfg_file, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp))) == 0
    version = FileSink(cfg_file).latest_version()
    cfgpath = str(cfg_file.root / "config" / "pipeline.toml")
    (cfg_file.root / "config" / "pipeline.toml").write_text(
        (cfg_file.root / "config" / "pipeline.toml").read_text().replace('sink = "file"', 'sink = "duckdb"'), encoding="utf-8")
    assert main(["--config", cfgpath, "migrate-sink"]) == 0
    assert version in capsys.readouterr().out
    cfg_db = write_config(tmp_path, sink="duckdb")
    db = DuckDBSink(cfg_db)
    assert db.versions() == [version]
    assert_frame_equal(db.read(version, "trends_weather"), FileSink(cfg_db).read(version, "trends_weather"), check_dtype=False)
    assert json.dumps(db.latest_metadata()["rows"], sort_keys=True) == json.dumps(
        FileSink(cfg_db).read_metadata(version)["rows"], sort_keys=True)
    assert main(["--config", cfgpath, "migrate-sink"]) == 0
    assert "ไม่มีเวอร์ชันใหม่" in capsys.readouterr().out                  # ทำซ้ำได้ ไม่ย้ายซ้ำ
