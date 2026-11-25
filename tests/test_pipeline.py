import json

import pandas as pd
import pytest

from dengue_pipeline.cli import main
from dengue_pipeline.errors import StageError
from dengue_pipeline.io import read_json
from dengue_pipeline.pipeline import RunArgs, run_pipeline
from dengue_pipeline.runstate import RunState

from conftest import fake_fetch_factory, make_trends, write_config


def audit(cfg):
    return pd.read_csv(cfg.reports_dir / "audit_log.csv")


def test_full_run_with_local_files_succeeds_and_writes_validated(cfg, input_files):
    tp, wp = input_files
    code = run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp)))
    assert code == 0
    state = RunState.latest(cfg)
    assert [state.stage_status(s) for s in ["ingest", "validate-source", "transform", "validate-merged", "load"]] \
        == ["success"] * 5
    vdir = cfg.abs(state.validated)
    meta = read_json(vdir / "metadata.json")
    assert meta["run_id"] == state.run_id
    assert meta["rows"]["trends_weather"] == len(make_trends())
    assert meta["source_rows"]["trends"] == len(make_trends())
    assert meta["raw"]["trends"]["sha256"] and meta["raw"]["weather"]["window"]["start"] == "2018-11-25"
    merged = pd.read_parquet(vdir / "trends_weather.parquet")
    assert merged.shape[1] == 44
    log = audit(cfg)
    assert set(log["stage"]) == {"validate-source", "validate-merged"}
    assert (log["outcome"] == "fail").sum() == 0
    # MRG-07 ตัดการเทียบกับ weekly sheet เพราะอินพุตเป็น csv: ต้องไม่ใช่การข้ามเงียบ (ผ่านพร้อมหมายเหตุ)
    mrg07 = log[log["rule_id"] == "MRG-07"].iloc[0]
    assert mrg07["outcome"] == "pass" and "csv" in str(mrg07["detail"])


def test_raw_snapshots_hold_checksums_and_original_bytes(cfg, input_files):
    tp, wp = input_files
    run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp)))
    state = RunState.latest(cfg)
    tdir = cfg.abs(state.raw["trends"])
    assert (tdir / "trends_merged.csv").read_bytes() == tp.read_bytes()
    assert read_json(tdir / "manifest.json")["kind"] == "csv"


def test_validation_failure_stops_pipeline_quarantines_and_leaves_validated_untouched(cfg, tmp_path, trends, weather):
    bad = trends.copy()
    bad.loc[5, "Dengue total"] += 1                      # TRD-03 ต้องไม่ผ่าน
    tp, wp = tmp_path / "bad_trends.csv", tmp_path / "weather_daily.csv"
    bad.to_csv(tp, index=False)
    weather.to_csv(wp, index=False)

    code = run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp)))
    assert code == 1
    state = RunState.latest(cfg)
    assert state.stage_status("validate-source") == "failed"
    assert state.stage_status("transform") is None       # หยุดทั้งสาย: ไม่ได้รันขั้นถัดไป
    assert state.validated is None and not cfg.validated_dir.exists()
    report = read_json(cfg.quarantine_dir / state.run_id / "validate-source" / "report.json")
    assert {r["rule_id"] for r in report["results"] if r["outcome"] == "fail"} >= {"TRD-03"}
    assert "TRD-03" in state.stages["validate-source"]["message"]
    assert (audit(cfg)["outcome"] == "fail").any()


def test_cannot_run_next_stage_after_failed_stage(cfg, tmp_path, trends, weather):
    bad = trends.copy()
    bad.loc[5, "Dengue total"] += 1
    tp, wp = tmp_path / "bad_trends.csv", tmp_path / "weather_daily.csv"
    bad.to_csv(tp, index=False)
    weather.to_csv(wp, index=False)
    run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp)))
    state = RunState.latest(cfg)
    assert run_pipeline(cfg, "transform", run_id=state.run_id) == 1
    assert RunState.latest(cfg).stage_status("transform") is None


def test_merged_validation_failure_quarantines_staging(cfg, input_files, monkeypatch):
    tp, wp = input_files
    import dengue_pipeline.stages.transform as transform_mod
    real = transform_mod.attach_weather

    def broken(trends, weather, lags):
        out = real(trends, weather, lags)
        out["rain_wk_mm_lag2"] = out["rain_wk_mm_lag2"] + 1.0     # lag ผิด -> MRG-05
        return out

    monkeypatch.setattr(transform_mod, "attach_weather", broken)
    code = run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp)))
    assert code == 1
    state = RunState.latest(cfg)
    assert state.stage_status("transform") == "success"
    assert state.stage_status("validate-merged") == "failed"
    assert state.stage_status("load") is None
    qdir = cfg.quarantine_dir / state.run_id / "validate-merged"
    assert (qdir / "report.json").exists() and (qdir / "staging" / "trends_weather.parquet").exists()
    assert not cfg.validated_dir.exists()


def test_quota_exhaustion_is_incomplete_not_failed_and_resumes_from_cache(cfg, tmp_path, trends):
    tp = tmp_path / "trends_merged.csv"
    trends.to_csv(tp, index=False)

    calls = []
    # โควตาหมดหลังดึงสำเร็จ 2 ครั้ง (จังหวัดแรก 2 ปี = 2018, 2019)
    args = RunArgs(trends_input=str(tp), fetch=fake_fetch_factory(fail_after=3, calls=calls), sleep_s=0)
    code = run_pipeline(cfg, "all", args=args)
    assert code == 2
    state = RunState.latest(cfg)
    assert state.stage_status("ingest") == "incomplete"
    assert "weather" not in state.raw                          # ไม่สร้าง raw อากาศครึ่งๆ กลางๆ
    assert "trends" in state.raw
    assert len(list((cfg.cache_dir / "weather").glob("*.csv"))) == 3   # ของที่ดึงได้อยู่ใน cache
    assert state.stage_status("validate-source") is None       # ไม่ตรวจเนื้อข้อมูลของชุดที่ไม่ครบ
    assert not (cfg.quarantine_dir.exists())                   # ไม่ใช่ความผิดของข้อมูล

    # วันถัดมา: โควตากลับมา ต่อจากเดิม ไม่ดึง chunk ที่มีแล้วซ้ำ
    calls2 = []
    args2 = RunArgs(fetch=fake_fetch_factory(calls=calls2), sleep_s=0)
    code2 = run_pipeline(cfg, "all", run_id=state.run_id, args=args2)
    assert len(calls2) == 3 * 2 - 3                             # 3 จังหวัด x 2 ปี - 3 chunk ที่มีแล้ว
    # ข้อมูลสังเคราะห์ของตัวจำลองมีฝน/อุณหภูมิคงที่ จึงผ่านกฎตรวจ (ช่วงวันที่ครบตาม manifest)
    final = RunState.load(cfg, state.run_id)
    assert final.stage_status("ingest") == "success"
    assert final.stage_status("validate-source") == "success", final.stages["validate-source"]
    assert code2 == 0 and final.stage_status("load") == "success"


def test_xlsx_whose_province_sheet_has_no_population_stops_cleanly(cfg, tmp_path, trends, weather):
    xlsx = tmp_path / "TRENDS.xlsx"
    with pd.ExcelWriter(xlsx) as xw:
        trends[[c for c in trends.columns if c not in
                ("Centroid latitude", "Centroid longitude", "Area (km²)", "Population",
                 "Population density (per km²)")]].to_excel(xw, sheet_name="weekly data", index=False)
        trends[["P-code", "Province", "Centroid latitude", "Centroid longitude"]] \
            .drop_duplicates("P-code").to_excel(xw, sheet_name="province data", index=False)
    wp = tmp_path / "weather_daily.csv"
    weather.to_csv(wp, index=False)
    code = run_pipeline(cfg, "all", args=RunArgs(trends_input=str(xlsx), weather_input=str(wp)))
    assert code == 1
    state = RunState.latest(cfg)
    assert state.stage_status("validate-source") == "success"      # ตรวจชีท weekly ได้ (TRD-04 = skip)
    assert state.stage_status("transform") == "failed"
    assert "Population" in state.stages["transform"]["message"]
    log = audit(cfg)
    assert log[log["rule_id"] == "TRD-04"].iloc[0]["outcome"] == "skip"


def test_second_run_checks_row_count_against_previous_validated(cfg, tmp_path, input_files, trends, weather):
    tp, wp = input_files
    assert run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp))) == 0
    # รอบใหม่ที่ TRENDS แถวน้อยลง ต้องไม่ผ่าน SRC-03
    smaller = trends[trends["Week start"] < trends["Week start"].max()]
    sp = tmp_path / "smaller.csv"
    smaller.to_csv(sp, index=False)
    import time; time.sleep(1.1)       # run_id ใช้เวลาระดับวินาที
    assert run_pipeline(cfg, "all", args=RunArgs(trends_input=str(sp), weather_input=str(wp))) == 1
    state = RunState.latest(cfg)
    assert "SRC-03" in state.stages["validate-source"]["message"]
    assert len([d for d in cfg.validated_dir.iterdir() if d.is_dir()]) == 1   # ชุดเดิมไม่ถูกแตะ


def test_stage_without_existing_run_is_refused(cfg):
    assert run_pipeline(cfg, "transform") == 1


def test_missing_trends_input_is_a_clear_failure(cfg, tmp_path):
    code = run_pipeline(cfg, "ingest", args=RunArgs(trends_input=str(tmp_path / "nope.csv")))
    assert code == 1
    state = RunState.latest(cfg)
    assert "ไม่พบไฟล์ TRENDS" in state.stages["ingest"]["message"]


def test_cli_run_and_status(cfg, input_files, capsys):
    tp, wp = input_files
    cfgfile = str(cfg.root / "config" / "pipeline.toml")
    assert main(["--config", cfgfile, "run", "--trends-input", str(tp), "--weather-input", str(wp)]) == 0
    assert main(["--config", cfgfile, "status"]) == 0
    out = capsys.readouterr().out
    assert "validate-merged" in out and "success" in out and "validated" in out
    assert main(["--config", cfgfile, "export"]) == 1


def test_validated_version_is_never_overwritten(cfg, input_files):
    from dengue_pipeline.sinks import FileSink
    tp, wp = input_files
    run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp)))
    state = RunState.latest(cfg)
    with pytest.raises(FileExistsError):
        FileSink(cfg).write(state.run_id, {"x": pd.DataFrame({"a": [1]})}, {})
