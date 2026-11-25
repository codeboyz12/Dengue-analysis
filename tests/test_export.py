"""ขั้น 6 export: ตารางฐาน 44 คอลัมน์ตาม contract v2 (ให้ notebook โหลดได้ทันที)"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from dengue_pipeline.cli import main
from dengue_pipeline.contract import Contract
from dengue_pipeline.errors import StageError
from dengue_pipeline.pipeline import RunArgs, run_pipeline
from dengue_pipeline.sinks import get_sink
from dengue_pipeline.stages.export import export_csv
from dengue_pipeline.validation.checks_export import check_export

CONTRACT = Path(__file__).resolve().parents[1] / "config" / "contract_v2.toml"


@pytest.fixture
def contract() -> Contract:
    return Contract.load(CONTRACT)


@pytest.fixture
def vcfg(cfg, input_files):
    """config ที่มีข้อมูลผ่านตรวจในชั้น validated แล้ว (ผ่านสายจริงทั้งสาย)"""
    cfg.data["export"] = {"contract": str(CONTRACT)}
    tp, wp = input_files
    assert run_pipeline(cfg, "all", args=RunArgs(trends_input=str(tp), weather_input=str(wp))) == 0
    return cfg


def exported(vcfg) -> tuple[dict, pd.DataFrame]:
    meta = export_csv(vcfg)
    return meta, pd.read_csv(meta["path"], parse_dates=["Week start", "Week end"])


def validated(vcfg) -> pd.DataFrame:
    sink = get_sink(vcfg)
    return sink.read(sink.latest_version(), "trends_weather")


# --- contract ----------------------------------------------------------------

def test_contract_is_the_44_column_base_table(contract):
    assert contract.version == 2 and len(contract.columns) == 44
    assert "D" not in contract.columns            # ป้ายนิยามใน notebook ที่เดียว


def test_no_contract_column_matches_a_forbidden_pattern(contract):
    hits = [c for c in contract.columns for p in contract.forbidden_patterns if re.search(p, c)]
    assert hits == []


@pytest.mark.parametrize("old,new,msg", [
    ('keys = ["P-code", "Week start"]', 'keys = ["Week start", "P-code"]', "P-code"),
    ('"Country", "Year"', '"Country", "Country", "Year"', "ซ้ำ"),
    ('keys = ["P-code", "Week start"]', 'keys = ["P-code", "Nope"]', "Nope"),
    ("'(?i)anomal'", "'(?i)anomal('", "regex"),
])
def test_contract_rejects_bad_definitions(tmp_path, old, new, msg):
    text = CONTRACT.read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(StageError, match=msg):
        Contract.load(bad)


# --- export -------------------------------------------------------------------

def test_export_writes_the_validated_table_unchanged_with_metadata(vcfg, contract):
    meta, out = exported(vcfg)
    src = validated(vcfg).sort_values(contract.keys).reset_index(drop=True)
    assert list(out.columns) == contract.columns and out.shape == (48, 44)
    assert not out.isna().any().any()
    pd.testing.assert_frame_equal(out, src[contract.columns], check_dtype=False)
    assert Path(meta["path"]).name == "trends_weather_merged.csv"
    assert meta["data_version"] == get_sink(vcfg).latest_version() and meta["contract_version"] == 2
    on_disk = json.loads((Path(meta["path"]).parent / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk["sha256"] == hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest()
    log = pd.read_csv(vcfg.reports_dir / "audit_log.csv")
    ex = log[log["stage"] == "export"]
    assert sorted(ex["rule_id"]) == ["EXP-01", "EXP-02", "EXP-03", "LKG-02"] and (ex["outcome"] == "pass").all()


def test_exported_file_passes_the_notebooks_own_load_checks(vcfg):
    _, out = exported(vcfg)
    assert out.shape[1] == 44 and not out.isna().any().any()                      # assert ใน section 2
    assert not out.duplicated(["P-code", "Week start"]).any()
    assert not out.duplicated(["P-code", "Year", "Epi week"]).any()
    assert (out["weather_days"] == out["week_days"]).all()


def test_export_is_deterministic_for_the_same_data_and_contract(vcfg):
    m1, _ = exported(vcfg)
    m2, _ = exported(vcfg)
    assert m1["sha256"] == m2["sha256"]


# --- กฎ -----------------------------------------------------------------------

def test_exp01_flags_missing_extra_and_reordered_columns(vcfg, contract):
    table = validated(vcfg)[contract.columns].sort_values(contract.keys)
    ok = {r.rule_id: r for r in check_export(table, contract)}
    assert not any(r.blocking_failure for r in ok.values())
    for bad in (table.drop(columns=["Population"]), table.assign(D=0.0),
                table[[contract.columns[1], contract.columns[0], *contract.columns[2:]]]):
        assert {r.rule_id: r for r in check_export(bad, contract)}["EXP-01"].blocking_failure


def test_exp02_flags_null_values(vcfg, contract):
    table = validated(vcfg)[contract.columns].sort_values(contract.keys).reset_index(drop=True)
    table.loc[3, "Population"] = None
    res = {r.rule_id: r for r in check_export(table, contract)}
    assert res["EXP-02"].blocking_failure and "Population" in res["EXP-02"].detail


def test_exp03_flags_duplicate_keys_and_unsorted_rows(vcfg, contract):
    table = validated(vcfg)[contract.columns].sort_values(contract.keys).reset_index(drop=True)
    assert {r.rule_id: r for r in check_export(table.iloc[::-1], contract)}["EXP-03"].blocking_failure
    dup = pd.concat([table, table.iloc[[0]]]).sort_values(contract.keys)
    assert {r.rule_id: r for r in check_export(dup, contract)}["EXP-03"].blocking_failure


def test_lkg02_flags_columns_computed_across_years(vcfg, contract):
    table = validated(vcfg)[contract.columns].sort_values(contract.keys)
    res = {r.rule_id: r for r in check_export(table.assign(dengue_rate_anomaly=0.0), contract)}
    assert res["LKG-02"].blocking_failure and "dengue_rate_anomaly" in res["LKG-02"].detail


# --- ขอบเขตการใช้งาน --------------------------------------------------------------

def test_export_without_validated_data_says_what_to_do(cfg):
    cfg.data["export"] = {"contract": str(CONTRACT)}
    with pytest.raises(StageError, match="validated"):
        export_csv(cfg)


def test_export_stops_when_validated_table_lacks_contract_columns(vcfg, tmp_path):
    wide = tmp_path / "wide.toml"
    wide.write_text(CONTRACT.read_text(encoding="utf-8").replace('"rain_wk_mm_lag4",', '"rain_wk_mm_lag4", "rain_wk_mm_lag5",'),
                    encoding="utf-8")
    with pytest.raises(StageError, match="rain_wk_mm_lag5"):
        export_csv(vcfg, contract_path=str(wide))


def test_failed_rule_writes_nothing(vcfg, tmp_path):
    strict = tmp_path / "strict.toml"
    strict.write_text(CONTRACT.read_text(encoding="utf-8").replace("'(?i)outbreak'", "'(?i)outbreak', '^Province$'"),
                      encoding="utf-8")
    from dengue_pipeline.errors import ValidationFailed
    with pytest.raises(ValidationFailed):
        export_csv(vcfg, contract_path=str(strict))
    assert not list(vcfg.exports_dir.rglob("*.csv"))


def test_cli_export_returns_zero_and_writes_file(vcfg, capsys):
    code = main(["--config", str(vcfg.root / "config" / "pipeline.toml"), "export", "--contract", str(CONTRACT)])
    assert code == 0 and "trends_weather_merged.csv" in capsys.readouterr().out
