"""ตรวจกฎและ contract กับข้อมูลจริง (ข้ามอัตโนมัติถ้าไม่ได้ตั้งตัวแปรแวดล้อม)

  TRENDS_WEATHER_CSV=/path/to/trends_weather_merged.csv python -m pytest tests/test_real_data.py -q
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from dengue_pipeline.contract import Contract
from dengue_pipeline.validation.checks_export import check_export
from dengue_pipeline.validation.checks_merged import check_merged
from dengue_pipeline.validation.checks_source import check_trends

REAL = os.environ.get("TRENDS_WEATHER_CSV")
pytestmark = pytest.mark.skipif(not REAL or not Path(REAL).exists(),
                                reason="ตั้ง TRENDS_WEATHER_CSV=path/to/trends_weather_merged.csv เพื่อรัน")
CONTRACT = Path(__file__).resolve().parents[1] / "config" / "contract_v2.toml"


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    return pd.read_csv(REAL, parse_dates=["Week start", "Week end"])


def test_source_rules_pass_on_the_real_table(real):
    res = check_trends(real, 77, 0.001, None)
    assert [r.rule_id for r in res if r.blocking_failure] == []


def test_merged_rules_pass_on_the_real_table(real):
    res = check_merged(real, {"rows_before": None, "weekly_rows": None}, None, 4, 342, 3)
    assert [r.rule_id for r in res if r.blocking_failure] == []
    assert {r.rule_id for r in res if r.outcome == "skip"} == {"MRG-02", "MRG-05"}   # ไม่มีตารางรายวัน/meta ให้เทียบ


def test_contract_v2_matches_the_real_table_exactly(real):
    contract = Contract.load(CONTRACT)
    assert list(real.columns) == contract.columns          # ชื่อและลำดับตรงกับไฟล์ที่ notebook ใช้
    table = real.sort_values(contract.keys).reset_index(drop=True)
    assert table.shape == (40579, 44)                       # assert ใน section 2 ของ notebook
    res = check_export(table, contract)
    assert [r.rule_id for r in res if r.blocking_failure] == []
