"""โครงของกฎตรวจ: ผลลัพธ์ของแต่ละกฎ, ระดับความรุนแรง, และตารางตรวจ (audit log)

ระดับ  BLOCK = ไม่ผ่านแล้วหยุดทั้งสาย
       INFO  = บันทึกค่าอย่างเดียว ไม่หยุด
ผลลัพธ์ pass | fail | skip  (skip = ตรวจไม่ได้เพราะไม่มีคอลัมน์/ข้อมูลที่กฎต้องใช้ ถูกบันทึกไว้เสมอ ไม่ถือว่าผ่านเงียบๆ)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

from dengue_pipeline.config import Config
from dengue_pipeline.errors import ValidationFailed
from dengue_pipeline.runstate import utc_now

BLOCK = "BLOCK"
INFO = "INFO"


@dataclass
class RuleResult:
    rule_id: str
    severity: str
    outcome: str          # pass | fail | skip
    measured: str
    detail: str = ""

    @property
    def blocking_failure(self) -> bool:
        return self.outcome == "fail" and self.severity == BLOCK


def check(rule_id: str, ok: bool, measured, detail: str = "", severity: str = BLOCK) -> RuleResult:
    return RuleResult(rule_id, severity, "pass" if ok else "fail", str(measured), detail)


def skip(rule_id: str, why: str, severity: str = BLOCK) -> RuleResult:
    return RuleResult(rule_id, severity, "skip", "-", why)


def info(rule_id: str, measured, detail: str = "") -> RuleResult:
    return RuleResult(rule_id, INFO, "pass", str(measured), detail)


def write_audit(cfg: Config, run_id: str, stage: str, results: list[RuleResult]) -> None:
    """ต่อท้าย reports/audit_log.csv: ทุกกฎ ทุกรอบ (รหัสกฎ, ผล, ค่าที่วัดได้, เวลา)"""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.reports_dir / "audit_log.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "stage", "rule_id", "severity", "outcome",
                        "measured", "detail", "checked_at"])
        now = utc_now()
        rows = [(run_id, stage, r.rule_id, r.severity, r.outcome, r.measured, r.detail, now) for r in results]
        for row in rows:
            w.writerow(list(row))
    from dengue_pipeline.sink_duckdb import mirror_audit       # สำเนาลงฐานข้อมูลเมื่อใช้ sink แบบ duckdb
    mirror_audit(cfg, [tuple("" if v is None else str(v) for v in row) for row in rows])


def raise_if_blocked(stage: str, results: list[RuleResult]) -> None:
    if any(r.blocking_failure for r in results):
        raise ValidationFailed(stage, results)


def summarize(results: list[RuleResult]) -> dict:
    out = {"pass": 0, "fail": 0, "skip": 0}
    for r in results:
        out[r.outcome] += 1
    return out
