"""ตัวรัน: สั่งรันทั้งสายหรือเฉพาะขั้น แปลงผลของแต่ละขั้นเป็น 3 สถานะ และจัดการ quarantine

รหัสออก (exit code): 0 สำเร็จ | 1 ตรวจไม่ผ่านหรือผิดพลาด | 2 ยังไม่ครบ (รันซ้ำเพื่อต่อ)
นโยบาย: กฎ BLOCK ข้อใดไม่ผ่าน -> หยุดทั้งสาย ข้อมูลรอบนั้นไป quarantine ชั้น validated ไม่ถูกแตะ
"""
from __future__ import annotations

import shutil
import traceback
from dataclasses import dataclass, field
from typing import Callable

from dengue_pipeline.config import Config
from dengue_pipeline.errors import IncompleteRun, StageError, ValidationFailed
from dengue_pipeline.io import write_json
from dengue_pipeline.rules import RuleResult
from dengue_pipeline.runstate import STAGES, RunState, Status, new_run_id, utc_now
from dengue_pipeline.stages import ingest, load, transform, validate_merged, validate_source

STAGE_FUNCS: dict[str, Callable] = {
    "ingest": ingest.run,
    "validate-source": validate_source.run,
    "transform": transform.run,
    "validate-merged": validate_merged.run,
    "load": load.run,
}

EXIT = {Status.SUCCESS: 0, Status.FAILED: 1, Status.INCOMPLETE: 2}


@dataclass
class RunArgs:
    trends_input: str | None = None
    weather_input: str | None = None
    # จุดเสียบสำหรับเทสและการจำลอง (ไม่เปิดใน CLI)
    fetch: Callable = field(default=None)
    sleep_s: float | None = None

    def __post_init__(self):
        if self.fetch is None:
            from dengue_pipeline.weather_fetch import fetch_chunk
            self.fetch = fetch_chunk


def quarantine(cfg: Config, state: RunState, stage: str, results: list[RuleResult]) -> str:
    """เก็บผลตรวจและข้อมูลที่ staging ของรอบนี้ไว้หาสาเหตุ ส่วน raw ห้ามแก้จึงไม่ย้าย แค่ชี้ไป"""
    qdir = cfg.quarantine_dir / state.run_id / stage
    qdir.mkdir(parents=True, exist_ok=True)
    write_json(qdir / "report.json", {
        "run_id": state.run_id, "stage": stage, "quarantined_at": utc_now(),
        "raw": state.raw,
        "results": [r.__dict__ for r in results],
    })
    if state.staging and cfg.abs(state.staging).exists():
        shutil.copytree(cfg.abs(state.staging), qdir / "staging", dirs_exist_ok=True)
    return str(qdir)


def _execute(cfg: Config, state: RunState, stage: str, args: RunArgs) -> Status:
    started = utc_now()
    print(f"▶ {stage}")
    try:
        msg = STAGE_FUNCS[stage](cfg, state, args)
        status = Status.SUCCESS
    except IncompleteRun as ex:
        msg, status = str(ex), Status.INCOMPLETE
    except ValidationFailed as ex:
        where = quarantine(cfg, state, stage, ex.results)
        failed = [f"{r.rule_id}: {r.measured}" for r in ex.results if r.blocking_failure]
        msg = f"ตรวจไม่ผ่าน ({'; '.join(failed)}) เก็บไว้ที่ {where}"
        status = Status.FAILED
    except StageError as ex:
        msg, status = str(ex), Status.FAILED
    except Exception as ex:  # ข้อผิดพลาดที่ไม่คาด: บันทึกและหยุด
        traceback.print_exc()
        msg, status = f"{type(ex).__name__}: {ex}", Status.FAILED
    state.mark(cfg, stage, status, msg, started)
    icon = {"success": "✔", "incomplete": "…", "failed": "✘"}[status.value]
    print(f"  {icon} {status.value}: {msg}")
    return status


def run_pipeline(cfg: Config, stage: str = "all", run_id: str | None = None,
                 args: RunArgs | None = None) -> int:
    args = args or RunArgs()
    if stage not in STAGES and stage != "all":
        raise ValueError(f"ไม่รู้จักขั้น: {stage}")

    if run_id:
        state = RunState.load(cfg, run_id)
    elif stage in ("all", "ingest"):
        state = RunState(run_id=new_run_id())
        state.save(cfg)
    else:
        state = RunState.latest(cfg)
        if state is None:
            print("ยังไม่มีรอบที่รันไว้ เริ่มด้วย: run --stage ingest (หรือ all)")
            return 1
    print(f"run_id = {state.run_id}")

    todo = STAGES if stage == "all" else [stage]
    last = Status.SUCCESS
    for s in todo:
        if stage == "all" and state.stage_status(s) == Status.SUCCESS.value:
            print(f"▶ {s}\n  ↷ ข้าม (สำเร็จแล้วในรอบนี้)")
            continue
        prev = STAGES[STAGES.index(s) - 1] if STAGES.index(s) > 0 else None
        if prev and state.stage_status(prev) != Status.SUCCESS.value:
            print(f"✘ {s}: ขั้นก่อนหน้า ({prev}) ยังไม่สำเร็จในรอบนี้ "
                  f"(สถานะ: {state.stage_status(prev) or 'ยังไม่ได้รัน'})")
            return 1
        last = _execute(cfg, state, s, args)
        if last != Status.SUCCESS:
            break
    return EXIT[last]
