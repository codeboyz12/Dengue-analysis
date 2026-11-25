"""สถานะของแต่ละรอบ (run) และบันทึกลง data/runs/<run_id>/run.json

สามสถานะของแต่ละขั้น: success / incomplete / failed
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from dengue_pipeline.config import Config
from dengue_pipeline.io import read_json, write_json

STAGES = ["ingest", "validate-source", "transform", "validate-merged", "load"]


class Status(str, Enum):
    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunState:
    run_id: str
    raw: dict = field(default_factory=dict)       # source -> path (สัมพัทธ์ต่อ data_dir)
    staging: str | None = None
    validated: str | None = None
    stages: dict = field(default_factory=dict)    # stage -> {status, started, ended, message}
    rules: dict = field(default_factory=dict)     # stage -> {pass, fail, skip}

    # --- ไฟล์ ----------------------------------------------------------
    @staticmethod
    def path(cfg: Config, run_id: str):
        return cfg.runs_dir / run_id / "run.json"

    def save(self, cfg: Config) -> None:
        write_json(self.path(cfg, self.run_id), asdict(self))

    @classmethod
    def load(cls, cfg: Config, run_id: str) -> "RunState":
        p = cls.path(cfg, run_id)
        if not p.exists():
            raise FileNotFoundError(f"ไม่พบ run: {run_id}")
        return cls(**read_json(p))

    @classmethod
    def latest(cls, cfg: Config) -> "RunState | None":
        if not cfg.runs_dir.exists():
            return None
        ids = sorted(d.name for d in cfg.runs_dir.iterdir() if (d / "run.json").exists())
        return cls.load(cfg, ids[-1]) if ids else None

    # --- สถานะของขั้น ----------------------------------------------------
    def stage_status(self, stage: str) -> str | None:
        return self.stages.get(stage, {}).get("status")

    def mark(self, cfg: Config, stage: str, status: Status, message: str, started: str) -> None:
        ended = utc_now()
        self.stages[stage] = {
            "status": status.value, "started": started, "ended": ended, "message": message,
        }
        self.save(cfg)
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.reports_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_id": self.run_id, "stage": stage, "status": status.value,
                "started": started, "ended": ended, "message": message,
            }, ensure_ascii=False) + "\n")
