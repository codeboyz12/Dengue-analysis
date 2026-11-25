"""โหลดและตรวจ contract ของ CSV ที่ส่งออก (config/contract_v2.toml)

contract นี้กำหนดเฉพาะโครงสร้างตารางฐาน (คอลัมน์ ลำดับ คีย์) ไม่นิยามป้ายหรือฟีเจอร์ — ส่วนนั้นอยู่ใน notebook
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dengue_pipeline.errors import StageError


@dataclass(frozen=True)
class Contract:
    version: int
    unit: str
    keys: list
    columns: list
    forbidden_patterns: list

    @property
    def output_columns(self) -> list[str]:
        return list(self.columns)

    @classmethod
    def load(cls, path: str | Path) -> "Contract":
        p = Path(path)
        if not p.exists():
            raise StageError(f"ไม่พบไฟล์ contract: {p}")
        with open(p, "rb") as f:
            d = tomllib.load(f)
        try:
            c = cls(version=int(d["version"]), unit=d["unit"], keys=list(d["keys"]),
                    columns=list(d["columns"]),
                    forbidden_patterns=list(d.get("leakage", {}).get("forbidden_patterns", [])))
        except KeyError as ex:
            raise StageError(f"contract {p.name} ขาดฟิลด์ {ex}") from ex
        c.validate()
        return c

    def validate(self) -> None:
        problems: list[str] = []
        if not self.keys or self.keys[0] != "P-code":
            problems.append("keys ตัวแรกต้องเป็น P-code")
        if len(set(self.columns)) != len(self.columns):
            problems.append("columns มีชื่อซ้ำ")
        missing = [k for k in self.keys if k not in self.columns]
        if missing:
            problems.append(f"keys ต้องอยู่ใน columns (ขาด {missing})")
        for pat in self.forbidden_patterns:
            try:
                re.compile(pat)
            except re.error as ex:
                problems.append(f"forbidden_patterns ใช้ regex ไม่ได้: {pat!r} ({ex})")
        if problems:
            raise StageError("contract ไม่ถูกต้อง: " + "; ".join(problems))
