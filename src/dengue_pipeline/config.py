"""โหลด config/pipeline.toml และกำหนด path ของแต่ละชั้นข้อมูล"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path
    data: dict

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """path = ไฟล์ toml; ไม่ระบุ = config/pipeline.toml ใต้โฟลเดอร์ปัจจุบัน"""
        cfg_path = Path(path) if path else Path.cwd() / "config" / "pipeline.toml"
        cfg_path = cfg_path.resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ config: {cfg_path}")
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
        return cls(root=cfg_path.parent.parent, data=data)

    # --- เข้าถึงค่า -------------------------------------------------
    def __getitem__(self, section: str) -> dict:
        return self.data[section]

    # --- โฟลเดอร์ของแต่ละชั้น ---------------------------------------
    def _p(self, key: str) -> Path:
        p = Path(self.data["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self._p("data_dir")

    @property
    def reports_dir(self) -> Path:
        return self._p("reports_dir")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def validated_dir(self) -> Path:
        return self.data_dir / "validated"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    def rel(self, p: Path) -> str:
        """path แบบสัมพัทธ์ต่อ data_dir สำหรับบันทึกใน run.json"""
        return str(Path(p).resolve().relative_to(self.data_dir.resolve()))

    def abs(self, rel_path: str) -> Path:
        return self.data_dir / rel_path
