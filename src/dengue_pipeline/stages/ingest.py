"""ขั้น 1 ingest: เก็บข้อมูลดิบตามที่ได้รับลงชั้น raw (ไม่แก้ค่า) พร้อม manifest

  raw/trends/<run_id>/  ไฟล์ TRENDS + manifest.json (sha256, ขนาด, เวลา, ที่มา)
  raw/weather/<run_id>/ weather_daily.csv + manifest.json (ขอบเขตวันที่ที่คาดหวัง ใช้ตรวจ WX-02)

รันซ้ำได้: ขั้นที่ทำเสร็จแล้วในรอบเดียวกันจะข้าม อากาศดึงต่อจาก cache
ถ้าโควตาหมดจะยกข้อยกเว้น IncompleteRun และไม่สร้าง raw ของอากาศครึ่งๆ กลางๆ
"""
from __future__ import annotations

import shutil
from pathlib import Path

from dengue_pipeline import weather_fetch
from dengue_pipeline.config import Config
from dengue_pipeline.errors import StageError
from dengue_pipeline.io import read_json, sha256_of, write_json
from dengue_pipeline.runstate import RunState, utc_now
from dengue_pipeline.trends_io import kind_of, load_provinces, load_weekly
from dengue_pipeline.validation.checks_source import weather_window


def _resolve_input(arg: str | None, cfg_value: str, cfg: Config) -> Path | None:
    if arg:
        return Path(arg).expanduser().resolve()
    if cfg_value:
        p = Path(cfg_value).expanduser()
        return p if p.is_absolute() else (cfg.root / p).resolve()
    return None


def _ingest_trends(cfg: Config, state: RunState, src: Path) -> Path:
    if not src.exists():
        raise StageError(f"ไม่พบไฟล์ TRENDS: {src} (ระบุด้วย --trends-input หรือ [trends].input ใน config)")
    dest_dir = cfg.raw_dir / "trends" / state.run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    weekly = load_weekly(dest, cfg)
    write_json(dest_dir / "manifest.json", {
        "source": "trends", "filename": src.name, "kind": kind_of(dest),
        "original_path": str(src), "fetched_at": utc_now(),
        "sha256": sha256_of(dest), "bytes": dest.stat().st_size, "rows": len(weekly),
    })
    state.raw["trends"] = cfg.rel(dest_dir)
    state.save(cfg)
    return dest


def _ingest_weather(cfg: Config, state: RunState, trends_file: Path, weather_in: Path | None,
                    fetch=weather_fetch.fetch_chunk, sleep_s: float | None = None) -> None:
    wcfg = cfg["weather"]
    weekly = load_weekly(trends_file, cfg)
    window = weather_window(weekly, wcfg["buffer_days"], wcfg["end_lag_days"])
    dest_dir = cfg.raw_dir / "weather" / state.run_id
    dest = dest_dir / "weather_daily.csv"

    if weather_in is not None:
        if not weather_in.exists():
            raise StageError(f"ไม่พบไฟล์อากาศ: {weather_in}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weather_in, dest)
        origin = f"file:{weather_in}"
    else:
        provinces = load_provinces(trends_file, cfg)
        df = weather_fetch.fetch_all(cfg, provinces, window, fetch=fetch, sleep_s=sleep_s)
        dest_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, encoding="utf-8")
        origin = f"api:openmeteo/{wcfg['model']}"

    rows = int(sum(1 for _ in open(dest, encoding="utf-8")) - 1)
    write_json(dest_dir / "manifest.json", {
        "source": "weather", "provider": wcfg["provider"], "model": wcfg["model"],
        "timezone": wcfg["timezone"], "origin": origin, "file": dest.name,
        "window": window, "fetched_at": utc_now(),
        "sha256": sha256_of(dest), "bytes": dest.stat().st_size, "rows": rows,
    })
    state.raw["weather"] = cfg.rel(dest_dir)
    state.save(cfg)


def run(cfg: Config, state: RunState, args) -> str:
    if "trends" in state.raw and (cfg.abs(state.raw["trends"]) / "manifest.json").exists():
        manifest = read_json(cfg.abs(state.raw["trends"]) / "manifest.json")
        trends_file = cfg.abs(state.raw["trends"]) / manifest["filename"]
        msg_t = "TRENDS: ข้าม (เก็บไว้แล้วในรอบนี้)"
    else:
        src = _resolve_input(getattr(args, "trends_input", None), cfg["trends"]["input"], cfg)
        if src is None:
            raise StageError("ยังไม่ระบุไฟล์ TRENDS")
        trends_file = _ingest_trends(cfg, state, src)
        msg_t = f"TRENDS: เก็บ {trends_file.name}"

    if "weather" in state.raw and (cfg.abs(state.raw["weather"]) / "manifest.json").exists():
        return f"{msg_t}; อากาศ: ข้าม (เก็บไว้แล้วในรอบนี้)"
    weather_in = _resolve_input(getattr(args, "weather_input", None), cfg["weather"]["input"], cfg)
    _ingest_weather(cfg, state, trends_file, weather_in,
                    fetch=getattr(args, "fetch", weather_fetch.fetch_chunk),
                    sleep_s=getattr(args, "sleep_s", None))
    return f"{msg_t}; อากาศ: เก็บแล้ว"
