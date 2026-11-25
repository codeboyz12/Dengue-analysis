"""คำสั่ง:  python -m dengue_pipeline run|status|export [ตัวเลือก]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dengue_pipeline.config import Config
from dengue_pipeline.pipeline import RunArgs, run_pipeline
from dengue_pipeline.runstate import STAGES, RunState


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dengue-pipeline")
    ap.add_argument("--config", default=None, help="ไฟล์ config (ค่าตั้งต้น config/pipeline.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="รันทั้งสายหรือเฉพาะขั้น")
    r.add_argument("--stage", default="all", choices=STAGES + ["all"])
    r.add_argument("--run-id", default=None, help="ต่อจากรอบที่มีอยู่ (ไม่ระบุ = รอบใหม่/รอบล่าสุด)")
    r.add_argument("--trends-input", default=None, help="TRENDS.xlsx หรือ csv ที่รวมแล้ว")
    r.add_argument("--weather-input", default=None,
                   help="weather_daily.csv ที่มีอยู่แล้ว (ไม่ระบุ = ดึงจาก Open-Meteo)")

    s = sub.add_parser("status", help="ดูสถานะของรอบ")
    s.add_argument("--run-id", default=None)

    sub.add_parser("migrate-sink", help="ย้ายเวอร์ชันที่อยู่ใน data/validated/ เข้าฐานข้อมูล DuckDB (ทำซ้ำได้ ข้ามเวอร์ชันที่มีแล้ว)")
    e = sub.add_parser("export", help="ส่งออก CSV สำหรับเทรนจากข้อมูลล่าสุดในชั้น validated")
    e.add_argument("--version", default=None, help="run_id ของชุดข้อมูลที่ validated (ไม่ระบุ = ล่าสุด)")
    e.add_argument("--contract", default=None, help="ไฟล์ contract (ค่าตั้งต้นตาม [export].contract)")

    i = sub.add_parser("inspect-trends", help="ดูโครงสร้างจริงของ TRENDS.xlsx (ชีท คอลัมน์ ชนิด) ก่อนรัน")
    i.add_argument("path", help="TRENDS.xlsx")

    t = sub.add_parser("train", help="เทรนและวัดผล M0/M1 บนทุก fold บันทึกลง mart.results")
    t.add_argument("--analysis-config", default="config/analysis.toml",
                   help="ไฟล์ analysis config (ค่าตั้งต้น config/analysis.toml)")
    t.add_argument("--run-id", default=None, help="data_version ที่จะใช้ (ไม่ระบุ = ล่าสุด)")

    an = sub.add_parser("analyze", help="ตรวจสมมติฐาน H1/H2/H3 partial Spearman + placebo บันทึกลง mart.h_results")
    an.add_argument("--analysis-config", default="config/analysis.toml",
                    help="ไฟล์ analysis config (ค่าตั้งต้น config/analysis.toml)")

    return ap


def inspect_trends(cfg: Config, path: str) -> int:
    """พิมพ์ชีท/คอลัมน์/ชนิดข้อมูล และบอกว่า province data เป็นรูปแบบที่ขั้นรวมสองชีทรองรับหรือไม่"""
    import pandas as pd
    xl = pd.ExcelFile(path)
    sheets = {n: xl.parse(n) for n in xl.sheet_names}
    for name, df in sheets.items():
        print(f"\n[{name}] {len(df):,} แถว x {df.shape[1]} คอลัมน์")
        for c in df.columns:
            print(f"  {str(c):<40} {df[c].dtype}  ตัวอย่าง: {df[c].iloc[0] if len(df) else '-'}")
    prov = sheets.get(cfg["trends"]["province_sheet"])
    if prov is not None and "P-code" in prov.columns:
        from dengue_pipeline.stages.transform import province_long
        long = province_long(prov)
        keys = ["P-code", "Year"] if "Year" in long.columns else ["P-code"]
        dup = int(long.duplicated(keys).sum())
        has_pop = "Population" in long.columns
        reshaped = long is not prov
        print(f"\nprovince data: {'แปลงคอลัมน์ Population YYYY เป็นแถวต่อจังหวัด-ปี, ' if reshaped else ''}"
              f"คีย์รวม {keys}, ซ้ำ {dup} แถว, มี Population: {has_pop} -> "
              + ("รองรับ" if dup == 0 and has_pop else "ไม่รองรับ (ดูข้อความใน merge_trends_sheets)"))
    return 0


def migrate_sink(cfg: Config) -> int:
    """ย้ายชุดข้อมูลที่ผ่านตรวจจากโฟลเดอร์ validated/ (FileSink) เข้า DuckDB โดยเก็บ metadata เดิมทั้งก้อน"""
    from dengue_pipeline.sink_duckdb import DuckDBSink
    from dengue_pipeline.sinks import FileSink
    src, dst = FileSink(cfg), DuckDBSink(cfg)
    have = set(dst.versions())
    moved = []
    for v in src.versions():
        if v in have:
            continue
        meta = src.read_metadata(v)
        tables = {name: src.read(v, name) for name in meta["rows"]}
        dst.write(v, tables, meta)
        moved.append(v)
    print(f"✔ ย้าย {len(moved)} เวอร์ชัน {moved} → {dst.path}" if moved else "ไม่มีเวอร์ชันใหม่ให้ย้าย")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = Config.load(a.config)

    if a.cmd == "run":
        return run_pipeline(cfg, a.stage, a.run_id,
                            RunArgs(trends_input=a.trends_input, weather_input=a.weather_input))

    if a.cmd == "status":
        state = RunState.load(cfg, a.run_id) if a.run_id else RunState.latest(cfg)
        if state is None:
            print("ยังไม่มีรอบที่รันไว้")
            return 1
        print(f"run_id = {state.run_id}")
        for st in STAGES:
            info = state.stages.get(st)
            print(f"  {st:<16} " + (f"{info['status']:<11} {info['message']}" if info else "ยังไม่ได้รัน"))
        if state.validated:
            print(f"  validated → {state.validated}")
        return 0

    if a.cmd == "migrate-sink":
        return migrate_sink(cfg)

    if a.cmd == "export":
        from dengue_pipeline.errors import StageError, ValidationFailed
        from dengue_pipeline.stages.export import export_csv
        try:
            meta = export_csv(cfg, a.version, a.contract)
        except (ValidationFailed, StageError) as ex:
            print(f"✘ export ไม่สำเร็จ: {ex}")
            return 1
        print(f"✔ export {meta['rows']:,} แถว x {len(meta['columns'])} คอลัมน์ → {meta['path']}")
        return 0

    if a.cmd == "inspect-trends":
        return inspect_trends(cfg, a.path)

    if a.cmd == "train":
        import tomllib
        analysis_path = (cfg.root / a.analysis_config
                         if not Path(a.analysis_config).is_absolute()
                         else Path(a.analysis_config))
        with open(analysis_path, "rb") as f:
            cfg_analysis = tomllib.load(f)
        from dengue_pipeline.stages.train import run as train_run
        msg = train_run(cfg, cfg_analysis, getattr(a, "run_id", None))
        print(f"✔ {msg}")
        return 0

    if a.cmd == "analyze":
        import tomllib
        analysis_path = (cfg.root / a.analysis_config
                         if not Path(a.analysis_config).is_absolute()
                         else Path(a.analysis_config))
        with open(analysis_path, "rb") as f:
            cfg_analysis = tomllib.load(f)
        from dengue_pipeline.stages.analyze import run as analyze_run
        msg = analyze_run(cfg, cfg_analysis)
        print(f"✔ {msg}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
