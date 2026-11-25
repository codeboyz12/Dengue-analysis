# Dengue × Weather Forecasting Pipeline

โปรเจกต์กลุ่มวิชา **CSS 342 Data Science and Data Engineering** (KMUTT 2569)
ตรวจสอบว่าข้อมูลสภาพอากาศ (ฝน, อุณหภูมิ) ช่วยพยากรณ์การเติบโตของไข้เลือดออกรายจังหวัดล่วงหน้า **4 สัปดาห์** ได้หรือไม่

---

## โครงสร้างทีม

| บทบาท | งาน |
|---|---|
| **Nut** (repo นี้) | Data pipeline: ingest → validate → transform → load → export |
| **DS Team (4 คน)** | Feature engineering, โมเดล, การทดสอบสมมติฐาน H1–H4 |

---

## ข้อมูลที่ใช้

| แหล่ง | รายละเอียด | ไฟล์ |
|---|---|---|
| **TRENDS** (กรมควบคุมโรค) | ผู้ป่วยไข้เลือดออก/ชิคุนกุนยา/HFMD รายสัปดาห์ 77 จังหวัด ปี 2016–2025 (2 ชีท) | `input/TRENDS.xlsx` |
| **ERA5** (Open-Meteo) | อากาศรายวัน: ฝน, tmax, tmin ที่จุด centroid จังหวัด 283,822 แถว | `input/weather_daily.csv` |

**หน่วยวิเคราะห์:** จังหวัด-สัปดาห์เต็ม 7 วัน = 40,579 แถว (77 จังหวัด × 527 สัปดาห์)

---

## สมมติฐาน

| ID | คำถาม | เกณฑ์ผ่าน |
|---|---|---|
| **H1** | ฝนมีความสัมพันธ์กับ D หรือไม่ | \|ρ\| ≥ 0.10 ใน ≥ 4/5 fold |
| **H2** | อุณหภูมิมีความสัมพันธ์กับ D หรือไม่ | \|ρ\| ≥ 0.10 ใน ≥ 4/5 fold |
| **H3** | สัญญาณอยู่ที่ lag กี่สัปดาห์ | contingent: ผ่านต่อเมื่อ H1 หรือ H2 ผ่านก่อน |
| **H4** | M1 (M0 + อากาศ) ลด MAE ≥ 5% เทียบ M0 (ประวัติ + ฤดูกาล) | ≥ 4/5 fold, CI ไม่คร่อมศูนย์ |

**D (target):** `log1p(R_next) − log1p(R_prev)` เมื่อ R_prev = อัตราป่วย t−3..t, R_next = t+1..t+4

---

## ผลสรุป

| สมมติฐาน | ผล | หมายเหตุ |
|---|---|---|
| H1 — ฝน | ❌ ไม่ผ่าน | \|ρ\| สูงสุด 0.075 (เกณฑ์ 0.10) |
| H2 — อุณหภูมิ | ❌ ไม่ผ่าน | 2019 ผ่านเกณฑ์แต่ 4 fold ที่เหลือต่ำกว่า — ไม่ generalizable |
| H3 — lag profile | — ตัดสินไม่ได้ | H1/H2 ไม่ผ่านทั้งคู่ → ไม่มีสัญญาณให้ระบุ lag |
| H4 — MAE | ❌ ไม่ผ่าน | Huber M1 ดีกว่าใน 1/5 fold (เกณฑ์ ≥ 4/5) |

> **ข้อสรุป:** อากาศไม่เพิ่มประโยชน์ในการพยากรณ์ไข้เลือดออกในข้อมูลชุดนี้ ซึ่งเป็น outcome ที่ proposal ระบุว่ายอมรับได้ โมเดลที่ดีที่สุดคือ ARIMA(1,1,1) walk-forward ต่อจังหวัด

---

## ติดตั้งโปรเจกต์

### วิธีที่ 1 — Docker (แนะนำ)

```bash
# 1. clone repo
git clone <repo-url>
cd dengue_pipeline

# 2. วางไฟล์ข้อมูล
mkdir -p input data reports/static
cp /path/to/TRENDS.xlsx input/
cp /path/to/weather_daily.csv input/

# 3. build image
docker compose build

# 4. รัน pipeline ครั้งแรก
docker compose run --rm pipeline run \
  --trends-input input/TRENDS.xlsx \
  --weather-input input/weather_daily.csv

# 5. ตรวจสถานะ
docker compose run --rm pipeline status

# 6. export ตารางฐาน (44 คอลัมน์)
docker compose run --rm pipeline export
```

> **Linux:** ต้อง export UID/GID ก่อน เพื่อให้ไฟล์ใน `data/` เป็นของผู้ใช้ปัจจุบัน
> ```bash
> export UID=$(id -u) GID=$(id -g)
> docker compose run --rm pipeline run ...
> ```

### วิธีที่ 2 — Local (Python 3.12)

```bash
# สร้าง virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# ติดตั้ง dependencies
pip install -e .
pip install statsmodels          # สำหรับ ARIMA
pip install tabm torch           # สำหรับ TabM-mini (optional)

# รัน pipeline
python -m dengue_pipeline run \
  --trends-input input/TRENDS.xlsx \
  --weather-input input/weather_daily.csv
```

---

## รันเทส

```bash
# Docker
docker compose --profile tests run --rm tests

# Local
pytest -q
```

---

## Dashboard และ Endpoints

Dashboard ทั้งหมดเปิดผ่าน dev server ตัวเดียว:

```bash
source .venv/bin/activate
python scripts/serve.py 8000
```

| URL | หน้า |
|---|---|
| `http://localhost:8000/` | หน้าหลัก |
| `http://localhost:8000/executive_summary.html` | ผลการวิจัย H1–H4 ครบถ้วน |
| `http://localhost:8000/online_learning.html` | Online Learning Experiment |

### API Endpoints

| Path | คำอธิบาย |
|---|---|
| `GET /data/<filename>` | ให้บริการไฟล์ใน `data/` เช่น `online_learning.json` |
| `GET /api/run-online-learning` | SSE stream — รัน simulation และ stream stdout แบบ real-time |

---

## รัน Experiments (Local)

```bash
source .venv/bin/activate

# Online Learning: LR + ARIMA per province + CUSUM drift detection
# (หรือกด ▶ Run Script ในหน้า online_learning.html)
python scripts/run_online_learning.py
# → data/online_learning.json

# XGBoost sensitivity M0 vs M1 (5 folds)
python scripts/run_xgb.py
# → data/sensitivity_xgb.csv

# TabM-mini sensitivity M0 vs M1 (5 folds)
python scripts/run_tabm.py
# → data/sensitivity_tabm.csv

# ARIMA/Holt-Winters walk-forward per province
python scripts/run_timeseries_models.py
# → reports/static/data.json

# Export DuckDB → data.json สำหรับ dashboard
python scripts/export_static_data.py
# → reports/static/data.json
```

---

## Pipeline Architecture

```
input/TRENDS.xlsx
input/weather_daily.csv
        │
        ▼
   [ingest]            ตรวจ checksum, parse TRENDS 2 ชีท, merge population wide→long
        │
   [validate_source]   ตรวจ 7 กฎ (checksum, row count, date range ฯลฯ)
        │
   [transform]         merge trends + weather รายวัน → รายสัปดาห์, คำนวณ lag อากาศ 0–8 wk
        │
   [validate_merged]   ตรวจ 14 กฎ: row count, null rate, date continuity, leakage
        │
   [load]              เขียนลง DuckDB (schema core/mart/ops, append-only + data_version)
        │
   [export]            ส่งออก CSV 44 คอลัมน์ (contract_v2) + metadata.json
        │
        ▼
  data/exports/contract_v2/<run_id>/trends_weather_merged.csv
  data/pipeline.duckdb  (mart.trends_weather_latest, ops.audit_log, ops.data_version)
```

---

## Feature Engineering

ฟีเจอร์สร้างใน `docs/notebook_snippets/build_features_sql.py` และ `get_fold_sql.py`

| Prefix | กลุ่ม | จำนวน |
|---|---|---|
| `hist_` | ประวัติโรคจังหวัด: lograte lag 0–3, R_prev, R_prev2, growth | 7 |
| `nat_` | ระดับประเทศ: lograte lag0, logR_prev, growth | 3 |
| `seas_` | Fourier ฤดูกาล sin/cos k=1..3 | 6 |
| `reg_` | เขตสุขภาพ one-hot (13 เขต → 12 dummy) | 12 |
| `wx_` | อากาศดิบ lag 0–4: rain, tmax, tmin | 15 |
| `wxa_` | anomaly อากาศ (เบี่ยงเบนจาก climatology train-only ต่อ fold) | 15 |

**M0** = hist + nat + seas + reg (26 ตัวแปร) — baseline ไม่มีอากาศ  
**M1** = M0 + wx + wxa (56 ตัวแปร)

---

## Folds (Expanding Window Cross-Validation)

| Fold | Train | Test | Gap |
|---|---|---|---|
| F1 | 2016–2018 | 2019 | 4 สัปดาห์ |
| F2 | 2016–2021 | 2022 | 4 สัปดาห์ |
| F3 | 2016–2022 | 2023 | 4 สัปดาห์ |
| F4 | 2016–2023 | 2024 | 4 สัปดาห์ |
| F5 | 2016–2024 | 2025 | 4 สัปดาห์ |

seed = 342 · anomaly climatology คำนวณจาก train ของ fold เท่านั้น (ป้องกัน data leakage)

---

## Online Learning Experiment

จำลองการพยากรณ์แบบ real-time ด้วย:
- **Linear Regression** (M0, 77 จังหวัดพร้อมกัน)
- **ARIMA(1,1,1)** ต่อจังหวัด (walk-forward — append observation ทุกสัปดาห์)
- **CUSUM** ตรวจ drift ของ MAE รายสัปดาห์ → retrain เมื่อ MAE แย่ลงต่อเนื่อง
- **Baseline** (no-retrain) สำหรับเปรียบเทียบ

ผล: LR retrain ช่วยได้, ARIMA trigger บ่อยแต่โมเดลใหม่แทบไม่ชนะโมเดลเดิมที่เทรน 3 ปี

---

## โครงสร้างไฟล์

```
dengue_pipeline/
├── config/
│   ├── pipeline.toml           ค่า pipeline: sink, db_file, export contract
│   ├── contract_v2.toml        schema CSV export 44 คอลัมน์
│   └── analysis.toml           pre-analysis spec: M0/M1, folds, H1–H4 criteria
├── docs/
│   ├── notebook_snippets/      build_features_sql.py, get_fold_sql.py
│   ├── roadmap_h1_h4.md
│   └── pre_analysis_spec.md
├── notebooks/                  pipeline_main_duckdb.ipynb
├── reports/static/
│   ├── executive_summary.html  dashboard ผลวิจัย H1–H4
│   ├── online_learning.html    Online Learning experiment
│   └── index.html
├── scripts/
│   ├── serve.py                dev HTTP server + SSE endpoint
│   ├── run_online_learning.py  LR + ARIMA + CUSUM simulation
│   ├── run_xgb.py              XGBoost sensitivity
│   ├── run_tabm.py             TabM-mini sensitivity
│   ├── run_timeseries_models.py
│   └── export_static_data.py
├── src/dengue_pipeline/        pipeline package
│   ├── cli.py                  run | status | export | train | analyze
│   ├── stages/                 ingest → validate_source → transform → validate_merged → load → export
│   ├── validation/             checks_source / checks_merged / checks_export
│   └── sink_duckdb.py
├── tests/                      pytest (93 ผ่าน / 3 ข้าม)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## คำสั่งอื่น ๆ

```bash
# ดูข้อมูล TRENDS โดยไม่รัน pipeline
python -m dengue_pipeline inspect-trends input/TRENDS.xlsx

# ตรวจสถานะ run ล่าสุด (exit 1 ถ้ายังไม่มี run)
python -m dengue_pipeline status

# migrate FileSink เก่าเข้า DuckDB (idempotent)
python -m dengue_pipeline migrate-sink

# DS subcommands (ต้องรัน pipeline ก่อน)
python -m dengue_pipeline train --analysis-config config/analysis.toml
python -m dengue_pipeline analyze --analysis-config config/analysis.toml
```

---

*CSS 342 · KMUTT 2569 · seed = 342*
