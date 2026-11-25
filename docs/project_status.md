# สรุปสถานะโปรเจกต์ — ไข้เลือดออก × สภาพอากาศ (CSS 342, KMUTT 2569)

วันที่อัปเดต: 2026-11-20 (อัปเดตครั้งสุดท้าย: 2026-11-20 เย็น)

---

## 1. ภาพรวมและคำถามวิจัย

โปรเจกต์นี้ศึกษาว่า **สภาพอากาศ (ฝน, อุณหภูมิสูงสุด, อุณหภูมิต่ำสุด) ช่วยพยากรณ์การเติบโตของไข้เลือดออกรายจังหวัดล่วงหน้า 4 สัปดาห์ได้หรือไม่** โดยใช้ข้อมูลรายสัปดาห์ระดับจังหวัด 77 จังหวัด ครอบคลุมปี 2016-2025

ผลที่ยอมรับได้ตาม proposal รวมถึง "อากาศไม่เพิ่มประโยชน์ในข้อมูลนี้" ซึ่งยังเป็นข้อค้นพบที่มีคุณค่าเชิงวิทยาศาสตร์

---

## 2. ชุดข้อมูลและหน่วยวิเคราะห์

| รายการ | รายละเอียด |
|---|---|
| ข้อมูลโรค | TRENDS.xlsx 2 ชีท (weekly + province) ปี 2016-2025 |
| ข้อมูลอากาศ | ERA5 ผ่าน Open-Meteo ที่จุด centroid จังหวัด 283,822 แถว (77 จังหวัด × 3,686 วัน) |
| หน่วยวิเคราะห์ | จังหวัด-สัปดาห์เต็ม 7 วัน รวม 40,579 แถว (77 จังหวัด × 527 สัปดาห์) |
| Target D | `log1p(R_next) − log1p(R_prev)` โดย R = ผู้ป่วยต่อ 1 แสนประชากร, R_prev = t−3..t, R_next = t+1..t+4 |

---

## 3. สมมติฐานที่ทดสอบ (H1–H4)

| ID | สมมติฐาน | เกณฑ์ผ่าน |
|---|---|---|
| **H1** | ฝนมีสัญญาณต่อ D หลังหักฤดูกาลและประวัติ | median \|ρ\| ≥ 0.10 เครื่องหมายเดียวกัน ≥ 4/5 fold ใน ≥ 2 จาก 5 lag |
| **H2** | อุณหภูมิมีสัญญาณต่อ D ที่เหลืออยู่ | เหมือน H1 กับ tmax/tmin |
| **H3** | สัญญาณอากาศอยู่ที่ lag ที่ใช้พยากรณ์ได้จริง | มี lag ผ่านเกณฑ์ H1/H2 และ lag ที่ดีที่สุดไม่อยู่ที่ขอบ |
| **H4** | M1 ลด MAE ≥ 5% เทียบ M0, ดีกว่า ≥ 4/5 fold, CI ไม่คร่อมศูนย์ | ดูผลปัจจุบันในหัวข้อ 7 |

---

## 4. โครงสร้าง Pipeline

```
input/TRENDS.xlsx + input/weather_daily.csv
         ↓
python -m dengue_pipeline run
   ingest → validate_source → transform → validate_merged → load → export
         ↓
data/pipeline.duckdb (mart.trends_weather, core.weather_daily, ops.*)
         ↓
python -m dengue_pipeline train
   build_features → get_fold (anomaly per fold) → M0/M1/sensitivity → bootstrap CI
         ↓
mart.results (42 แถว: 7 folds × 3 models)
```

---

## 5. การตัดสินใจที่ล็อกแล้ว (Pre-Analysis Specification)

ล็อกไว้ใน `config/analysis.toml` และ `docs/pre_analysis_spec.md` ก่อนดูผลบน test set

| รายการ | ค่าที่ล็อก |
|---|---|
| โมเดลหลัก H4 | HuberRegressor (epsilon=1.35, alpha=1.0) |
| M0 features | hist (lag 0-3, rolling 4w) + nat + Fourier K=3 + reg |
| M1 features | M0 + wx_lag 0-4 + anomaly (climatology จาก train เท่านั้น) |
| Folds หลัก | test 2019, 2022, 2023, 2024, 2025 (expanding train, gap 4 สัปดาห์) |
| Folds เสริม | 2020, 2021 (แยกรายงาน) |
| Bootstrap | block-week, 1,000 replicates (debug ใช้ 100), seed=342 |
| จังหวัดข้างเคียง | ไม่รวมใน M0 (sensitivity analysis เท่านั้น) |

---

## 6. สิ่งที่ทำเสร็จแล้ว

### 6.1 Data Pipeline (ครบ)
- **ingest → export** ครบสายตั้งแต่ TRENDS.xlsx + weather_daily.csv จนถึง `trends_weather_merged.csv` 44 คอลัมน์
- **DuckDB sink** พร้อม schema `core/mart/ops`, append-only, transaction, view `*_latest`
- **Validation** ครอบคลุม 23+ กฎ (source, merged, export) รวม leakage test LKG-01 และ LKG-ANOM
- **Docker** python:3.12-slim multi-stage, non-root uid 1000
- **CLI** `run | status | export | inspect-trends | migrate-sink | train`

### 6.2 Feature Engineering
- `features.py` — `build_features()` SQL-based, อ่าน constants จาก `analysis.toml`
  - weather lag 0-8 (lag 5-8 สร้างจาก `wk_*` ใน SQL window function)
  - Fourier seasonality K=3
  - national-level features, regional dummies
- `folds.py` — `get_fold()` anomaly per fold (climatology จาก train เท่านั้น, ผ่าน leakage test)

### 6.3 Leakage Tests (ใหม่)
- **LKG-01** ขยายครอบคลุม lag 1-8 (เดิม lag 1-4)
- **LKG-ANOM** ใหม่: เปลี่ยนอากาศปีทดสอบ → climatology และ anomaly ของ train ต้องไม่เปลี่ยน
- รัน pytest 99 passed / 3 skipped

### 6.4 Modelling Pipeline
- `modelling.py` — `run_experiment()` loop folds × M0/M1/sensitivity
  - StandardScaler (fit บน train transform ทั้ง train+test)
  - Bootstrap CI block-week ด้วย `joblib.Parallel(prefer='threads')` (thread-safe กับ PyTorch)
  - Sensitivity: ElasticNetCV (inner CV), HistGradientBoosting, **XGBRegressor (ใหม่)**, **TabMMini (ใหม่)**
- `stages/train.py` — CLI wrapper + เขียน `mart.results` (append-only)
- **รันสำเร็จ** บนข้อมูลจริง 7 folds ใน ~6 นาที (100 replicates; TabMMini เพิ่มเวลาต่อ fold)

### 6.4.1 XGBRegressor
- hyperparameters จากเพื่อนทีม DS: `objective='reg:absoluteerror'`, `n_estimators=500`, `lr=0.03`, `max_depth=4`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5.0`
- ต้องการ `brew install libomp` บน macOS (OpenMP runtime สำหรับ parallelization)

### 6.4.2 TabMMini
- k=8 ensemble MLP, 2 layers width=64, dropout=0.1, MAE loss, AdamW lr=0.002
- 18 epochs × 3 seeds averaged per fold
- ใช้ M0/M1 feature set เดียวกันกับโมเดลอื่น (Decision B)
- ต้องการ `torch>=2.0` (`pip install torch`)

### 6.6 Hypothesis Testing — H1/H2/H3 (ใหม่)
- `hypotheses.py` — `compute_partial_spearman()`, `run_placebo()`, `run_h_analysis()`
- `stages/analyze.py` — CLI wrapper เขียน `mart.h_results`, `mart.h_placebo`, `mart.h_decisions`
- **CLI** `python -m dengue_pipeline analyze`
- **รันสำเร็จ** บนข้อมูลจริง 7 folds H1/H2 + 5 folds placebo ใน ~12 นาที

### 6.7 Dashboard (ใหม่)
- `dashboard/app.py` — Streamlit 3 แท็บ: Pipeline Health / EDA / H1-H4 Tracker
- **H1-H4 Tracker** มีส่วน "เปรียบเทียบ MAE ทุกโมเดล" — pivot table + line chart (toggle M0/M1)
- อ่าน DuckDB `read_only=True` โดยตรง
- รันด้วย `streamlit run dashboard/app.py`
- **หมายเหตุ:** ต้องปิด dashboard ก่อนรัน `train` (DuckDB write lock conflict)

### 6.8 Static Dashboard + Section 7 (ใหม่ 2026-11-20)
- `reports/static/index.html` + `reports/static/data.json` — Single-file HTML dashboard (Chart.js 4.4.3)
- เปิดด้วย `python3 -m http.server 8080` จาก `reports/static/` → `localhost:8080`
- `scripts/export_static_data.py` — export DuckDB results → data.json
- `scripts/run_timeseries_models.py` — ARIMA walk-forward, ARIMAX+rain(oracle), SARIMA open-loop, Seasonal Naive
- **Section ที่มี:** Header · Summary cards H1–H4 · EDA · H1/H2 ρ charts · H3 card · H4 primary (CI bars) · H4 sensitivity (รวม ARIMA/ARIMAX) · Time-Series MAE comparison · **Section 7** (Statistical Analysis / ML Model / Time-Series Analysis) · Data Overview
- **Section 7** ครอบคลุม: สมมติฐาน null/alternative, Partial Spearman + Bootstrap อธิบายเป็นภาษาไทย, algorithm table, train/test split, feature description ไม่ใช้ชื่อตัวแปร

### 6.9 Time-Series Models (ใหม่ 2026-11-20)
รันด้วย `python scripts/run_timeseries_models.py` — walk-forward per province, results ใน data.json

| โมเดล | วิธี | 2019 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Naive(D=0) | ไม่เปลี่ยน | 0.3095 | 0.1890 | 0.3054 | 0.2910 | 0.2295 |
| ARIMA(1,1,1) | walk-forward | 0.2684 | 0.1773 | 0.2598 | 0.2443 | 0.2056 |
| ARIMAX+rain(oracle) | walk-forward + ฝนจริงอนาคต | 0.2667 | 0.1796 | 0.2595 | 0.2419 | 0.2082 |
| SARIMA open-loop | forecast จาก train end | 0.6448 | 0.5445 | 0.9558 | 0.7707 | 0.4371 |
| Huber Baseline (ML ref) | — | 0.3021 | 0.2983 | 0.2943 | 0.2914 | 0.2830 |

**ข้อค้นพบ:** ARIMA walk-forward ดีกว่า ML baseline ทุก fold; ARIMAX oracle ดีกว่า ARIMA เพียง 0.1–1% → ยืนยัน H1 (ฝน lag 0–4 ไม่มีผล)

### 6.5 Notebook Snippets (reference สำหรับทีม DS)
- `docs/notebook_snippets/build_features_sql.py` — เวอร์ชัน DuckDB SQL ที่ทีม DS วางในเซลล์ได้
- `docs/notebook_snippets/get_fold_sql.py` — fold + anomaly SQL
- ผลเทียบกับ pandas เดิมที่ tolerance 1e-9 บนข้อมูลจริง ทุก 7 folds

---

## 7. ผลการทดลองล่าสุด (2026-11-20, debug run: 100 replicates)

`config_hash = f51af9b2b3f7`, `data_version = 20261118T054719Z`

### HuberRegressor — M0 vs M1 (primary)

| fold | MAE M0 | MAE M1 | Δ MAE (%) | CI 95% | ผ่าน H4? |
|---|---|---|---|---|---|
| 2019 | 0.3021 | 0.3601 | **−19.2%** | −23.6 to −14.2 | ❌ M1 แย่กว่า |
| 2022 | 0.2983 | 0.2900 | +2.8% | +1.4 to +3.8 | ❌ ต่ำกว่า 5% |
| 2023 | 0.2943 | 0.2957 | −0.5% | −1.8 to +0.8 | ❌ CI คร่อมศูนย์ |
| 2024 | 0.2914 | 0.2751 | **+5.6%** | +3.4 to +7.4 | ✅ |
| 2025 | 0.2830 | 0.2795 | +1.2% | −0.3 to +2.7 | ❌ CI คร่อมศูนย์ |

**สรุป H4 (preliminary):** ไม่ผ่าน — M1 ดีกว่าเพียง 1/5 fold หลัก (ต้องการ 4/5) และ MAE ลด ≥ 5% เพียง 1 fold

### Sensitivity — ElasticNetCV (inner CV: l1_ratio=[0.1,0.5,0.9], alpha=[0.001,0.01,0.1], cv=5)

| fold | MAE M0 | MAE M1 | Δ MAE (%) |
|---|---|---|---|
| 2019 | 0.3027 | 0.3625 | **−19.9%** (M1 แย่กว่า) |
| 2022 | 0.2988 | 0.2935 | +1.8% |
| 2023 | 0.3055 | 0.2990 | +2.1% |
| 2024 | 0.2933 | 0.2796 | **+4.7%** |
| 2025 | 0.2828 | 0.2792 | +1.3% |

### Sensitivity — HistGradientBoosting

| fold | MAE M0 | MAE M1 | Δ MAE (%) |
|---|---|---|---|
| 2019 | 0.3426 | 0.3402 | +0.7% |
| 2022 | 0.2973 | 0.2970 | +0.1% |
| 2023 | 0.3100 | 0.3050 | +1.6% |
| 2024 | 0.2854 | 0.2657 | **+6.9%** |
| 2025 | 0.2719 | 0.2608 | **+4.1%** |

**ข้อสังเกต sensitivity:** ทั้ง ElasticNetCV และ HistGradientBoosting สอดคล้องกับ Huber — 2024 เด่นสุดในทุกโมเดล, 2019 M1 แย่กว่าทุกโมเดล, fold อื่นต่ำกว่า 5% ทั้งหมด ผลนี้เพิ่มความน่าเชื่อถือให้ข้อสรุป H4

---

## 7b. ผลการทดลอง H1/H2/H3 (2026-11-20)

`mart.h_results` 105 แถว, `mart.h_placebo` 75 แถว

### H1 — ฝน (rain) partial Spearman ρ vs D

| lag | median |ρ| (avg 5 folds) | median ρ | ผ่านเกณฑ์ 0.10? |
|-----|----------------------|----------|----------------|
| 0 | 0.053 | +0.032 | ❌ |
| 1 | 0.067 | +0.060 | ❌ |
| 2 | **0.075** | +0.074 | ❌ |
| 3 | 0.067 | +0.062 | ❌ |
| 4 | 0.057 | +0.049 | ❌ |

**H1 ไม่ผ่าน** — |ρ| สูงสุด 0.075 ต่ำกว่าเกณฑ์ 0.10 ทุก lag

### H2 — อุณหภูมิ (tmax/tmin) partial Spearman ρ vs D

| var | lag | median |ρ| | median ρ | ผ่านเกณฑ์ 0.10? |
|-----|-----|---------|----------|----------------|
| tmax | 1 | **0.105** | −0.098 | ❌ same-sign ไม่ผ่าน |
| tmin | 1 | **0.104** | −0.097 | ❌ same-sign ไม่ผ่าน |
| tmin | 2 | **0.110** | −0.103 | ❌ same-sign ไม่ผ่าน |

tmax/tmin มี |ρ| เกิน 0.10 ที่ lag 1-2 แต่ **เครื่องหมายไม่สม่ำเสมอ** ข้าม 5 fold (ไม่ผ่าน same-sign criterion ≥ 4/5 fold) → **H2 ไม่ผ่าน**

### H3 — ตัดสินไม่ได้ (H1 และ H2 ไม่ผ่านทั้งคู่ ตามนิยามที่ lock ไว้)

### Placebo Test

| var | lag | ρ_real | placebo P95 | real < P95? |
|-----|-----|--------|-------------|-------------|
| rain | 2 | 0.074 | 0.123 | ✓ อยู่ใน null |
| tmax | 1 | −0.098 | 0.123 | ✓ อยู่ใน null |
| tmin | 2 | −0.103 | 0.122 | ✓ อยู่ใน null |

**ρ_real < placebo P95 ทุก lag ทุกตัวแปร** — สัญญาณที่เห็นอยู่ในช่วง noise ของ null distribution ยืนยัน H1/H2 ไม่ผ่าน

### สรุปผลสมมติฐานทั้งหมด

| สมมติฐาน | ผล | หมายเหตุ |
|----------|-----|---------|
| **H1** (ฝน) | ❌ ไม่ผ่าน | |ρ| สูงสุด 0.075 < 0.10 |
| **H2** (อุณหภูมิ) | ❌ ไม่ผ่าน | เครื่องหมายไม่สม่ำเสมอข้าม fold |
| **H3** (lag profile) | — ตัดสินไม่ได้ | H1/H2 ไม่ผ่าน |
| **H4** (MAE ลด ≥ 5%) | ❌ ไม่ผ่าน | ผ่านเพียง 1/5 fold |

**ข้อสรุปโดยรวม:** อากาศไม่เพิ่มประโยชน์ในการพยากรณ์ไข้เลือดออกในข้อมูลชุดนี้ ทั้งในแง่ correlation (H1/H2) และประสิทธิภาพโมเดล (H4) ผลนี้สอดคล้องกับ EDA เบื้องต้นและเป็น outcome ที่ proposal ระบุว่ายอมรับได้

---

## 8. ข้อสังเกตและประเด็นที่ต้องติดตาม

1. **F1 (2019) ผิดปกติ** — M1 แย่กว่า M0 ถึง 19-20% ทุกโมเดล training data บาง (10,241 แถว = 2016-2018 เท่านั้น) อาจ overfit บน weather features สอดคล้องกับที่ CLAUDE.md ระบุว่า F1 มี inner split บางเกินไป
2. **ผลเบื้องต้นสอดคล้องกับ EDA** — อากาศอย่างเดียวลด MAE ≈ 1-2% (EDA เห็น ≈ 1.4-2%) ผล 2024 เป็น outlier ที่น่าสนใจและสอดคล้องข้ามทุกโมเดล
3. **Bootstrap 100 replicates** ตัวเลข CI ยังไม่เสถียร สำหรับรายงานจริงต้องรัน 1,000 replicates (แก้ค่ากลับใน `config/analysis.toml`)
4. **ElasticNetCV** แก้แล้ว — เปลี่ยนจาก ElasticNet (regularization แรงเกินจน M0=M1) เป็น ElasticNetCV ด้วย inner CV เลือก hyperparameter อัตโนมัติ ผลตอนนี้สอดคล้องกับ Huber

---

## 9. สิ่งที่ต้องทำถัดไป

### ด่วน (ก่อน final run)
- [x] **แก้ ElasticNet** — เปลี่ยนเป็น ElasticNetCV ด้วย inner CV
- [x] **Static dashboard** พร้อมใช้งาน รวม Section 7 และ ARIMA/ARIMAX
- [x] **Bug fix ARIMAX baseline** — ใช้ ARIMA MAE เป็น mae_m0 แทน Huber M0
- [x] **OpenMP deadlock (TabMMini)** — แก้ด้วย `torch.set_num_threads(1)` ก่อน `torch.tensor()`
- [ ] **เปลี่ยน `n_replicates` กลับเป็น 1,000** ใน `config/analysis.toml` สำหรับ final run
- [ ] **อัปเดต `ref_D_and_hist`** ใน `config/analysis.toml` หลัง build_features ด้วย MAX_WX_LAG=8

### ขั้น C (ที่เหลือ)
- [ ] บันทึก config hash + seed + data_version ต่อ run ใน `ops` (ทำแล้วบางส่วนใน `mart.results`)
- [ ] ให้ notebook ของ DS อ่าน `config/analysis.toml` ตรงๆ ไม่ hardcode

### ขั้น D — Dashboard
- [x] `dashboard/app.py` — Streamlit 3 แท็บ พร้อมใช้งาน

### ขั้น E — H1/H2/H3
- [x] partial Spearman ระหว่าง anomaly อากาศ lag 0-4 กับ D ภายในจังหวัด
- [x] placebo test (สลับ wxa rows ภายในจังหวัด 200 replicates)
- [ ] ขยาย lag เป็น 0-8 สำหรับ H3 (ตัดสินไม่ได้เพราะ H1/H2 ไม่ผ่าน — ไม่จำเป็นต้องรัน)

### งานค้างเล็กๆ
- [x] pin `duckdb==2.0.0` ใน `pyproject.toml` (ทำแล้ว)
- [x] `pull_policy: build` ใน docker-compose service tests (ทำแล้ว)
- [ ] ตรวจ notebook kernel ให้ใช้ `.venv` (Python 3.12) ไม่ใช่ env เก่า

---

## 10. โครงสร้างไฟล์สำคัญ

```
config/
  pipeline.toml          ค่า pipeline (sink, lags, validation)
  analysis.toml          Pre-analysis spec ที่ lock แล้ว (M0/M1/folds/bootstrap)
  contract_v2.toml       สัญญา export 44 คอลัมน์

src/dengue_pipeline/
  cli.py                 run | status | export | inspect-trends | migrate-sink | train | analyze
  features.py            build_features() SQL-based
  folds.py               make_folds(), get_fold() anomaly per fold
  modelling.py           run_experiment() + TabMMiniRegressor + bootstrap CI (threads)
  hypotheses.py          H1/H2/H3 partial Spearman + placebo test
  stages/train.py        CLI wrapper → เขียน mart.results
  stages/analyze.py      CLI wrapper → เขียน mart.h_results, mart.h_placebo, mart.h_decisions

dashboard/
  app.py                 Streamlit dashboard (Pipeline Health / EDA / H1-H4 Tracker)

reports/static/
  index.html             Static executive dashboard (Chart.js, ไม่ต้องการ server พิเศษ)
  data.json              Snapshot ผลทั้งหมด (H1-H4, TS models, EDA)

scripts/
  export_static_data.py  DuckDB → data.json
  run_timeseries_models.py  ARIMA/ARIMAX/SARIMA walk-forward → data.json

docs/
  pre_analysis_spec.md   เอกสารนิยาม M0/M1/เกณฑ์ H1-H4 (ล็อกก่อน unblind)
  roadmap_h1_h4.md       roadmap สมมติฐาน H1-H4 พร้อมสถานะหลักฐาน
  notebook_snippets/     build_features_sql.py, get_fold_sql.py (reference สำหรับทีม DS)

data/pipeline.duckdb
  mart.trends_weather    ข้อมูลรายสัปดาห์ทุกเวอร์ชัน
  mart.results           ผลการทดลอง M0/M1 ทุก fold
  mart.h_results         partial Spearman ρ ทุก fold ทุก var ทุก lag
  mart.h_placebo         ผล placebo test บน main folds
  mart.h_decisions       ผลการตัดสิน H1/H2/H3 (JSON)
  ops.audit_log          log ผลการตรวจทุกกฎ
  ops.data_version       ทะเบียน run_id
```
