# Dengue × Weather Forecasting Pipeline

A data pipeline and analysis project for **CSS 342 Data Science and Data Engineering** (KMUTT, 2025).

Investigates whether weather variables (rainfall, temperature) help forecast dengue fever growth by province **4 weeks ahead** across Thailand.

---

## Data Sources

| Source | Description | File |
|---|---|---|
| **TRENDS** (Department of Disease Control) | Weekly dengue/chikungunya/HFMD cases, 77 provinces, 2016–2025 (2 sheets) | `input/TRENDS.xlsx` |
| **ERA5** (Open-Meteo) | Daily weather: rainfall, tmax, tmin at province centroids — 283,822 rows | `input/weather_daily.csv` |

**Unit of analysis:** province-week (full 7-day weeks) = 40,579 rows (77 provinces × 527 weeks)

---

## Hypotheses

| ID | Question | Pass Criterion |
|---|---|---|
| **H1** | Does rainfall correlate with D? | \|ρ\| ≥ 0.10 in ≥ 4/5 folds |
| **H2** | Does temperature correlate with D? | \|ρ\| ≥ 0.10 in ≥ 4/5 folds |
| **H3** | At which lag does the signal peak? | Contingent: requires H1 or H2 to pass first |
| **H4** | Does M1 (M0 + weather) reduce MAE ≥ 5% vs M0 (history + seasonality)? | ≥ 4/5 folds, CI does not cross zero |

**Target D:** `log1p(R_next) − log1p(R_prev)` where R_prev = incidence rate t−3..t, R_next = t+1..t+4

---

## Results

| Hypothesis | Result | Notes |
|---|---|---|
| H1 — Rainfall | ❌ Rejected | Max \|ρ\| = 0.075 (threshold 0.10) |
| H2 — Temperature | ❌ Rejected | 2019 passes but 4 remaining folds below threshold — not generalizable |
| H3 — Lag profile | — Inconclusive | H1/H2 both rejected → no signal to characterize |
| H4 — MAE | ❌ Rejected | Huber M1 wins in only 1/5 folds (threshold ≥ 4/5) |

> **Conclusion:** Weather does not improve dengue forecasting in this dataset — an outcome the original proposal considered acceptable. The best-performing model is ARIMA(1,1,1) walk-forward per province.

---

## Installation

### Option 1 — Docker (recommended)

```bash
# 1. Clone repo
git clone https://github.com/codeboyz12/Dengue-analysis.git
cd Dengue-analysis

# 2. Place data files
mkdir -p input data reports/static
cp /path/to/TRENDS.xlsx input/
# weather_daily.csv is optional — omit --weather-input to auto-fetch from Open-Meteo ERA5
# (requires internet; ~13 min for 77 provinces × 10 years; resumes on re-run if daily limit hit)

# 3. Build image
docker compose build

# 4a. Run pipeline — auto-fetch weather from Open-Meteo (recommended)
docker compose run --rm pipeline run \
  --trends-input input/TRENDS.xlsx

# 4b. Run pipeline — use existing weather_daily.csv (faster)
docker compose run --rm pipeline run \
  --trends-input input/TRENDS.xlsx \
  --weather-input input/weather_daily.csv

# 5. Check status
docker compose run --rm pipeline status

# 6. Export base table (44 columns)
docker compose run --rm pipeline export
```

> **Linux:** Export UID/GID first so files in `data/` are owned by the current user:
> ```bash
> export UID=$(id -u) GID=$(id -g)
> docker compose run --rm pipeline run ...
> ```

### Option 2 — Local (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
pip install statsmodels          # for ARIMA
pip install tabm torch           # for TabM-mini (optional)

# Auto-fetch weather from Open-Meteo ERA5 (omit --weather-input)
python -m dengue_pipeline run \
  --trends-input input/TRENDS.xlsx

# Or use an existing weather_daily.csv
python -m dengue_pipeline run \
  --trends-input input/TRENDS.xlsx \
  --weather-input input/weather_daily.csv
```

---

## Running Tests

```bash
# Docker
docker compose --profile tests run --rm tests

# Local
pytest -q
```

---

## Dashboard & Endpoints

All dashboards are served from a single dev server:

```bash
source .venv/bin/activate
python scripts/serve.py 8000
```

| URL | Page |
|---|---|
| `http://localhost:8000/` | Home |
| `http://localhost:8000/executive_summary.html` | Full H1–H4 research results |
| `http://localhost:8000/online_learning.html` | Online Learning Experiment |

### API Endpoints

| Path | Description |
|---|---|
| `GET /data/<filename>` | Serve files from `data/`, e.g. `online_learning.json` |
| `GET /api/run-online-learning` | SSE stream — runs simulation and streams stdout in real-time |

---

## Running Experiments (Local)

```bash
source .venv/bin/activate

# Online Learning: LR + ARIMA per province + CUSUM drift detection
# (or click ▶ Run Script on the online_learning.html page)
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

# Export DuckDB → data.json for dashboard
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
   [ingest]            Verify checksum, parse TRENDS 2 sheets, merge population wide→long
        │
   [validate_source]   Check 7 rules (checksum, row count, date range, etc.)
        │
   [transform]         Merge trends + daily weather → weekly, compute weather lags 0–8 wk
        │
   [validate_merged]   Check 14 rules: row count, null rate, date continuity, leakage
        │
   [load]              Write to DuckDB (schema core/mart/ops, append-only + data_version)
        │
   [export]            Export 44-column CSV (contract_v2) + metadata.json
        │
        ▼
  data/exports/contract_v2/<run_id>/trends_weather_merged.csv
  data/pipeline.duckdb  (mart.trends_weather_latest, ops.audit_log, ops.data_version)
```

---

## Feature Engineering

Features are built in `docs/notebook_snippets/build_features_sql.py` and `get_fold_sql.py`.

| Prefix | Group | Count |
|---|---|---|
| `hist_` | Province disease history: lograte lag 0–3, R_prev, R_prev2, growth | 7 |
| `nat_` | National level: lograte lag0, logR_prev, growth | 3 |
| `seas_` | Fourier seasonality sin/cos k=1..3 | 6 |
| `reg_` | Health region one-hot (13 regions → 12 dummies) | 12 |
| `wx_` | Raw weather lag 0–4: rain, tmax, tmin | 15 |
| `wxa_` | Weather anomaly (deviation from train-only climatology per fold) | 15 |

**M0** = hist + nat + seas + reg (26 features) — weather-free baseline  
**M1** = M0 + wx + wxa (56 features)

---

## Folds (Expanding Window Cross-Validation)

| Fold | Train | Test | Gap |
|---|---|---|---|
| F1 | 2016–2018 | 2019 | 4 weeks |
| F2 | 2016–2021 | 2022 | 4 weeks |
| F3 | 2016–2022 | 2023 | 4 weeks |
| F4 | 2016–2023 | 2024 | 4 weeks |
| F5 | 2016–2024 | 2025 | 4 weeks |

seed = 342 · anomaly climatology computed from each fold's train set only (prevents data leakage)

---

## Online Learning Experiment

Simulates real-time forecasting with:
- **Linear Regression** (M0, all 77 provinces simultaneously)
- **ARIMA(1,1,1)** per province (walk-forward — appends observation each week)
- **CUSUM** drift detection on weekly MAE → retrain when MAE degrades persistently
- **Baseline** (no-retrain) for comparison

**Result:** LR benefits from retraining; ARIMA triggers frequently but new models rarely beat the original trained on 3 years of data.

---

## File Structure

```
dengue_pipeline/
├── config/
│   ├── pipeline.toml           Pipeline settings: sink, db_file, export contract
│   ├── contract_v2.toml        CSV export schema: 44 columns, key P-code + Week start
│   └── analysis.toml           Pre-analysis spec: M0/M1, folds, H1–H4 criteria
├── docs/
│   ├── notebook_snippets/      build_features_sql.py, get_fold_sql.py
│   ├── roadmap_h1_h4.md
│   └── pre_analysis_spec.md
├── notebooks/                  pipeline_main_duckdb.ipynb
├── reports/static/
│   ├── executive_summary.html  H1–H4 research results dashboard
│   ├── online_learning.html    Online Learning experiment
│   └── index.html
├── scripts/
│   ├── serve.py                Dev HTTP server + SSE endpoint
│   ├── run_online_learning.py  LR + ARIMA + CUSUM simulation
│   ├── run_xgb.py              XGBoost sensitivity
│   ├── run_tabm.py             TabM-mini sensitivity
│   ├── run_timeseries_models.py
│   └── export_static_data.py
├── src/dengue_pipeline/        Pipeline package
│   ├── cli.py                  run | status | export | train | analyze
│   ├── stages/                 ingest → validate_source → transform → validate_merged → load → export
│   ├── validation/             checks_source / checks_merged / checks_export
│   └── sink_duckdb.py
├── tests/                      pytest (93 passed / 3 skipped)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Other Commands

```bash
# Inspect TRENDS file without running the full pipeline
python -m dengue_pipeline inspect-trends input/TRENDS.xlsx

# Check status of latest run (exits 1 if no run exists yet)
python -m dengue_pipeline status

# Migrate old FileSink data into DuckDB (idempotent)
python -m dengue_pipeline migrate-sink

# DS subcommands (requires pipeline run first)
python -m dengue_pipeline train --analysis-config config/analysis.toml
python -m dengue_pipeline analyze --analysis-config config/analysis.toml
```

---

*CSS 342 · KMUTT 2025 · seed = 342*
