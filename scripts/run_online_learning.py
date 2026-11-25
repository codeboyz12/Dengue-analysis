"""
Online Learning Simulation — drift detection + adaptive retraining
Models : LinearRegression (M0 features) + ARIMA(1,1,1) per province
Drift  : CUSUM on cross-province weekly MAE
Retrain: last 4 / 24 / 56 / 112 weeks (sliding window)

Reference: Page (1954) Biometrika; Farrington & Andrews (2003) Stat. in Medicine
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json, time, warnings
import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA as sARIMA

warnings.filterwarnings("ignore")
np.random.seed(342)

# ── config ────────────────────────────────────────────────────────────────────
SEED            = 342
TARGET          = "D"
MAX_WX_LAG      = 4
FOURIER_K       = 3
WX_BASE         = {"rain": "rain_wk_mm", "tmax": "tmax_mean_c", "tmin": "tmin_mean_c"}
USE_NAT         = True
TRAIN_YEARS     = (2016, 2018)
TEST_START_YEAR = 2019
RETRAIN_WINDOWS = [4, 24, 56, 112]   # weeks
WARMUP_WEEKS    = 12                  # establish CUSUM baseline
CUSUM_K_MULT    = 0.5                 # slack  k = K * σ₀
CUSUM_H_MULT    = 5.0                 # trigger when S_t > H * σ₀
EVAL_WINDOW     = 4                   # weeks used to eval candidate retrain models

DB_PATH  = ROOT / "data" / "pipeline.duckdb"
DB_TABLE = "mart.trends_weather_latest"
OUT_PATH = ROOT / "data" / "online_learning.json"

# ── load features ─────────────────────────────────────────────────────────────
import docs.notebook_snippets.build_features_sql as _bfm
_bfm.WX_BASE = WX_BASE; _bfm.MAX_WX_LAG = MAX_WX_LAG
_bfm.FOURIER_K = FOURIER_K; _bfm.TARGET = TARGET
from docs.notebook_snippets.build_features_sql import build_features_sql

print("Loading data...", flush=True)
con = duckdb.connect(str(DB_PATH), read_only=True)
raw = con.execute(f"SELECT * EXCLUDE(data_version) FROM {DB_TABLE}").df()
con.close()
df = build_features_sql(raw)
print(f"  df {df.shape}", flush=True)

HIST = ["hist_lograte_lag0","hist_lograte_lag1","hist_lograte_lag2","hist_lograte_lag3",
        "hist_logR_prev","hist_logR_prev2","hist_growth_prev"]
NAT  = ["nat_lograte_lag0","nat_logR_prev","nat_growth_prev"]
SEAS = [f"seas_sin{k}" for k in range(1, FOURIER_K+1)] + \
       [f"seas_cos{k}" for k in range(1, FOURIER_K+1)]
REG  = [c for c in df.columns if c.startswith("reg_")]
M0   = HIST + (NAT if USE_NAT else []) + SEAS + REG

usable = df[M0 + [TARGET]].notna().all(axis=1)
df = df[usable].sort_values(["Week start", "P-code"]).reset_index(drop=True)
print(f"  usable {df.shape}", flush=True)

test_start  = pd.Timestamp(f"{TEST_START_YEAR}-01-01")
df_train    = df[df["Year"].between(*TRAIN_YEARS) & (df["label_end"] < test_start)].copy()
df_test     = df[df["Year"] >= TEST_START_YEAR].copy()
provinces   = sorted(df["P-code"].unique())
test_weeks  = sorted(df_test["Week start"].unique())
print(f"  train={len(df_train):,}  test={len(df_test):,}  "
      f"{len(provinces)} provinces  {len(test_weeks)} test-weeks", flush=True)

# ── ARIMA helpers ──────────────────────────────────────────────────────────────
def fit_arima(series):
    """Fit ARIMA(1,1,1). Returns result or None if too little data / fails."""
    if len(series) < 10:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sARIMA(series, order=(1, 1, 1)).fit(disp=False)
    except Exception:
        return None

def arima_forecast1(result, fallback):
    """One-step-ahead forecast; fall back to scalar if model fails."""
    if result is None:
        return float(fallback)
    try:
        return float(result.forecast(1).iloc[0])
    except Exception:
        return float(fallback)

def arima_append(result, new_obs):
    """Extend ARIMA model with one new observation (parameters frozen)."""
    if result is None:
        return None
    try:
        return result.append([new_obs], refit=False)
    except Exception:
        return None

# ── initial training ──────────────────────────────────────────────────────────
print("\nTraining initial Linear Regression...", flush=True)
lr_model = LinearRegression().fit(df_train[M0], df_train[TARGET])
lr_model_base = LinearRegression().fit(df_train[M0], df_train[TARGET])  # frozen baseline
print("  done", flush=True)

print("Training initial ARIMA per province...", flush=True)
t0 = time.time()
prov_state = {}
prov_state_base = {}  # frozen baseline — walk-forward only, no retrain
for p in provinces:
    s = df_train[df_train["P-code"] == p].sort_values("Week start")[TARGET].values
    prov_state[p]      = {"hist": list(s), "result": fit_arima(s)}
    prov_state_base[p] = {"hist": list(s), "result": fit_arima(s)}
print(f"  done [{time.time()-t0:.1f}s]", flush=True)

# ── CUSUM detector ─────────────────────────────────────────────────────────────
class CUSUM:
    """One-sided upper CUSUM for detecting increase in MAE."""
    def __init__(self, name):
        self.name   = name
        self._wup   = []     # warmup MAE values
        self.S      = 0.0
        self.mu0    = None
        self.k      = None
        self.h      = None

    @property
    def ready(self):
        return self.mu0 is not None

    def step(self, mae):
        """Update CUSUM. Returns (triggered: bool, S_t: float)."""
        if not self.ready:
            self._wup.append(mae)
            if len(self._wup) >= WARMUP_WEEKS:
                a         = np.array(self._wup)
                self.mu0  = float(a.mean())
                sig       = max(float(a.std()), 1e-6)
                self.k    = CUSUM_K_MULT * sig
                self.h    = CUSUM_H_MULT * sig
                self.S    = 0.0
                print(f"  [{self.name}] CUSUM active  "
                      f"µ₀={self.mu0:.4f}  k={self.k:.4f}  h={self.h:.4f}", flush=True)
            return False, 0.0
        self.S = max(0.0, self.S + (mae - self.mu0 - self.k))
        return self.S > self.h, round(self.S, 5)

    def reset(self):
        self.S = 0.0

cusum_lr = CUSUM("LinReg")
cusum_ar = CUSUM("ARIMA")

# ── retrain helpers ────────────────────────────────────────────────────────────
def eval_lr_future(model, wi):
    """Eval model on EVAL_WINDOW weeks AFTER trigger (wi..wi+EVAL_WINDOW)."""
    eval_wks = set(test_weeks[wi: wi + EVAL_WINDOW])
    rows = df_test[df_test["Week start"].isin(eval_wks)]
    if len(rows) == 0:
        return None
    return round(float(np.abs(model.predict(rows[M0]) - rows[TARGET].values).mean()), 4)

def eval_arima_future(new_prov, wi):
    """Eval per-province ARIMA on EVAL_WINDOW steps ahead from trigger."""
    eval_wks = list(test_weeks[wi: wi + EVAL_WINDOW])
    if not eval_wks:
        return None
    errs = []
    for step, wk in enumerate(eval_wks, start=1):
        rows = df_test[df_test["Week start"] == wk]
        for _, row in rows.iterrows():
            p  = row["P-code"]
            st = new_prov.get(p)
            if st is None or st["result"] is None:
                fb = float(np.mean(st["hist"][-4:])) if st and st["hist"] else 0.0
                pr = fb
            else:
                try:
                    pr = float(st["result"].forecast(step).iloc[-1])
                except Exception:
                    pr = float(np.mean(st["hist"][-4:])) if st["hist"] else 0.0
            errs.append(abs(pr - row[TARGET]))
    return round(float(np.mean(errs)), 4) if errs else None

def retrain_lr_windows(wi):
    res = {}
    for w in RETRAIN_WINDOWS:
        train_wks = set(test_weeks[max(0, wi - w): wi])
        rows = df_test[df_test["Week start"].isin(train_wks)]
        if len(rows) < 30:
            res[f"{w}w"] = {"n": int(len(rows)), "mae_eval": None, "note": "too few rows"}
            continue
        m   = LinearRegression().fit(rows[M0], rows[TARGET])
        mae = eval_lr_future(m, wi)
        res[f"{w}w"] = {"n": int(len(rows)), "mae_eval": mae, "_model": m}
    return res

def retrain_arima_windows(wi):
    res = {}
    for w in RETRAIN_WINDOWS:
        t0 = time.time()
        train_wks = set(test_weeks[max(0, wi - w): wi])
        new_prov  = {}
        n_total   = 0
        for p in provinces:
            sub = df_test[(df_test["P-code"] == p) & (df_test["Week start"].isin(train_wks))]
            s   = sub.sort_values("Week start")[TARGET].values
            new_prov[p] = {"hist": list(s), "result": fit_arima(s)}
            n_total += len(s)
        mae     = eval_arima_future(new_prov, wi)
        elapsed = round(time.time() - t0, 1)
        res[f"{w}w"] = {"n": n_total, "mae_eval": mae, "time_s": elapsed, "_prov": new_prov}
        print(f"      ARIMA {w}w: n={n_total} eval_mae={mae} [{elapsed}s]", flush=True)
    return res

# ── main simulation loop ───────────────────────────────────────────────────────
print("\n=== Simulation ===", flush=True)

out = {
    "config": {
        "retrain_windows": RETRAIN_WINDOWS,
        "cusum_k_mult": CUSUM_K_MULT,
        "cusum_h_mult": CUSUM_H_MULT,
        "warmup_weeks": WARMUP_WEEKS,
        "eval_window": EVAL_WINDOW,
        "train_years": list(TRAIN_YEARS),
        "test_start_year": TEST_START_YEAR,
        "n_provinces": len(provinces),
        "n_features_m0": len(M0),
    },
    "weeks":        [],
    "lr_mae":       [],
    "ar_mae":       [],
    "lr_mae_base":  [],
    "ar_mae_base":  [],
    "cusum_lr":     [],
    "cusum_ar":     [],
    "triggers":     [],
}

for wi, week in enumerate(test_weeks):
    wdf = df_test[df_test["Week start"] == week]
    if len(wdf) == 0:
        continue

    # ── predict ───────────────────────────────────────────────────────────────
    actuals = wdf[TARGET].values

    lr_preds = lr_model.predict(wdf[M0])
    lr_mae   = float(np.abs(lr_preds - actuals).mean())

    lr_preds_base = lr_model_base.predict(wdf[M0])
    lr_mae_base   = float(np.abs(lr_preds_base - actuals).mean())

    ar_preds = []
    ar_preds_base = []
    for _, row in wdf.iterrows():
        p  = row["P-code"]
        fb = float(np.mean(prov_state[p]["hist"][-4:])) if prov_state[p]["hist"] else 0.0
        ar_preds.append(arima_forecast1(prov_state[p]["result"], fb))
        fb_b = float(np.mean(prov_state_base[p]["hist"][-4:])) if prov_state_base[p]["hist"] else 0.0
        ar_preds_base.append(arima_forecast1(prov_state_base[p]["result"], fb_b))
    ar_mae      = float(np.abs(np.array(ar_preds)       - actuals).mean())
    ar_mae_base = float(np.abs(np.array(ar_preds_base)  - actuals).mean())

    # ── update ARIMA state with actual observations ───────────────────────────
    for _, row in wdf.iterrows():
        p   = row["P-code"]
        obs = float(row[TARGET])
        prov_state[p]["hist"].append(obs)
        prov_state[p]["result"] = arima_append(prov_state[p]["result"], obs)
        prov_state_base[p]["hist"].append(obs)
        prov_state_base[p]["result"] = arima_append(prov_state_base[p]["result"], obs)

    # ── CUSUM ─────────────────────────────────────────────────────────────────
    lr_trig, lr_S = cusum_lr.step(lr_mae)
    ar_trig, ar_S = cusum_ar.step(ar_mae)

    out["weeks"].append(str(week.date()))
    out["lr_mae"].append(round(lr_mae, 4))
    out["ar_mae"].append(round(ar_mae, 4))
    out["lr_mae_base"].append(round(lr_mae_base, 4))
    out["ar_mae_base"].append(round(ar_mae_base, 4))
    out["cusum_lr"].append(lr_S)
    out["cusum_ar"].append(ar_S)

    # ── retrain on trigger ────────────────────────────────────────────────────
    for mname, triggered, cur_mae, retrain_fn, cusum in [
        ("linreg", lr_trig, lr_mae, lambda: retrain_lr_windows(wi),    cusum_lr),
        ("arima",  ar_trig, ar_mae, lambda: retrain_arima_windows(wi), cusum_ar),
    ]:
        if not triggered:
            continue

        print(f"\n  🔔 {mname} trigger @ {week.date()}  "
              f"S={cusum.S:.3f}  MAE={cur_mae:.4f}", flush=True)

        retrain_res = retrain_fn()

        # pick best window (must beat current)
        best_mae = cur_mae
        best_w   = None
        best_obj = None
        for wk, v in retrain_res.items():
            m = v.get("mae_eval")
            if m is not None and m < best_mae:
                best_mae = m
                best_w   = wk
                best_obj = v.get("_model") if mname == "linreg" else v.get("_prov")

        replaced = best_obj is not None
        if replaced:
            if mname == "linreg":
                lr_model = best_obj
            else:
                for p in provinces:
                    if p in best_obj:
                        prov_state[p]["result"] = best_obj[p]["result"]
            cusum.reset()
            print(f"  ✓ replaced → {best_w}  "
                  f"{cur_mae:.4f} → {best_mae:.4f}", flush=True)
        else:
            print("  ✗ no window improved — keeping current model", flush=True)

        # clean internal objects before JSON serialisation
        clean = {}
        for wk, v in retrain_res.items():
            clean[wk] = {k: val for k, val in v.items()
                         if not k.startswith("_")}

        out["triggers"].append({
            "week_index":   wi,
            "week":         str(week.date()),
            "model":        mname,
            "cusum_value":  round(cusum.S if not replaced else 0.0, 3),
            "mae_before":   round(cur_mae, 4),
            "mae_after":    round(best_mae, 4) if replaced else None,
            "winner_window": best_w,
            "replaced":     replaced,
            "windows":      clean,
        })

    trigger_this_week = [t for t in out["triggers"] if t["week_index"] == wi]
    print(json.dumps({"__progress__": {
        "wi":           wi,
        "n":            len(test_weeks),
        "week":         str(week.date()),
        "lr_mae":       round(lr_mae, 4),
        "ar_mae":       round(ar_mae, 4),
        "lr_mae_base":  round(lr_mae_base, 4),
        "ar_mae_base":  round(ar_mae_base, 4),
        "cusum_lr":     round(cusum_lr.S, 5),
        "cusum_ar":     round(cusum_ar.S, 5),
        "triggers":     trigger_this_week,
    }}), flush=True)

    if wi % 26 == 0:
        print(f"  [{wi+1:3d}/{len(test_weeks)}] {week.date()}  "
              f"LR={lr_mae:.3f}  AR={ar_mae:.3f}  "
              f"S_lr={lr_S:.2f}  S_ar={ar_S:.2f}", flush=True)

print(f"\n✓ Simulation done.  {len(out['triggers'])} trigger(s) fired.", flush=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f)
print(f"Saved → {OUT_PATH}", flush=True)
