"""TabM-mini sensitivity — M0 vs M1 ครบ 5 fold (ไม่ import xgboost)"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time, warnings, duckdb
import numpy as np
import pandas as pd
import torch
import tabm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(342); np.random.seed(342)

SEED = 342; TARGET = "D"; MAX_WX_LAG = 4; FOURIER_K = 3
WX_BASE = {"rain": "rain_wk_mm", "tmax": "tmax_mean_c", "tmin": "tmin_mean_c"}
USE_NAT = True; DEVICE = "cpu"
DB_PATH = ROOT / "data" / "pipeline.duckdb"
DB_TABLE = "mart.trends_weather_latest"

MAIN_FOLDS = [
    {"name": "F1", "train_years": (2016, 2018), "test_year": 2019},
    {"name": "F2", "train_years": (2016, 2021), "test_year": 2022},
    {"name": "F3", "train_years": (2016, 2022), "test_year": 2023},
    {"name": "F4", "train_years": (2016, 2023), "test_year": 2024},
    {"name": "F5", "train_years": (2016, 2024), "test_year": 2025},
]

import docs.notebook_snippets.build_features_sql as _bfm
_bfm.WX_BASE = WX_BASE; _bfm.MAX_WX_LAG = MAX_WX_LAG
_bfm.FOURIER_K = FOURIER_K; _bfm.TARGET = TARGET
from docs.notebook_snippets.build_features_sql import build_features_sql

print("Loading...", flush=True)
con = duckdb.connect(str(DB_PATH), read_only=True)
raw = con.execute(f"SELECT * EXCLUDE(data_version) FROM {DB_TABLE}").df()
con.close()
df = build_features_sql(raw)
print(f"  df {df.shape}", flush=True)

HIST = ["hist_lograte_lag0","hist_lograte_lag1","hist_lograte_lag2","hist_lograte_lag3",
        "hist_logR_prev","hist_logR_prev2","hist_growth_prev"]
NAT  = ["nat_lograte_lag0","nat_logR_prev","nat_growth_prev"]
SEAS = [f"seas_sin{k}" for k in range(1,FOURIER_K+1)] + [f"seas_cos{k}" for k in range(1,FOURIER_K+1)]
REG  = [c for c in df.columns if c.startswith("reg_")]
WX   = [f"wx_{v}_lag{k}" for v in WX_BASE for k in range(MAX_WX_LAG+1)]
WXA  = [f"wxa_{v}_lag{k}" for v in WX_BASE for k in range(MAX_WX_LAG+1)]
ALL  = HIST + (NAT if USE_NAT else []) + SEAS + REG + WX + WXA
M0   = HIST + (NAT if USE_NAT else []) + SEAS + REG
M1   = M0 + WX + WXA

_fcon = duckdb.connect()
_fcon.execute("SET threads=1")
_fcon.register("feat", df)
_CACHE = {}

def get_fold(fold):
    if fold["name"] in _CACHE:
        return _CACHE[fold["name"]]
    ws = '"Week start"'
    W  = 'OVER (PARTITION BY "P-code" ORDER BY "Week start")'
    keys = list(WX_BASE)
    clim = ", ".join(f'avg(wk_{k}) AS wk_{k}_clim' for k in keys)
    anom = ", ".join(f'wk_{k} - wk_{k}_clim AS anom_{k}' for k in keys)
    wxa_parts = []
    for k in keys:
        for i in range(MAX_WX_LAG+1):
            if i == 0:
                wxa_parts.append(f'anom_{k} AS wxa_{k}_lag0')
            else:
                lag_ws = f'lag({ws}, {i}) {W}'
                cond   = f"date_diff('day', CAST({lag_ws} AS DATE), CAST({ws} AS DATE)) = {i*7}"
                wxa_parts.append(f'CASE WHEN {cond} THEN lag(anom_{k}, {i}) {W} END AS wxa_{k}_lag{i}')
    wxa    = ", ".join(wxa_parts)
    usable = " AND ".join([f'"{TARGET}" IS NOT NULL'] + [f'"{c}" IS NOT NULL' for c in ALL])
    lo, hi = fold["train_years"]
    test_start = pd.Timestamp(f"{fold['test_year']}-01-01")
    sql = f"""
    WITH clim AS (SELECT "P-code","Month",{clim} FROM feat
                  WHERE week_days=7 AND "Week end"<$1 GROUP BY 1,2),
    a AS (SELECT f.*,{', '.join(f'c.wk_{k}_clim' for k in keys)}
          FROM feat f LEFT JOIN clim c ON f."P-code"=c."P-code" AND f."Month"=c."Month"),
    b AS (SELECT *,{anom} FROM a),
    d AS (SELECT *,{wxa} FROM b)
    SELECT * EXCLUDE(_tr,_te),_tr,_te FROM (
        SELECT *,({usable}) AS _ok,
               ("Year" BETWEEN {lo} AND {hi} AND label_end<$1) AS _tr,
               ("Year"={fold['test_year']}) AS _te
        FROM d) WHERE _ok AND (_tr OR _te)
    ORDER BY "P-code","Week start"
    """
    d  = _fcon.execute(sql, [test_start.to_pydatetime()]).df()
    tr = d[d["_tr"]].drop(columns=["_ok","_tr","_te"]).reset_index(drop=True)
    te = d[d["_te"]].drop(columns=["_ok","_tr","_te"]).reset_index(drop=True)
    _CACHE[fold["name"]] = {"train": tr, "test": te}
    return _CACHE[fold["name"]]

class TabMWrapper:
    def __init__(self, n_feat, epochs=80, batch=512, lr=1e-3,
                 d_block=64, n_blocks=2, k=16, patience=8):
        torch.manual_seed(SEED)
        self.epochs = epochs; self.batch = batch; self.patience = patience
        self.scaler = StandardScaler()
        self.model  = tabm.TabM.make(
            n_num_features=n_feat, d_out=1,
            d_block=d_block, n_blocks=n_blocks, k=k,
            arch_type="tabm-mini",
        ).to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y.values if hasattr(y,"values") else np.array(y), dtype=torch.float32)
        n  = len(Xt)
        best_loss, wait, best_w = float("inf"), 0, None
        self.model.train()
        for ep in range(self.epochs):
            idx = torch.randperm(n)
            ep_loss = 0.0
            for s in range(0, n, self.batch):
                b    = idx[s:s+self.batch]
                pred = self.model(x_num=Xt[b]).mean(dim=1).squeeze(-1)
                loss = torch.mean(torch.abs(pred - yt[b]))
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                ep_loss += loss.item() * len(b)
            ep_loss /= n
            if ep_loss < best_loss - 1e-5:
                best_loss, wait = ep_loss, 0
                best_w = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_w:
            self.model.load_state_dict(best_w)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xt), self.batch):
                out = self.model(x_num=Xt[s:s+self.batch]).mean(dim=1).squeeze(-1)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)

def run_tabm(feats, feat_set):
    rows = []
    for fold in MAIN_FOLDS:
        t0 = time.time()
        fd = get_fold(fold)
        tr, te = fd["train"], fd["test"]
        m = TabMWrapper(n_feat=len(feats))
        m.fit(tr[feats], tr[TARGET])
        pred = np.asarray(m.predict(te[feats]), dtype=float)
        y    = te[TARGET].values
        mae  = float(np.abs(y - pred).mean())
        mae0 = float(np.abs(y).mean())
        rows.append({"model": "TabM-mini", "feature_set": feat_set,
                     "fold": fold["name"], "test_year": fold["test_year"],
                     "mae": round(mae,4), "mae_zero": round(mae0,4),
                     "skill_pct": round(100*(1-mae/mae0),2)})
        print(f"  TabM {feat_set} {fold['name']} ({fold['test_year']}) MAE={mae:.4f} [{time.time()-t0:.1f}s]", flush=True)
    return pd.DataFrame(rows)

print("\n=== TabM-mini M0 ===", flush=True); r_m0 = run_tabm(M0, "M0")
print("\n=== TabM-mini M1 ===", flush=True); r_m1 = run_tabm(M1, "M1")

comp = r_m0[["fold","test_year","mae"]].merge(r_m1[["fold","mae"]], on="fold", suffixes=("_m0","_m1"))
comp["delta_pct"] = (100*(1 - comp["mae_m1"]/comp["mae_m0"])).round(2)
comp["pass_5pct"] = comp["delta_pct"] >= 5.0
print(f"\nTabM-mini avg ΔMAE={comp['delta_pct'].mean():.2f}%  pass={int(comp['pass_5pct'].sum())}/5")
print(comp.to_string(index=False))

out = pd.concat([r_m0, r_m1], ignore_index=True)
out_path = ROOT / "data" / "sensitivity_tabm.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
