"""modelling.py — รัน M0/M1/sensitivity experiments บนทุก fold พร้อม bootstrap CI

ใช้ร่วมกับ features.py (build_features) และ folds.py (make_folds, get_fold)
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dengue_pipeline.features import build_features
from dengue_pipeline.folds import get_fold, make_folds


# ---------------------------------------------------------------------------
# TabM-mini ensemble MLP — optional (ต้องการ torch)
# ---------------------------------------------------------------------------

class TabMMiniRegressor:
    """TabM-mini k=8 ensemble MLP · sklearn-compatible interface
    ต้องการ: pip install torch  (optional dep)
    """

    def __init__(self, k: int = 8, width: int = 64, dropout: float = 0.1,
                 lr: float = 0.002, weight_decay: float = 0.01,
                 batch_size: int = 512, n_epochs: int = 18,
                 n_seeds: int = 3, random_state: int = 342):
        self.k = k
        self.width = width
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_seeds = n_seeds
        self.random_state = random_state

    @staticmethod
    def _build_net(n_in: int, k: int, width: int, dropout: float):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(width, 1),
                    )
                    for _ in range(k)
                ])

            def forward(self, x):
                import torch
                return torch.stack(
                    [h(x).squeeze(-1) for h in self.heads], dim=1
                ).mean(dim=1)

        return _Net()

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import torch
        except ImportError:
            raise ImportError("TabMMini ต้องการ torch: pip install torch")
        # ป้องกัน OpenMP deadlock เมื่อรันหลัง XGBoost (libomp ชนกัน)
        torch.set_num_threads(1)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = len(X_t)

        self._nets: list = []
        for s in range(self.n_seeds):
            torch.manual_seed(self.random_state + s)
            net = self._build_net(X.shape[1], self.k, self.width, self.dropout)
            opt = torch.optim.AdamW(
                net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            net.train()
            for _ in range(self.n_epochs):
                perm = torch.randperm(n)
                for i in range(0, n, self.batch_size):
                    idx = perm[i: i + self.batch_size]
                    loss = torch.mean(torch.abs(net(X_t[idx]) - y_t[idx]))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            self._nets.append(net)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        torch.set_num_threads(1)
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        for net in self._nets:
            with torch.no_grad():
                preds.append(net(X_t).numpy())
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Bootstrap worker (รับ numpy arrays เท่านั้น เพราะ DuckDB connection ไม่ share ข้าม process)
# ---------------------------------------------------------------------------

def _bootstrap_worker(
    X_tr_np: np.ndarray,
    y_tr_np: np.ndarray,
    X_te_np: np.ndarray,
    y_te_np: np.ndarray,
    week_idx_te: np.ndarray,
    m0_idx: list[int],
    m1_idx: list[int],
    epsilon: float,
    alpha: float,
    seed: int,
) -> float:
    """บล็อก-สัปดาห์ bootstrap: สุ่มสัปดาห์ (ทุก row ของสัปดาห์นั้นติดมาด้วย)
    คืน delta_mae_pct = (mae0 - mae1) / mae0 (บวก = M1 ดีกว่า)
    """
    rng = np.random.default_rng(seed)
    unique_weeks = np.unique(week_idx_te)
    boot_weeks = rng.choice(unique_weeks, size=len(unique_weeks), replace=True)
    mask = np.isin(week_idx_te, boot_weeks)
    if mask.sum() == 0:
        return np.nan
    m0 = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500).fit(
        X_tr_np[:, m0_idx], y_tr_np)
    m1 = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500).fit(
        X_tr_np[:, m1_idx], y_tr_np)
    mae0 = np.mean(np.abs(m0.predict(X_te_np[mask][:, m0_idx]) - y_te_np[mask]))
    mae1 = np.mean(np.abs(m1.predict(X_te_np[mask][:, m1_idx]) - y_te_np[mask]))
    if mae0 == 0:
        return np.nan
    return (mae0 - mae1) / mae0


# ---------------------------------------------------------------------------
# สร้าง feature column lists จาก cfg_analysis และ feat DataFrame
# ---------------------------------------------------------------------------

def _build_col_lists(feat: pd.DataFrame, cfg_analysis: dict) -> dict[str, list[str]]:
    """คืน dict of column lists: m0, m1, all_feature_cols"""
    m0_cfg = cfg_analysis["features"]["M0"]
    seas_k = m0_cfg["seas_fourier_k"]

    hist_cols = m0_cfg["hist"]
    nat_cols = m0_cfg["nat"]
    seas_cols = [f"seas_sin{k}" for k in range(1, seas_k + 1)] + \
                [f"seas_cos{k}" for k in range(1, seas_k + 1)]
    reg_cols = sorted([c for c in feat.columns if c.startswith("reg_")])

    m0_cols = hist_cols + nat_cols + seas_cols + reg_cols

    wx_lags = cfg_analysis["features"]["M1"]["wx_lags"]
    wx_vars = cfg_analysis["features"]["M1"]["wx_vars"]
    wx_cols = [f"wx_{v}_lag{k}" for v in wx_vars for k in wx_lags]
    wxa_cols = [f"wxa_{v}_lag{k}" for v in wx_vars for k in wx_lags]

    m1_cols = m0_cols + wx_cols + wxa_cols

    # M1_lag58: M0 + อากาศ lag 5-8 (sensitivity analysis)
    wx_lags_ext = cfg_analysis["features"]["sensitivity"]["wx_lags_extended"]
    wx_cols_58 = [f"wx_{v}_lag{k}" for v in wx_vars for k in wx_lags_ext]
    wxa_cols_58 = [f"wxa_{v}_lag{k}" for v in wx_vars for k in wx_lags_ext]
    m1_lag58_cols = m0_cols + wx_cols_58 + wxa_cols_58

    return {"m0": m0_cols, "m1": m1_cols, "m1_lag58": m1_lag58_cols}


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

def run_experiment(con, cfg_analysis: dict) -> pd.DataFrame:
    """รัน M0/M1/sensitivity บน all folds

    con          = duckdb connection (read_only หรือ write ก็ได้)
    cfg_analysis = dict จาก config/analysis.toml
    คืน DataFrame ที่มี schema: run_id, config_hash, data_version, fold,
        model, feature_set, mae, delta_mae_pct, ci_lower, ci_upper, ci_method,
        n_train, n_test, created_at
    """
    TARGET = cfg_analysis["target"]["name"]
    model_cfg = cfg_analysis["model"]["primary"]
    epsilon = model_cfg["epsilon_locked"]
    alpha = model_cfg["alpha_locked"]
    n_boot = cfg_analysis["bootstrap"]["n_replicates"]
    boot_seed = cfg_analysis["bootstrap"]["seed"]
    ci_level = cfg_analysis["bootstrap"]["confidence_level"]
    sens_names = cfg_analysis["model"]["sensitivity"]["names"]

    # อ่าน data_version ล่าสุดจาก ops
    try:
        data_version = con.execute(
            "SELECT max(data_version) FROM ops.data_version").fetchone()[0] or ""
    except Exception:
        data_version = ""

    # build features
    print("[run_experiment] กำลังสร้าง feature table...")
    feat = build_features(con, cfg_analysis)
    print(f"[run_experiment] feat shape: {feat.shape}")

    col_lists = _build_col_lists(feat, cfg_analysis)
    m0_cols = col_lists["m0"]
    m1_cols = col_lists["m1"]
    m1_lag58_cols = col_lists["m1_lag58"]

    # ตรวจว่าคอลัมน์มีอยู่จริงใน feat
    missing_m0 = [c for c in m0_cols if c not in feat.columns]
    if missing_m0:
        raise ValueError(f"M0 cols ไม่อยู่ใน feat: {missing_m0[:5]}...")

    main_folds, supp_folds = make_folds(cfg_analysis)
    all_folds = main_folds + supp_folds
    fold_type_map = {f["name"]: "main" for f in main_folds}
    fold_type_map.update({f["name"]: "supp" for f in supp_folds})

    results = []
    created_at = datetime.datetime.now().isoformat(timespec="seconds")

    for fold in tqdm(all_folds, desc="folds"):
        fold_name = fold["name"]
        fold_data = get_fold(feat, fold, cfg_analysis)
        tr = fold_data["train"]
        te = fold_data["test"]

        # ตรวจว่า M1 cols ที่ต้องการมีใน fold data (wxa_* ถูกสร้างโดย get_fold)
        m1_cols_avail = [c for c in m1_cols if c in tr.columns]
        if set(m1_cols) != set(m1_cols_avail):
            missing_m1 = [c for c in m1_cols if c not in tr.columns]
            print(f"  [fold {fold_name}] M1 cols ที่ไม่มี: {missing_m1[:3]}... ใช้เฉพาะที่มี")
            m1_cols_use = m1_cols_avail
        else:
            m1_cols_use = m1_cols

        m1_lag58_avail = [c for c in m1_lag58_cols if c in tr.columns]
        m1_lag58_use = m1_lag58_avail

        # กรองแถว M0 (ไม่มี null ใน M0 cols + TARGET)
        m0_mask_tr = tr[m0_cols + [TARGET]].notna().all(axis=1)
        m0_mask_te = te[m0_cols + [TARGET]].notna().all(axis=1)
        tr_m0 = tr[m0_mask_tr]
        te_m0 = te[m0_mask_te]

        # กรองแถว M1 (ต้องไม่มี null ใน M1 cols ด้วย)
        m1_mask_tr = tr[m1_cols_use + [TARGET]].notna().all(axis=1)
        m1_mask_te = te[m1_cols_use + [TARGET]].notna().all(axis=1)
        tr_m1 = tr[m1_mask_tr]
        te_m1 = te[m1_mask_te]

        if len(tr_m0) == 0 or len(te_m0) == 0:
            print(f"  [fold {fold_name}] ข้ามเพราะ train/test ว่าง (M0)")
            continue

        X_tr_m0 = tr_m0[m0_cols].to_numpy()
        y_tr_m0 = tr_m0[TARGET].to_numpy()
        X_te_m0 = te_m0[m0_cols].to_numpy()
        y_te_m0 = te_m0[TARGET].to_numpy()

        scaler_m0 = StandardScaler().fit(X_tr_m0)
        X_tr_m0 = scaler_m0.transform(X_tr_m0)
        X_te_m0 = scaler_m0.transform(X_te_m0)

        # ---- Huber M0 ----
        hb_m0 = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
        hb_m0.fit(X_tr_m0, y_tr_m0)
        mae_m0 = float(np.mean(np.abs(hb_m0.predict(X_te_m0) - y_te_m0)))

        results.append({
            "run_id": None,
            "config_hash": None,
            "data_version": data_version,
            "fold": fold_name,
            "model": "HuberRegressor",
            "feature_set": "M0",
            "mae": mae_m0,
            "delta_mae_pct": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_method": None,
            "n_train": len(tr_m0),
            "n_test": len(te_m0),
            "created_at": created_at,
        })

        if len(tr_m1) == 0 or len(te_m1) == 0:
            print(f"  [fold {fold_name}] ข้าม M1 เพราะ train/test ว่าง")
            continue

        # ใช้ชุดแถวเดียวกัน M1/M1_lag58 (intersection ของทุก mask)
        all_cols_union = list(set(m0_cols) | set(m1_cols_use) | set(m1_lag58_use))
        full_mask_tr = tr[all_cols_union + [TARGET]].notna().all(axis=1)
        full_mask_te = te[all_cols_union + [TARGET]].notna().all(axis=1)
        tr_full = tr[full_mask_tr]
        te_full = te[full_mask_te]

        if len(tr_full) == 0 or len(te_full) == 0:
            print(f"  [fold {fold_name}] ข้าม M1 comparison เพราะไม่มีแถวร่วม")
            continue

        # สร้าง index mapping สำหรับ bootstrap
        all_cols_ordered = all_cols_union
        col_to_idx = {c: i for i, c in enumerate(all_cols_ordered)}
        m0_idx = [col_to_idx[c] for c in m0_cols]
        m1_idx = [col_to_idx[c] for c in m1_cols_use]
        m1_lag58_idx = [col_to_idx[c] for c in m1_lag58_use if c in col_to_idx]

        X_tr_np = tr_full[all_cols_ordered].to_numpy()
        y_tr_np = tr_full[TARGET].to_numpy()
        X_te_np = te_full[all_cols_ordered].to_numpy()
        y_te_np = te_full[TARGET].to_numpy()

        scaler = StandardScaler().fit(X_tr_np)
        X_tr_np = scaler.transform(X_tr_np)
        X_te_np = scaler.transform(X_te_np)

        # week index สำหรับ block bootstrap
        week_idx_te = pd.factorize(te_full["Week start"])[0]

        # ---- Huber M1 บน full intersection ----
        hb_m0_full = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
        hb_m0_full.fit(X_tr_np[:, m0_idx], y_tr_np)
        mae_m0_full = float(np.mean(np.abs(hb_m0_full.predict(X_te_np[:, m0_idx]) - y_te_np)))

        hb_m1 = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
        hb_m1.fit(X_tr_np[:, m1_idx], y_tr_np)
        mae_m1 = float(np.mean(np.abs(hb_m1.predict(X_te_np[:, m1_idx]) - y_te_np)))

        delta_pct = (mae_m0_full - mae_m1) / mae_m0_full if mae_m0_full > 0 else None

        # ---- Bootstrap CI (parallel) ----
        boot_deltas = Parallel(n_jobs=1)(
            delayed(_bootstrap_worker)(
                X_tr_np, y_tr_np, X_te_np, y_te_np, week_idx_te,
                m0_idx, m1_idx, epsilon, alpha, boot_seed + i
            )
            for i in tqdm(range(n_boot), desc=f"  boot fold={fold_name}", leave=False)
        )
        boot_arr = np.array([x for x in boot_deltas if x is not None and not np.isnan(x)])
        alpha_ci = (1 - ci_level) / 2
        ci_lower = float(np.quantile(boot_arr, alpha_ci)) if len(boot_arr) > 0 else None
        ci_upper = float(np.quantile(boot_arr, 1 - alpha_ci)) if len(boot_arr) > 0 else None

        results.append({
            "run_id": None,
            "config_hash": None,
            "data_version": data_version,
            "fold": fold_name,
            "model": "HuberRegressor",
            "feature_set": "M1",
            "mae": mae_m1,
            "delta_mae_pct": delta_pct,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_method": "block_week",
            "n_train": len(tr_full),
            "n_test": len(te_full),
            "created_at": created_at,
        })

        # ---- Huber M1_lag58 (sensitivity: อากาศ lag 5-8) ----
        if len(m1_lag58_idx) > 0:
            hb_m1_58 = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
            hb_m1_58.fit(X_tr_np[:, m1_lag58_idx], y_tr_np)
            mae_m1_58 = float(np.mean(np.abs(hb_m1_58.predict(X_te_np[:, m1_lag58_idx]) - y_te_np)))
            delta_pct_58 = (mae_m0_full - mae_m1_58) / mae_m0_full if mae_m0_full > 0 else None

            boot_deltas_58 = Parallel(n_jobs=1)(
                delayed(_bootstrap_worker)(
                    X_tr_np, y_tr_np, X_te_np, y_te_np, week_idx_te,
                    m0_idx, m1_lag58_idx, epsilon, alpha, boot_seed + i
                )
                for i in tqdm(range(n_boot), desc=f"  boot58 fold={fold_name}", leave=False)
            )
            boot_arr_58 = np.array([x for x in boot_deltas_58 if x is not None and not np.isnan(x)])
            ci_lower_58 = float(np.quantile(boot_arr_58, alpha_ci)) if len(boot_arr_58) > 0 else None
            ci_upper_58 = float(np.quantile(boot_arr_58, 1 - alpha_ci)) if len(boot_arr_58) > 0 else None

            results.append({
                "run_id": None,
                "config_hash": None,
                "data_version": data_version,
                "fold": fold_name,
                "model": "HuberRegressor",
                "feature_set": "M1_lag58",
                "mae": mae_m1_58,
                "delta_mae_pct": delta_pct_58,
                "ci_lower": ci_lower_58,
                "ci_upper": ci_upper_58,
                "ci_method": "block_week",
                "n_train": len(tr_full),
                "n_test": len(te_full),
                "created_at": created_at,
            })

        # ---- Sensitivity models ----
        for sens_name in sens_names:
            fs_sets = [("M0", m0_cols, m0_idx), ("M1", m1_cols_use, m1_idx)]
            if len(m1_lag58_idx) > 0:
                fs_sets.append(("M1_lag58", m1_lag58_use, m1_lag58_idx))
            for fs_label, fs_cols, fs_idx in fs_sets:
                try:
                    if sens_name == "ElasticNetCV":
                        model_s = ElasticNetCV(
                            l1_ratio=[0.1, 0.5, 0.9],
                            alphas=[1e-3, 1e-2, 1e-1],
                            cv=5,
                            max_iter=5000,
                            n_jobs=-1,
                            random_state=cfg_analysis["folds"]["seed"],
                        )
                    elif sens_name == "HistGradientBoostingRegressor":
                        model_s = HistGradientBoostingRegressor(
                            random_state=cfg_analysis["folds"]["seed"])
                    elif sens_name == "XGBRegressor":
                        from xgboost import XGBRegressor
                        model_s = XGBRegressor(
                            objective="reg:absoluteerror",
                            n_estimators=500,
                            learning_rate=0.03,
                            max_depth=4,
                            min_child_weight=10,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=5.0,
                            random_state=cfg_analysis["folds"]["seed"],
                            n_jobs=-1,
                            verbosity=0,
                        )
                    elif sens_name == "TabMMini":
                        model_s = TabMMiniRegressor(
                            random_state=cfg_analysis["folds"]["seed"])
                    else:
                        continue

                    model_s.fit(X_tr_np[:, fs_idx], y_tr_np)
                    mae_s = float(np.mean(np.abs(
                        model_s.predict(X_te_np[:, fs_idx]) - y_te_np)))
                    results.append({
                        "run_id": None,
                        "config_hash": None,
                        "data_version": data_version,
                        "fold": fold_name,
                        "model": sens_name,
                        "feature_set": fs_label,
                        "mae": mae_s,
                        "delta_mae_pct": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "ci_method": None,
                        "n_train": len(tr_full),
                        "n_test": len(te_full),
                        "created_at": created_at,
                    })
                except ImportError as exc:
                    print(f"  [fold {fold_name}] ข้าม {sens_name} ({fs_label}): {exc}")

    return pd.DataFrame(results)
