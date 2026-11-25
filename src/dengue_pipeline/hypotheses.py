"""hypotheses.py — ตรวจสมมติฐาน H1/H2/H3 ด้วย partial Spearman + placebo test

H1: ฝน (rain) มีสัญญาณต่อ D หลังหักฤดูกาลและประวัติ (lag 0-4)
H2: อุณหภูมิ (tmax/tmin) มีสัญญาณต่อ D ที่เหลืออยู่ (lag 0-4)
H3: สัญญาณอากาศอยู่ที่ lag ที่ใช้พยากรณ์ได้จริง (best lag ไม่อยู่ที่ขอบ 0 หรือ 8)
"""
from __future__ import annotations

import copy
import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from dengue_pipeline.features import build_features
from dengue_pipeline.folds import get_fold, make_folds


# ---------------------------------------------------------------------------
# Partial Spearman ระดับจังหวัด
# ---------------------------------------------------------------------------

def _partial_spearman(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    covariate: str,
    min_obs: int = 20,
) -> Optional[float]:
    """partial Spearman(target, predictor | covariate) สำหรับจังหวัดเดียว

    residualize target และ predictor บน covariate ด้วย OLS แล้ว Spearman ρ
    คืน None ถ้าข้อมูลไม่พอหรือ variance เป็นศูนย์
    """
    sub = df[[target, predictor, covariate]].dropna()
    if len(sub) < min_obs:
        return None

    X = np.column_stack([np.ones(len(sub)), sub[covariate].values])

    def _resid(y: np.ndarray) -> np.ndarray:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ beta

    r_y = _resid(sub[target].values)
    r_p = _resid(sub[predictor].values)

    if r_y.std() == 0 or r_p.std() == 0:
        return None

    rho, _ = stats.spearmanr(r_y, r_p)
    return float(rho)


# ---------------------------------------------------------------------------
# คำนวณ partial Spearman ต่อ fold
# ---------------------------------------------------------------------------

def _get_fold_extended(feat: pd.DataFrame, fold: dict, cfg_analysis: dict, lags: list[int]) -> dict:
    """get_fold โดยขยาย wx_lags ให้ครอบคลุม lags ที่ต้องการ"""
    cfg_h = copy.deepcopy(cfg_analysis)
    cfg_h["features"]["M1"]["wx_lags"] = lags
    return get_fold(feat, fold, cfg_h)


def compute_partial_spearman(
    feat: pd.DataFrame,
    fold: dict,
    cfg_analysis: dict,
    lags: list[int],
) -> pd.DataFrame:
    """คำนวณ partial Spearman ρ บน train rows ของ fold ต่อจังหวัด ต่อ var ต่อ lag

    คืน DataFrame: province, var, lag, rho
    """
    TARGET = cfg_analysis["target"]["name"]
    COVARIATE = cfg_analysis["hypotheses"]["H1"]["covariate"]
    wx_vars = list(dict.fromkeys(
        cfg_analysis["hypotheses"]["H1"]["vars"] +
        cfg_analysis["hypotheses"]["H2"]["vars"]
    ))

    tr = _get_fold_extended(feat, fold, cfg_analysis, lags)["train"]

    rows = []
    for pcode, grp in tr.groupby("P-code"):
        for var in wx_vars:
            for lag in lags:
                col = f"wxa_{var}_lag{lag}"
                if col not in tr.columns:
                    continue
                rho = _partial_spearman(grp, TARGET, col, COVARIATE)
                if rho is not None:
                    rows.append({"province": pcode, "var": var, "lag": lag, "rho": rho})

    return pd.DataFrame(rows)


def aggregate_rho(rho_df: pd.DataFrame, fold_name: str) -> pd.DataFrame:
    """aggregate: median ρ และ median |ρ| ข้ามจังหวัด ต่อ var ต่อ lag"""
    if rho_df.empty:
        return pd.DataFrame()
    agg = (rho_df.groupby(["var", "lag"])["rho"]
           .agg(median_rho="median",
                median_abs_rho=lambda x: x.abs().median(),
                n_prov="count")
           .reset_index())
    agg["fold"] = fold_name
    return agg


# ---------------------------------------------------------------------------
# Decision H1/H2
# ---------------------------------------------------------------------------

def decide_h1h2(all_agg: pd.DataFrame, cfg_analysis: dict) -> dict:
    """ตัดสิน H1/H2 จาก aggregated results ทุก fold (5 main folds)

    คืน dict: var → {"passed": bool, "pass_lags": list, "n_pass_lags": int}
    """
    threshold = cfg_analysis["hypotheses"]["H1"]["threshold_abs_rho"]
    min_pass_lags = cfg_analysis["hypotheses"]["H1"]["min_pass_lags"]
    min_sign_folds = cfg_analysis["hypotheses"]["H1"]["same_sign_min_folds"]
    main_folds = [str(y) for y in cfg_analysis["folds"]["test_years_main"]]

    decisions = {}
    for var in all_agg["var"].unique():
        sub_main = all_agg[(all_agg["var"] == var) & (all_agg["fold"].isin(main_folds))]
        pass_lags = []
        for lag, grp in sub_main.groupby("lag"):
            if grp["median_abs_rho"].median() < threshold:
                continue
            signs = np.sign(grp["median_rho"])
            if max((signs > 0).sum(), (signs < 0).sum()) >= min_sign_folds:
                pass_lags.append(int(lag))
        decisions[var] = {
            "passed": len(pass_lags) >= min_pass_lags,
            "pass_lags": pass_lags,
            "n_pass_lags": len(pass_lags),
        }
    return decisions


# ---------------------------------------------------------------------------
# Placebo test
# ---------------------------------------------------------------------------

def run_placebo(
    feat: pd.DataFrame,
    fold: dict,
    cfg_analysis: dict,
    lags: list[int],
    n: int = 200,
    seed: int = 342,
) -> pd.DataFrame:
    """placebo: สลับ wxa rows ภายในจังหวัดแบบสุ่ม n รอบ

    คืน DataFrame: fold, var, lag, rho_real, placebo_median, placebo_p95, n_placebo
    """
    TARGET = cfg_analysis["target"]["name"]
    COVARIATE = cfg_analysis["hypotheses"]["H1"]["covariate"]
    wx_vars = list(dict.fromkeys(
        cfg_analysis["hypotheses"]["H1"]["vars"] +
        cfg_analysis["hypotheses"]["H2"]["vars"]
    ))
    wx_cols = [f"wxa_{var}_lag{lag}" for var in wx_vars for lag in lags]

    tr = _get_fold_extended(feat, fold, cfg_analysis, lags)["train"]
    rng = np.random.default_rng(seed)

    # real ρ (median ข้ามจังหวัด)
    real_rows = []
    for pcode, grp in tr.groupby("P-code"):
        for var in wx_vars:
            for lag in lags:
                col = f"wxa_{var}_lag{lag}"
                if col not in tr.columns:
                    continue
                rho = _partial_spearman(grp, TARGET, col, COVARIATE)
                if rho is not None:
                    real_rows.append({"province": pcode, "var": var, "lag": lag, "rho": rho})

    real_df = pd.DataFrame(real_rows)
    if real_df.empty:
        return pd.DataFrame()

    real_median = real_df.groupby(["var", "lag"])["rho"].median()

    # placebo replicates: permute wxa rows ภายในจังหวัด
    placebo_store: dict[tuple, list] = {(var, lag): [] for var in wx_vars for lag in lags}

    for _ in range(n):
        tr_p = tr.copy()
        for pcode, grp_idx in tr.groupby("P-code").groups.items():
            perm = rng.permutation(len(grp_idx))
            existing = [c for c in wx_cols if c in tr_p.columns]
            tr_p.loc[grp_idx, existing] = tr_p.loc[grp_idx, existing].values[perm]

        for pcode, grp in tr_p.groupby("P-code"):
            for var in wx_vars:
                for lag in lags:
                    col = f"wxa_{var}_lag{lag}"
                    if col not in tr_p.columns:
                        continue
                    rho = _partial_spearman(grp, TARGET, col, COVARIATE)
                    if rho is not None:
                        placebo_store[(var, lag)].append(rho)

    result_rows = []
    for (var, lag), rho_real in real_median.items():
        p_arr = np.array(placebo_store[(var, lag)])
        result_rows.append({
            "fold": fold["name"],
            "var": var,
            "lag": int(lag),
            "rho_real": float(rho_real),
            "placebo_median": float(np.median(p_arr)) if len(p_arr) > 0 else None,
            "placebo_p95": float(np.percentile(np.abs(p_arr), 95)) if len(p_arr) > 0 else None,
            "n_placebo": len(p_arr),
        })
    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Entry point หลัก
# ---------------------------------------------------------------------------

def run_h_analysis(con, cfg_analysis: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """รัน H1/H2/H3 analysis ทั้งหมด

    คืน (rho_df, placebo_df, decisions)
      rho_df     — partial Spearman aggregated ทุก fold ทุก var ทุก lag (column: analysis="H1H2"|"H3")
      placebo_df — placebo results บน main folds
      decisions  — {"rain": {...}, "tmax": {...}, "tmin": {...}, "H3": {...}}
    """
    print("[analyze] กำลังสร้าง feature table...")
    feat = build_features(con, cfg_analysis)
    print(f"[analyze] feat shape: {feat.shape}")

    h1_lags = cfg_analysis["hypotheses"]["H1"]["lags"]
    h3_lags = cfg_analysis["hypotheses"]["H3"]["lags"]
    main_folds, supp_folds = make_folds(cfg_analysis)
    all_folds = main_folds + supp_folds

    # --- H1/H2: partial Spearman lag 0-4 บนทุก fold ---
    all_rho = []
    for fold in tqdm(all_folds, desc="H1/H2 folds"):
        rho_df = compute_partial_spearman(feat, fold, cfg_analysis, lags=h1_lags)
        all_rho.append(aggregate_rho(rho_df, fold["name"]))

    rho_h12 = pd.concat([r for r in all_rho if not r.empty], ignore_index=True)
    rho_h12["analysis"] = "H1H2"

    decisions_h12 = decide_h1h2(rho_h12, cfg_analysis)

    h1_vars = cfg_analysis["hypotheses"]["H1"]["vars"]
    h2_vars = cfg_analysis["hypotheses"]["H2"]["vars"]
    h1_pass = any(decisions_h12.get(v, {}).get("passed", False) for v in h1_vars)
    h2_pass = any(decisions_h12.get(v, {}).get("passed", False) for v in h2_vars)
    h3_eligible = h1_pass or h2_pass

    # --- H3: lag 0-8 บน main folds (ถ้า H1 หรือ H2 ผ่าน) ---
    h3_rho_frames = []
    h3_decision: dict = {"eligible": h3_eligible}

    if h3_eligible:
        print("[analyze] H1 หรือ H2 ผ่าน → รัน H3 (lag 0-8) บน main folds")
        for fold in tqdm(main_folds, desc="H3 folds"):
            rho_df = compute_partial_spearman(feat, fold, cfg_analysis, lags=h3_lags)
            h3_rho_frames.append(aggregate_rho(rho_df, fold["name"]))

        rho_h3 = pd.concat([r for r in h3_rho_frames if not r.empty], ignore_index=True)
        rho_h3["analysis"] = "H3"

        edge_lags = cfg_analysis["hypotheses"]["H3"]["edge_lags"]
        best = rho_h3.groupby(["var", "lag"])["median_abs_rho"].mean().reset_index()
        for var in best["var"].unique():
            best_lag = int(best[best["var"] == var]
                           .sort_values("median_abs_rho", ascending=False)
                           .iloc[0]["lag"])
            h3_decision[var] = {
                "best_lag": best_lag,
                "at_edge": best_lag in edge_lags,
                "passed": best_lag not in edge_lags,
            }

        rho_all = pd.concat([rho_h12, rho_h3], ignore_index=True)
    else:
        print("[analyze] H1 และ H2 ไม่ผ่านทั้งคู่ → H3 = ตัดสินไม่ได้")
        rho_all = rho_h12

    decisions = {**decisions_h12, "H3": h3_decision}

    # --- Placebo: main folds ---
    print("[analyze] รัน placebo test บน main folds...")
    placebo_frames = []
    for fold in tqdm(main_folds, desc="placebo folds"):
        placebo_frames.append(
            run_placebo(feat, fold, cfg_analysis, lags=h1_lags,
                        seed=cfg_analysis["bootstrap"]["seed"])
        )
    placebo_all = pd.concat([p for p in placebo_frames if not p.empty], ignore_index=True)

    return rho_all, placebo_all, decisions
