"""folds.py — สร้าง fold definitions และแบ่งข้อมูล train/test พร้อม anomaly per fold

SQL logic เหมือน docs/notebook_snippets/get_fold_sql.py ทุกจุด
ต่างกันตรงที่อ่าน constants จาก cfg_analysis แทนตัวแปร global
"""
from __future__ import annotations

import duckdb
import pandas as pd

from dengue_pipeline.features import WX_BASE


def make_folds(cfg_analysis: dict) -> tuple[list, list]:
    """สร้าง MAIN_FOLDS และ SUPP_FOLDS จาก cfg_analysis

    คืน (MAIN_FOLDS, SUPP_FOLDS) แต่ละ element เป็น dict:
        {"name": str, "train_years": (lo, hi), "test_year": int}
    """
    test_main = cfg_analysis["folds"]["test_years_main"]
    test_supp = cfg_analysis["folds"]["test_years_supplementary"]
    all_years = sorted(set(test_main) | set(test_supp))
    # train_start คงที่ที่ 2016 (ปีแรกของข้อมูล)
    train_start = 2016

    def _fold(test_year: int) -> dict:
        # expanding train: train ถึงปีก่อน test_year (เว้น gap 4 สัปดาห์ผ่าน label_end < test_start)
        return {
            "name": str(test_year),
            "train_years": (train_start, test_year - 1),
            "test_year": test_year,
        }

    main_folds = [_fold(y) for y in test_main]
    supp_folds = [_fold(y) for y in test_supp]
    return main_folds, supp_folds


def get_fold(feat: pd.DataFrame, fold: dict, cfg_analysis: dict) -> dict:
    """คืน {"train": df, "test": df, "test_start": Timestamp}

    คำนวณ anomaly per fold ด้วย SQL climatology จาก train เท่านั้น (เหมือน get_fold_sql)
    ใช้ duckdb in-memory ชั่วคราว ไม่ depend on external connection
    """
    MAX_WX_LAG = max(cfg_analysis["features"]["M1"]["wx_lags"])
    TARGET = cfg_analysis["target"]["name"]

    con = duckdb.connect()
    try:
        con.register("feat", feat)

        # สร้าง ALL_FEATURES เพื่อกรองแถวที่ไม่ null (เหมือนใน notebook)
        # ใช้ feature groups จาก cfg_analysis
        m0_cfg = cfg_analysis["features"]["M0"]
        hist_cols = m0_cfg["hist"]
        nat_cols = m0_cfg["nat"]
        seas_k = m0_cfg["seas_fourier_k"]

        # seas cols: seas_sin1..k, seas_cos1..k
        seas_cols = [f"seas_sin{k}" for k in range(1, seas_k + 1)] + \
                    [f"seas_cos{k}" for k in range(1, seas_k + 1)]

        # reg cols: อ่านจาก feat เอง (reg_1..reg_13)
        reg_cols = [c for c in feat.columns if c.startswith("reg_")]

        # wx cols สำหรับ M1 (lag 0-4)
        wx_lags = cfg_analysis["features"]["M1"]["wx_lags"]
        wx_vars = cfg_analysis["features"]["M1"]["wx_vars"]
        wx_cols = [f"wx_{v}_lag{k}" for v in wx_vars for k in wx_lags]

        ALL_FEATURES = hist_cols + nat_cols + seas_cols + reg_cols

        # SQL logic เหมือน _fold_sql ใน notebook snippets
        W = 'OVER (PARTITION BY "P-code" ORDER BY "Week start")'
        keys = list(WX_BASE)
        clim = ", ".join(f'avg(wk_{k}) AS wk_{k}_clim' for k in keys)
        anom = ", ".join(f'wk_{k} - wk_{k}_clim AS anom_{k}' for k in keys)
        wxa = ", ".join(
            f'anom_{k} AS wxa_{k}_lag0' if i == 0 else
            f"CASE WHEN date_diff('day', CAST(lag(\"Week start\", {i}) {W} AS DATE), CAST(\"Week start\" AS DATE)) = {7*i} "
            f'THEN lag(anom_{k}, {i}) {W} END AS wxa_{k}_lag{i}'
            for k in keys for i in range(MAX_WX_LAG + 1))
        usable = " AND ".join([f'"{TARGET}" IS NOT NULL'] + [f'"{c}" IS NOT NULL' for c in ALL_FEATURES])
        lo, hi = fold["train_years"]

        sql = f'''
        WITH clim AS (
            SELECT "P-code", "Month", {clim} FROM feat
            WHERE week_days = 7 AND "Week end" < $1 GROUP BY 1, 2),
        a AS (
            SELECT f.*, {", ".join(f"c.wk_{k}_clim" for k in keys)}
            FROM feat f LEFT JOIN clim c ON f."P-code" = c."P-code" AND f."Month" = c."Month"),
        b AS (SELECT *, {anom} FROM a),
        d AS (SELECT *, {wxa} FROM b)
        SELECT * EXCLUDE (_tr, _te) , _tr, _te FROM (
            SELECT *, ({usable}) AS _ok,
                   ("Year" BETWEEN {lo} AND {hi} AND label_end < $1) AS _tr,
                   ("Year" = {fold["test_year"]}) AS _te
            FROM d) WHERE _ok AND (_tr OR _te)
        ORDER BY "P-code", "Week start"
        '''

        test_start = pd.Timestamp(f"{fold['test_year']}-01-01")
        d = con.execute(sql, [test_start.to_pydatetime()]).df()
        tr = d[d["_tr"]].drop(columns=["_ok", "_tr", "_te"]).reset_index(drop=True)
        te = d[d["_te"]].drop(columns=["_ok", "_tr", "_te"]).reset_index(drop=True)

        # unit test ของ fold
        assert tr["label_end"].max() < test_start, "label ของ train ล้ำเข้าช่วงทดสอบ"
        assert te["Week start"].min() >= test_start, "test เริ่มก่อนต้นปีทดสอบ"

        return {"train": tr, "test": te, "test_start": test_start}
    finally:
        con.close()
