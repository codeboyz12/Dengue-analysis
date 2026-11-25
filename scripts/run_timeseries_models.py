"""
รัน ARIMA(1,1,1) และ Holt-Winters (ETS) ต่อจังหวัดต่อ fold
แล้วคำนวณ MAE เทียบกับ D (4-week ahead log-rate change)
ผลเพิ่มเข้า reports/static/data.json ในส่วน h4.sensitivity
"""
import json, warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent.parent
DB     = ROOT / "data" / "pipeline.duckdb"
OUTJSON = ROOT / "reports" / "static" / "data.json"

FOLDS_MAIN = {
    "2019": pd.Timestamp("2019-01-01"),
    "2022": pd.Timestamp("2022-01-01"),
    "2023": pd.Timestamp("2023-01-01"),
    "2024": pd.Timestamp("2024-01-01"),
    "2025": pd.Timestamp("2025-01-01"),
}
HORIZON = 4   # สัปดาห์ล่วงหน้า


# ── โหลดข้อมูล ─────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("""
        SELECT pcode, ws, rate, rain, R_prev, R_next,
               ln(1 + R_next) - ln(1 + R_prev) AS D,
               year(ws) AS yr,
               date_diff('day', CAST(ws_l3 AS DATE), CAST(ws AS DATE)) AS days_back,
               date_diff('day', CAST(ws AS DATE), CAST(ws_f4 AS DATE)) AS days_fwd
        FROM (
            SELECT
                "P-code" AS pcode,
                "Week start" AS ws,
                CAST("Dengue rate per 100,000" AS DOUBLE) AS rate,
                CAST(rain_wk_mm AS DOUBLE) AS rain,
                AVG(CAST("Dengue rate per 100,000" AS DOUBLE)) OVER (
                    PARTITION BY "P-code" ORDER BY "Week start"
                    ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                ) AS R_prev,
                AVG(CAST("Dengue rate per 100,000" AS DOUBLE)) OVER (
                    PARTITION BY "P-code" ORDER BY "Week start"
                    ROWS BETWEEN 1 FOLLOWING AND 4 FOLLOWING
                ) AS R_next,
                lag("Week start", 3) OVER (PARTITION BY "P-code" ORDER BY "Week start") AS ws_l3,
                lead("Week start", 4) OVER (PARTITION BY "P-code" ORDER BY "Week start") AS ws_f4
            FROM mart.trends_weather_latest
            WHERE week_days = 7
        ) sub
        WHERE date_diff('day', CAST(ws_l3 AS DATE), CAST(ws AS DATE)) = 21
          AND date_diff('day', CAST(ws AS DATE), CAST(ws_f4 AS DATE)) = 28
        ORDER BY pcode, ws
    """).df()
    con.close()
    return df


# ── สร้าง rain exog matrix (lag 0–4) ─────────────────────────────────────────
def build_rain_exog(rain_series: np.ndarray) -> np.ndarray:
    """คืน matrix (n, 5) คอลัมน์ k = rain[t-k] สำหรับ lag 0,1,2,3,4"""
    n = len(rain_series)
    med = float(np.nanmedian(rain_series))
    out = np.full((n, 5), med)
    for k in range(5):
        src = np.where(np.isnan(rain_series), med, rain_series)
        out[k:, k] = src[:n - k]
    return out


def build_rain_exog_lag58(rain_series: np.ndarray) -> np.ndarray:
    """คืน matrix (n, 4) คอลัมน์ i = rain[t-lag] สำหรับ lag 5,6,7,8"""
    n = len(rain_series)
    med = float(np.nanmedian(rain_series))
    out = np.full((n, 4), med)
    src = np.where(np.isnan(rain_series), med, rain_series)
    for i, k in enumerate([5, 6, 7, 8]):
        if k < n:
            out[k:, i] = src[:n - k]
    return out


# ── SARIMA open-loop forecast ─────────────────────────────────────────────────
def sarima_openloop_forecast(train_log_rate: np.ndarray, n_steps: int) -> np.ndarray:
    """SARIMA(1,1,1)(1,0,0,52) open-loop จาก train end"""
    try:
        result = SARIMAX(
            train_log_rate,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 52),
        ).fit(disp=False)
        return result.get_forecast(steps=n_steps).predicted_mean
    except Exception:
        # fallback ARIMA(1,1,1)
        try:
            result = ARIMA(train_log_rate, order=(1, 1, 1)).fit()
            return result.get_forecast(steps=n_steps).predicted_mean
        except Exception:
            last = train_log_rate[-1]
            return np.full(n_steps, last)


# ── คำนวณ D จาก forecast ─────────────────────────────────────────────────────
def forecast_to_d(fc: np.ndarray, offset: int, R_prev_arr: np.ndarray) -> np.ndarray:
    """
    fc     : forecast ใน log1p(rate) space จากท้าย train ถึงสิ้นสุด
    offset : จำนวน step จาก train end ถึง test week ที่ 1
    R_prev_arr : ค่า R_prev จริงของแต่ละ test week
    คืน D_pred array ขนาด len(R_prev_arr)
    """
    D_preds = []
    for j, R_prev in enumerate(R_prev_arr):
        # ตำแหน่งใน fc ที่ตรงกับ test week j + 1..4 (ค่าในอนาคต 4 สัปดาห์)
        i0 = offset + j + 1
        i1 = i0 + HORIZON
        if i1 > len(fc):
            D_preds.append(np.nan)
            continue
        R_next_pred = max(0.0, np.mean(np.expm1(fc[i0:i1])))
        D_preds.append(np.log1p(R_next_pred) - np.log1p(max(0.0, R_prev)))
    return np.array(D_preds)


# ── loop หลัก ────────────────────────────────────────────────────────────────
def run_all(panel: pd.DataFrame) -> dict:
    """คืน {model: {fold: mae}}"""
    results = {
        "Naive(D=0)":               {},   # D_pred=0
        "SeasonalNaive":            {},   # R_next จากปีที่แล้ว
        "ARIMA(1,1,1)":             {},   # walk-forward
        "ARIMAX+rain(oracle)":      {},   # walk-forward + rain จริง lag 0-4 (oracle)
        "ARIMAX+rain(lag5-8)":      {},   # walk-forward + rain lag 5-8 (ไม่ต้องการ oracle)
        "SARIMA(1,1,1)(1,0,0,52)":  {},   # open-loop จาก train end
    }
    provinces = panel["pcode"].unique()

    for fold_name, test_start in FOLDS_MAIN.items():
        test_yr = int(fold_name)
        print(f"  fold {fold_name}...", flush=True)

        fold_maes_naive       = []
        fold_maes_snaive      = []
        fold_maes_arima       = []
        fold_maes_arimax      = []
        fold_maes_arimax_lag58 = []
        fold_maes_sarima      = []

        for pcode in provinces:
            prov = panel[panel["pcode"] == pcode].sort_values("ws").reset_index(drop=True)

            # train: rate จนถึง test_start (ก่อนปีทดสอบ)
            # test:  แถวในปีทดสอบที่มี D ไม่ null
            train_mask = prov["ws"] < test_start
            test_mask  = (prov["yr"] == test_yr) & prov["D"].notna()

            if train_mask.sum() < 60 or test_mask.sum() < 4:
                continue  # ข้อมูลน้อยเกินไป

            train_rate     = prov.loc[train_mask, "rate"].values
            train_log_rate = np.log1p(np.maximum(train_rate, 0))

            test_rows  = prov[test_mask].reset_index(drop=True)
            D_actual   = test_rows["D"].values
            R_prev_arr = test_rows["R_prev"].values

            # ── Naive D=0 ──
            D_naive = np.zeros(len(D_actual))
            fold_maes_naive.append(np.mean(np.abs(D_naive - D_actual)))

            # ── Seasonal Naive: R_next_pred = rate จากปีที่แล้วช่วงเดียวกัน ──
            D_snaive = []
            for j in range(len(test_rows)):
                ws_t = test_rows["ws"].iloc[j]
                # หาแถวที่ week start ≈ ws_t - 52 สัปดาห์ → R_next ของปีที่แล้ว
                target_prev_yr = ws_t - pd.Timedelta(weeks=52)
                prev_rows = prov[
                    (prov["ws"] >= target_prev_yr - pd.Timedelta(days=3)) &
                    (prov["ws"] <= target_prev_yr + pd.Timedelta(days=3)) &
                    prov["R_next"].notna()
                ]
                if len(prev_rows) > 0:
                    R_next_pred = float(prev_rows["R_next"].iloc[0])
                    R_prev_now  = float(test_rows["R_prev"].iloc[j])
                    D_snaive.append(np.log1p(max(0.0, R_next_pred)) - np.log1p(max(0.0, R_prev_now)))
                else:
                    D_snaive.append(0.0)
            D_snaive = np.array(D_snaive)
            fold_maes_snaive.append(np.mean(np.abs(D_snaive - D_actual)))

            # ── ARIMA walk-forward + ARIMAX oracle ──
            test_rate     = prov.loc[test_mask, "rate"].values
            test_log_rate = np.log1p(np.maximum(test_rate, 0))

            # rain series ทั้งหมดของจังหวัด (ใช้ทั้ง train+test สำหรับ oracle)
            full_rain     = prov["rain"].fillna(prov["rain"].median()).values
            train_rain    = full_rain[train_mask.values]
            exog_train    = build_rain_exog(train_rain)

            try:
                arima_base = ARIMA(train_log_rate, order=(1, 1, 1)).fit()
            except Exception:
                arima_base = None

            # ARIMAX lag 0-4: fit บน train ด้วย rain exog lag 0-4
            try:
                arimax_base = SARIMAX(
                    train_log_rate,
                    exog=exog_train,
                    order=(1, 1, 1),
                ).fit(disp=False)
            except Exception:
                arimax_base = None

            # ARIMAX lag 5-8: ใช้ rain ที่ผ่านมา 5-8 สัปดาห์ (ไม่ต้องการ oracle)
            exog_train_58 = build_rain_exog_lag58(train_rain)
            try:
                arimax_lag58_base = SARIMAX(
                    train_log_rate,
                    exog=exog_train_58,
                    order=(1, 1, 1),
                ).fit(disp=False)
            except Exception:
                arimax_lag58_base = None

            # SARIMA open-loop: fit ครั้งเดียว forecast ล่วงหน้าจาก train end
            last_train_ws  = prov.loc[train_mask, "ws"].iloc[-1]
            first_test_ws  = test_rows["ws"].iloc[0]
            offset = max(1, int((first_test_ws - last_train_ws).days // 7))
            n_steps_sarima = offset + len(test_rows) + HORIZON + 1
            try:
                fc_sarima = sarima_openloop_forecast(train_log_rate, n_steps_sarima)
                D_sarima_ol = forecast_to_d(fc_sarima, offset - 1, R_prev_arr)
                ok_s = ~np.isnan(D_sarima_ol)
                if ok_s.sum() > 0:
                    fold_maes_sarima.append(np.mean(np.abs(D_sarima_ol[ok_s] - D_actual[ok_s])))
            except Exception:
                pass

            D_arima_wf       = []
            D_arimax_wf      = []
            D_arimax_lag58_wf = []

            # หา index ของ test weeks ใน full_rain
            test_indices = np.where(test_mask.values)[0]

            for j in range(len(test_log_rate)):
                history       = np.concatenate([train_log_rate, test_log_rate[:j + 1]])
                R_prev        = R_prev_arr[j]
                global_idx    = test_indices[j]   # ตำแหน่งใน prov ทั้งหมด

                # ── ARIMA walk-forward ──
                if arima_base is not None:
                    try:
                        fc = arima_base.apply(history).forecast(steps=HORIZON)
                        R_next_pred = max(0.0, np.mean(np.expm1(fc)))
                        D_arima_wf.append(np.log1p(R_next_pred) - np.log1p(max(0.0, R_prev)))
                    except Exception:
                        D_arima_wf.append(np.nan)
                else:
                    D_arima_wf.append(np.nan)

                # ── ARIMAX oracle: ใช้ rain จริงสำหรับ 4 steps ข้างหน้า ──
                if arimax_base is not None:
                    try:
                        exog_history = build_rain_exog(full_rain[:global_idx + 1])
                        # oracle exog สำหรับ forecast 4 ขั้ว: rain ที่ t+1..t+4
                        future_rain = full_rain[global_idx + 1 : global_idx + 1 + HORIZON]
                        if len(future_rain) < HORIZON:
                            future_rain = np.pad(
                                future_rain, (0, HORIZON - len(future_rain)),
                                constant_values=float(np.nanmedian(full_rain))
                            )
                        # สร้าง exog matrix สำหรับ forecast:
                        # row h = [rain[t+h], rain[t+h-1], ..., rain[t+h-4]]
                        exog_fc = np.column_stack([
                            [full_rain[max(0, global_idx + h - k)] for h in range(1, HORIZON + 1)]
                            for k in range(5)
                        ])
                        fc_mx = arimax_base.apply(history, exog=exog_history).forecast(
                            steps=HORIZON, exog=exog_fc
                        )
                        R_next_pred = max(0.0, np.mean(np.expm1(fc_mx)))
                        D_arimax_wf.append(np.log1p(R_next_pred) - np.log1p(max(0.0, R_prev)))
                    except Exception:
                        D_arimax_wf.append(np.nan)
                else:
                    D_arimax_wf.append(np.nan)

                # ── ARIMAX lag 5-8: ไม่ต้องการ oracle (rain ทุกค่าเป็นอดีต) ──
                if arimax_lag58_base is not None:
                    try:
                        exog_history_58 = build_rain_exog_lag58(full_rain[:global_idx + 1])
                        exog_fc_58 = np.column_stack([
                            [full_rain[max(0, global_idx + h - lag)]
                             for h in range(1, HORIZON + 1)]
                            for lag in [5, 6, 7, 8]
                        ])
                        fc_lag58 = arimax_lag58_base.apply(
                            history, exog=exog_history_58
                        ).forecast(steps=HORIZON, exog=exog_fc_58)
                        R_next_pred = max(0.0, np.mean(np.expm1(fc_lag58)))
                        D_arimax_lag58_wf.append(
                            np.log1p(R_next_pred) - np.log1p(max(0.0, R_prev))
                        )
                    except Exception:
                        D_arimax_lag58_wf.append(np.nan)
                else:
                    D_arimax_lag58_wf.append(np.nan)

            D_arima_wf        = np.array(D_arima_wf)
            D_arimax_wf       = np.array(D_arimax_wf)
            D_arimax_lag58_wf = np.array(D_arimax_lag58_wf)

            ok_a = ~np.isnan(D_arima_wf)
            if ok_a.sum() > 0:
                fold_maes_arima.append(np.mean(np.abs(D_arima_wf[ok_a] - D_actual[ok_a])))

            ok_mx = ~np.isnan(D_arimax_wf)
            if ok_mx.sum() > 0:
                fold_maes_arimax.append(np.mean(np.abs(D_arimax_wf[ok_mx] - D_actual[ok_mx])))

            ok_mx58 = ~np.isnan(D_arimax_lag58_wf)
            if ok_mx58.sum() > 0:
                fold_maes_arimax_lag58.append(
                    np.mean(np.abs(D_arimax_lag58_wf[ok_mx58] - D_actual[ok_mx58]))
                )

        def _r(lst): return round(float(np.mean(lst)), 4) if lst else None
        results["Naive(D=0)"][fold_name]              = _r(fold_maes_naive)
        results["SeasonalNaive"][fold_name]            = _r(fold_maes_snaive)
        results["ARIMA(1,1,1)"][fold_name]             = _r(fold_maes_arima)
        results["ARIMAX+rain(oracle)"][fold_name]      = _r(fold_maes_arimax)
        results["ARIMAX+rain(lag5-8)"][fold_name]      = _r(fold_maes_arimax_lag58)
        results["SARIMA(1,1,1)(1,0,0,52)"][fold_name] = _r(fold_maes_sarima)

        def _fmt(v): return f"{v:.4f}" if v is not None else "—"
        arima_v   = results["ARIMA(1,1,1)"][fold_name]
        arimax_v  = results["ARIMAX+rain(oracle)"][fold_name]
        arimax58_v = results["ARIMAX+rain(lag5-8)"][fold_name]
        diff_txt = ""
        if arima_v and arimax_v:
            diff = arimax_v - arima_v
            diff_txt = f"  Δ(oracle-ARIMA)={diff:+.4f}"
        diff58_txt = ""
        if arima_v and arimax58_v:
            diff58 = arimax58_v - arima_v
            diff58_txt = f"  Δ(lag58-ARIMA)={diff58:+.4f}"
        print(f"    Naive={_fmt(results['Naive(D=0)'][fold_name])}  "
              f"ARIMA={_fmt(arima_v)}  "
              f"ARIMAX(oracle)={_fmt(arimax_v)}{diff_txt}  "
              f"ARIMAX(lag5-8)={_fmt(arimax58_v)}{diff58_txt}")

    return results


# ── รวมผลเข้า JSON ────────────────────────────────────────────────────────────
def merge_into_json(ts_results: dict):
    with open(OUTJSON) as f:
        data = json.load(f)

    # ลบ TS models เก่าออกก่อน (ถ้ามี) แล้วเพิ่มใหม่
    existing = [r for r in data["h4"]["sensitivity"]
                if r["model"] not in ts_results]

    for model_name, fold_maes in ts_results.items():
        for fold, mae in fold_maes.items():
            if mae is None:
                continue
            # ARIMAX เปรียบกับ ARIMA เป็น baseline (ไม่ใช่ Huber M0)
            if model_name in ("ARIMAX+rain(oracle)", "ARIMAX+rain(lag5-8)"):
                m0_ref = ts_results.get("ARIMA(1,1,1)", {}).get(fold)
            else:
                m0_ref = next(
                    (r["mae_m0"] for r in data["h4"]["results"]
                     if r["fold"] == fold and r["model"] == "HuberRegressor"),
                    None,
                )
            delta = round((m0_ref - mae) / m0_ref * 100, 4) if m0_ref else None
            existing.append({
                "fold": fold,
                "model": model_name,
                "mae_m0": m0_ref,
                "mae_m1": mae,
                "delta_pct": delta,
            })

    data["h4"]["sensitivity"] = existing

    # เพิ่ม ts_mae section สำหรับ chart แบบ standalone (MAE absolute)
    data["ts_mae"] = {
        model: {fold: mae for fold, mae in fd.items() if mae is not None}
        for model, fd in ts_results.items()
    }

    with open(OUTJSON, "w") as f:
        json.dump(data, f, ensure_ascii=False, default=str)

    print(f"\nบันทึก {OUTJSON}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    print("โหลดข้อมูล...")
    panel = load_panel()
    print(f"  {len(panel):,} แถว  {panel['pcode'].nunique()} จังหวัด\n")

    t0 = time.time()
    ts_results = run_all(panel)
    print(f"\nรวม {time.time()-t0:.1f}s")

    merge_into_json(ts_results)
