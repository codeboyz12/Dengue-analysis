"""dashboard/app.py — Streamlit dashboard สำหรับ dengue pipeline

รัน: streamlit run dashboard/app.py
อ่าน DuckDB read_only=True (ไม่กระทบ pipeline writer)
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "pipeline.duckdb"
RUNS_JSONL = ROOT / "reports" / "runs.jsonl"
MAIN_FOLDS = ["2019", "2022", "2023", "2024", "2025"]

st.set_page_config(page_title="Dengue Pipeline Dashboard", layout="wide")
st.title("Dengue × Weather Pipeline — Dashboard")


@st.cache_resource
def get_con():
    return duckdb.connect(str(DB_PATH), read_only=True)


con = get_con()
tab1, tab2, tab3 = st.tabs(["🔧 Pipeline Health", "📊 EDA", "🔬 H1–H4 Tracker"])

# ==========================================================================
# Tab 1 — Pipeline Health
# ==========================================================================
with tab1:
    st.header("สุขภาพ Pipeline")

    if RUNS_JSONL.exists():
        runs_raw = [json.loads(l) for l in RUNS_JSONL.read_text().splitlines() if l.strip()]
        runs_df = pd.DataFrame(runs_raw)
        runs_df["duration_s"] = (
            pd.to_datetime(runs_df["ended"]) - pd.to_datetime(runs_df["started"])
        ).dt.total_seconds().round(1)
        pivot = (runs_df.pivot_table(index="run_id", columns="stage",
                                     values="status", aggfunc="first").reset_index())
        st.subheader("Run History")
        st.dataframe(pivot, use_container_width=True)
        st.subheader("ระยะเวลาต่อ Stage (วินาที)")
        st.dataframe(runs_df[runs_df["status"] == "success"][["run_id", "stage", "duration_s"]],
                     use_container_width=True)
    else:
        st.info("ยังไม่มี reports/runs.jsonl")

    try:
        audit = con.execute("""
            SELECT run_id, stage, rule_id, severity, outcome, measured
            FROM ops.audit_log ORDER BY checked_at DESC
        """).df()
        c1, c2 = st.columns(2)
        c1.metric("✅ Pass", int((audit["outcome"] == "pass").sum()))
        c2.metric("❌ Fail/Skip", int((audit["outcome"] != "pass").sum()))
        st.subheader("Validation Rules")
        st.dataframe(audit, use_container_width=True)
    except Exception as e:
        st.warning(f"audit_log: {e}")

    try:
        dv = con.execute("""
            SELECT data_version, written_at, pipeline_version,
                   JSON_EXTRACT_STRING(metadata, '$.rows.trends_weather') AS rows_merged
            FROM ops.data_version ORDER BY written_at DESC
        """).df()
        st.subheader("Data Versions")
        st.dataframe(dv, use_container_width=True)
    except Exception as e:
        st.warning(f"data_version: {e}")

# ==========================================================================
# Tab 2 — EDA
# ==========================================================================
with tab2:
    st.header("EDA พื้นฐาน")

    @st.cache_data
    def load_eda():
        c = get_con()
        return c.execute("""
            SELECT "P-code", "Province", "Health region",
                   "Year", "Week start", week_days,
                   CAST("Dengue total" AS DOUBLE) AS dengue,
                   CAST("Population" AS DOUBLE) AS pop,
                   rain_wk_mm, tmax_mean_c, tmin_mean_c,
                   dayofyear("Week start") AS doy
            FROM mart.trends_weather_latest
            WHERE week_days = 7
        """).df()

    @st.cache_data
    def load_features():
        c = get_con()
        return c.execute("""
            SELECT "P-code", "Province", "Year", "Week start",
                   CAST("Dengue total" AS DOUBLE) / CAST("Population" AS DOUBLE) * 1e5 AS rate,
                   rain_wk_mm, tmax_mean_c, tmin_mean_c
            FROM mart.trends_weather_latest
            WHERE week_days = 7
        """).df()

    try:
        df = load_eda()

        # 1. อัตรารายปี
        st.subheader("① อัตราผู้ป่วยไข้เลือดออกรายปี (ต่อ 1 แสนประชากร)")
        annual = (df.groupby("Year")
                  .apply(lambda g: g["dengue"].sum() / g["pop"].mean() * 1e5, include_groups=False)
                  .reset_index(name="rate"))
        chart_annual = (alt.Chart(annual).mark_line(point=True)
                        .encode(x=alt.X("Year:O", title="ปี"),
                                y=alt.Y("rate:Q", title="อัตรา/1แสน"),
                                tooltip=["Year", alt.Tooltip("rate:Q", format=".1f")])
                        .properties(height=250))
        st.altair_chart(chart_annual, use_container_width=True)

        # 2. Seasonal pattern
        st.subheader("② ฤดูกาล — อัตราเฉลี่ยต่อสัปดาห์ในปี (เฉลี่ยข้ามจังหวัดและปี)")
        df["week_no"] = df["Week start"].dt.isocalendar().week.astype(int)
        seasonal = (df.groupby("week_no")
                    .apply(lambda g: g["dengue"].sum() / g["pop"].mean() * 1e5, include_groups=False)
                    .reset_index(name="rate"))
        chart_seas = (alt.Chart(seasonal).mark_area(opacity=0.5, color="#e45756")
                      .encode(x=alt.X("week_no:Q", title="สัปดาห์ในปี"),
                              y=alt.Y("rate:Q", title="อัตรา/1แสน"),
                              tooltip=["week_no", alt.Tooltip("rate:Q", format=".2f")])
                      .properties(height=220))
        st.altair_chart(chart_seas, use_container_width=True)

        # 3. Distribution of D (target) per fold year
        st.subheader("③ การกระจาย Target D = log1p(R_next) − log1p(R_prev) ต่อ fold year")

        @st.cache_data
        def load_D():
            c = get_con()
            # คำนวณ D จาก mart (ใช้ build_features logic แบบ approximate)
            return c.execute("""
                WITH src AS (
                    SELECT "P-code", "Year", "Week start",
                           CAST("Population" AS DOUBLE) AS pop,
                           CAST("Dengue total" AS DOUBLE) AS c,
                           week_days,
                           lag("Week start", 3) OVER w AS ws_l3,
                           lag("Week start", 7) OVER w AS ws_l7,
                           lead("Week start", 4) OVER w AS ws_f4,
                           CASE WHEN count(CASE WHEN week_days=7 THEN c END)
                                    OVER (PARTITION BY "P-code" ORDER BY "Week start"
                                          ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                                THEN sum(CASE WHEN week_days=7 THEN c END)
                                    OVER (PARTITION BY "P-code" ORDER BY "Week start"
                                          ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) END AS s4
                    FROM mart.trends_weather_latest
                    WINDOW w AS (PARTITION BY "P-code" ORDER BY "Week start")
                ),
                s2 AS (
                    SELECT *,
                           CASE WHEN date_diff('day', ws_l3, "Week start") = 21 THEN s4 END AS s4_b,
                           CASE WHEN date_diff('day', ws_l7, "Week start") = 49
                                THEN lag(s4, 4) OVER (PARTITION BY "P-code" ORDER BY "Week start")
                           END AS s4_b7,
                           CASE WHEN date_diff('day', "Week start", ws_f4) = 28
                                THEN lead(s4, 4) OVER (PARTITION BY "P-code" ORDER BY "Week start")
                           END AS s4_f
                    FROM src
                )
                SELECT "Year",
                       ln(1 + s4_f / pop * 1e5) - ln(1 + s4_b / pop * 1e5) AS D
                FROM s2
                WHERE s4_b IS NOT NULL AND s4_f IS NOT NULL AND week_days = 7
            """).df()

        d_df = load_D()
        d_df["fold"] = d_df["Year"].astype(str)
        d_df["fold_type"] = d_df["fold"].apply(
            lambda y: "main" if y in MAIN_FOLDS else ("supp" if y in ["2020","2021"] else "train"))

        chart_D = (alt.Chart(d_df[d_df["D"].notna()])
                   .transform_density("D", as_=["D", "density"], groupby=["fold"])
                   .mark_area(opacity=0.4)
                   .encode(x=alt.X("D:Q", title="D = log1p(R_next) − log1p(R_prev)"),
                           y=alt.Y("density:Q", title=""),
                           color=alt.Color("fold:N", title="ปี"),
                           tooltip=["fold"])
                   .properties(height=250))
        st.altair_chart(chart_D, use_container_width=True)

        # 4. National vs Province correlation
        st.subheader("④ National vs Province — 66% ของ variance เป็น national effect")
        feat = load_features()
        nat = feat.groupby(["Year", "Week start"])["rate"].mean().reset_index(name="nat_rate")
        prov_nat = feat.merge(nat, on=["Year", "Week start"])
        sample = prov_nat.sample(min(3000, len(prov_nat)), random_state=42)
        chart_scatter = (alt.Chart(sample)
                         .mark_point(opacity=0.2, size=15)
                         .encode(x=alt.X("nat_rate:Q", title="อัตราระดับประเทศ/1แสน"),
                                 y=alt.Y("rate:Q", title="อัตราระดับจังหวัด/1แสน"),
                                 tooltip=["Year", "Province"])
                         .properties(height=280))
        # regression line
        reg = prov_nat[["nat_rate", "rate"]].dropna()
        m, b = np.polyfit(reg["nat_rate"], reg["rate"], 1)
        x_range = pd.DataFrame({"nat_rate": [reg["nat_rate"].min(), reg["nat_rate"].max()]})
        x_range["rate"] = m * x_range["nat_rate"] + b
        chart_reg = (alt.Chart(x_range).mark_line(color="red", strokeDash=[4,2])
                     .encode(x="nat_rate:Q", y="rate:Q"))
        st.altair_chart((chart_scatter + chart_reg), use_container_width=True)
        r2 = np.corrcoef(reg["nat_rate"], reg["rate"])[0,1]**2
        st.caption(f"R² = {r2:.2f} — ความสัมพันธ์ระหว่างอัตราจังหวัดกับอัตราประเทศ (จุดสุ่ม 3,000 จาก {len(prov_nat):,})")

    except Exception as e:
        st.error(f"EDA error: {e}")
        import traceback; st.code(traceback.format_exc())

# ==========================================================================
# Tab 3 — H1–H4 Tracker
# ==========================================================================
with tab3:
    st.header("H1–H4 Hypothesis Tracker")

    # ------------------------------------------------------------------
    # H4 — ΔMAE bar chart with CI and 5% threshold
    # ------------------------------------------------------------------
    st.subheader("H4 — ΔMAE (%) M1 vs M0 · HuberRegressor · main folds")
    try:
        res = con.execute("""
            SELECT fold, round(delta_mae_pct*100,2) AS delta_pct,
                   round(ci_lower*100,2) AS ci_lo,
                   round(ci_upper*100,2) AS ci_hi
            FROM mart.results
            WHERE model='HuberRegressor' AND feature_set='M1'
              AND fold IN ('2019','2022','2023','2024','2025')
            ORDER BY created_at DESC, fold
            LIMIT 5
        """).df()

        res["pass"] = (res["delta_pct"] >= 5) & (res["ci_lo"] > 0)
        res["color"] = res["pass"].map({True: "✅ ผ่าน", False: "❌ ไม่ผ่าน"})

        bars = (alt.Chart(res)
                .mark_bar()
                .encode(
                    x=alt.X("fold:O", title="Fold (ปีทดสอบ)", sort=MAIN_FOLDS),
                    y=alt.Y("delta_pct:Q", title="ΔMAE (%) — บวก = M1 ดีกว่า"),
                    color=alt.Color("color:N",
                                    scale=alt.Scale(domain=["✅ ผ่าน", "❌ ไม่ผ่าน"],
                                                    range=["#2ecc71", "#e74c3c"]),
                                    legend=alt.Legend(title="")),
                    tooltip=["fold",
                             alt.Tooltip("delta_pct:Q", title="ΔMAE%", format=".2f"),
                             alt.Tooltip("ci_lo:Q", title="CI lower%", format=".2f"),
                             alt.Tooltip("ci_hi:Q", title="CI upper%", format=".2f")]
                ))

        ci_bars = (alt.Chart(res)
                   .mark_errorbar(ticks=True)
                   .encode(
                       x=alt.X("fold:O", sort=MAIN_FOLDS),
                       y=alt.Y("ci_lo:Q", title=""),
                       y2=alt.Y2("ci_hi:Q"),
                   ))

        threshold = (alt.Chart(pd.DataFrame({"y": [5]}))
                     .mark_rule(color="green", strokeDash=[6, 3], strokeWidth=2)
                     .encode(y="y:Q"))

        zero = (alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(color="gray", strokeDash=[2, 2])
                .encode(y="y:Q"))

        h4_chart = (bars + ci_bars + threshold + zero).properties(height=300)
        st.altair_chart(h4_chart, use_container_width=True)
        st.caption("เส้นสีเขียว = เกณฑ์ผ่าน +5% · error bar = CI 95% block-week bootstrap")

        n_pass = int(res["pass"].sum())
        st.info(f"**H4: {'❌ ไม่ผ่าน' if n_pass < 4 else '✅ ผ่าน'}** — M1 ดีกว่า M0 ≥ 5% ใน {n_pass}/5 fold (เกณฑ์ ≥ 4/5)")

    except Exception as e:
        st.warning(f"H4: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Model Comparison — MAE ทุกโมเดล ต่อ fold
    # ------------------------------------------------------------------
    st.subheader("เปรียบเทียบ MAE ทุกโมเดล · main folds")
    try:
        all_res = con.execute("""
            WITH latest AS (
                SELECT fold, model, feature_set, mae,
                       ROW_NUMBER() OVER (
                           PARTITION BY fold, model, feature_set
                           ORDER BY created_at DESC
                       ) AS rn
                FROM mart.results
                WHERE fold IN ('2019','2022','2023','2024','2025')
            )
            SELECT fold, model, feature_set, round(mae, 4) AS mae
            FROM latest WHERE rn = 1
            ORDER BY fold, model, feature_set
        """).df()

        fs_sel = st.radio("Feature set", ["M1", "M0"], horizontal=True, key="fs_cmp")
        cmp = all_res[all_res["feature_set"] == fs_sel].copy()

        if cmp.empty:
            st.info(f"ยังไม่มีข้อมูล {fs_sel} ใน mart.results — รัน train ก่อน")
        else:
            # pivot เพื่อแสดงตาราง
            tbl = cmp.pivot(index="model", columns="fold", values="mae")
            tbl.columns.name = None
            tbl.index.name = "โมเดล"
            # เพิ่มคอลัมน์ mean
            tbl["mean"] = tbl.mean(axis=1).round(4)
            tbl = tbl.sort_values("mean")
            st.dataframe(tbl.style.highlight_min(axis=0, color="#c8f7c5")
                            .format("{:.4f}"),
                         use_container_width=True)

            # Line chart
            chart_cmp = (
                alt.Chart(cmp)
                .mark_line(point=True)
                .encode(
                    x=alt.X("fold:O", sort=MAIN_FOLDS, title="Fold (ปีทดสอบ)"),
                    y=alt.Y("mae:Q", title="MAE", scale=alt.Scale(zero=False)),
                    color=alt.Color("model:N", title="โมเดล"),
                    strokeDash=alt.StrokeDash("model:N"),
                    tooltip=["fold", "model",
                             alt.Tooltip("mae:Q", format=".4f", title="MAE")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_cmp, use_container_width=True)
            st.caption(f"Feature set: {fs_sel} · สีเขียวในตาราง = MAE ต่ำสุดต่อ fold")

    except Exception as e:
        st.warning(f"Model comparison: {e}")
        import traceback; st.code(traceback.format_exc())

    st.divider()

    # ------------------------------------------------------------------
    # H1/H2 — ρ profile per var, one chart each
    # ------------------------------------------------------------------
    st.subheader("H1/H2 — Partial Spearman |ρ| ต่อ lag · เส้นแดง = เกณฑ์ 0.10 · แถบเทา = placebo P95")

    try:
        h_res = con.execute("""
            SELECT fold, var, lag, median_abs_rho, median_rho
            FROM mart.h_results
            WHERE analysis = 'H1H2'
            ORDER BY created_at DESC, var, lag, fold
        """).df()

        # deduplicate: เอา created_at ล่าสุด (query แล้วเรียง DESC แต่ limit ไม่ได้ทำ)
        # ใช้ groupby เอา median ข้าม fold
        h_agg = (h_res.groupby(["var", "lag"])
                 .agg(median_abs_rho=("median_abs_rho", "mean"),
                      median_rho=("median_rho", "mean"))
                 .reset_index())

        pb = con.execute("""
            SELECT var, lag, avg(placebo_p95) AS placebo_p95
            FROM mart.h_placebo
            GROUP BY var, lag
        """).df()

        h_full = h_agg.merge(pb, on=["var", "lag"], how="left")

        # ρ per fold (สำหรับเส้นย่อย)
        h_fold = h_res[h_res["fold"].isin(MAIN_FOLDS)].copy()

        threshold_rule = (alt.Chart(pd.DataFrame({"y": [0.10]}))
                          .mark_rule(color="red", strokeDash=[5, 3], strokeWidth=1.5)
                          .encode(y="y:Q"))

        var_labels = {"rain": "H1 · ฝน (rain)", "tmax": "H2 · อุณหภูมิสูงสุด (tmax)", "tmin": "H2 · อุณหภูมิต่ำสุด (tmin)"}

        for var in ["rain", "tmax", "tmin"]:
            st.write(f"**{var_labels.get(var, var)}**")
            sub = h_full[h_full["var"] == var].sort_values("lag")
            sub_fold = h_fold[h_fold["var"] == var]

            # แถบ placebo P95
            band = (alt.Chart(sub)
                    .mark_area(opacity=0.15, color="gray")
                    .encode(x=alt.X("lag:Q", title="Lag (สัปดาห์)", scale=alt.Scale(domain=[-0.3, 4.3])),
                            y=alt.Y("y_zero:Q"),
                            y2=alt.Y2("placebo_p95:Q"))
                    .transform_calculate(y_zero="0"))

            # เส้นย่อยต่อ fold
            fold_lines = (alt.Chart(sub_fold)
                          .mark_line(opacity=0.35, strokeDash=[3, 2])
                          .encode(x=alt.X("lag:Q"),
                                  y=alt.Y("median_abs_rho:Q", title="|ρ| median ข้ามจังหวัด"),
                                  color=alt.Color("fold:N", legend=None)))

            # เส้นหลัก (avg ข้าม fold)
            main_line = (alt.Chart(sub)
                         .mark_line(point=True, strokeWidth=2.5)
                         .encode(x=alt.X("lag:Q"),
                                 y=alt.Y("median_abs_rho:Q"),
                                 tooltip=["lag",
                                          alt.Tooltip("median_abs_rho:Q", format=".4f", title="|ρ| avg"),
                                          alt.Tooltip("placebo_p95:Q", format=".4f", title="placebo P95")]))

            chart = (band + fold_lines + main_line + threshold_rule).properties(height=220)
            st.altair_chart(chart, use_container_width=True)

        st.caption("เส้นทึบ = avg |ρ| ข้าม 5 main fold · เส้นบาง = แต่ละ fold · แถบเทา = placebo P95 · เส้นแดง = เกณฑ์ 0.10")

    except Exception as e:
        st.warning(f"H1/H2: {e}")
        import traceback; st.code(traceback.format_exc())

    st.divider()

    # ------------------------------------------------------------------
    # Decision table
    # ------------------------------------------------------------------
    st.subheader("สรุปผลการตัดสิน H1–H4")
    try:
        dec_row = con.execute("""
            SELECT decisions_json FROM mart.h_decisions ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        rows = []
        if dec_row:
            d = json.loads(dec_row[0])
            for key, val in d.items():
                if key == "H3":
                    rows.append({"สมมติฐาน": "H3 (lag profile)", "ตัวแปร": "—",
                                 "ผล": "— ตัดสินไม่ได้" if not val.get("eligible") else
                                       ("✅ ผ่าน" if all(v.get("passed") for k,v in val.items() if k!="eligible") else "❌ ไม่ผ่าน"),
                                 "หมายเหตุ": "H1/H2 ไม่ผ่าน" if not val.get("eligible") else ""})
                else:
                    h = "H1" if key == "rain" else "H2"
                    rows.append({"สมมติฐาน": f"{h} ({key})", "ตัวแปร": key,
                                 "ผล": "✅ ผ่าน" if val.get("passed") else "❌ ไม่ผ่าน",
                                 "หมายเหตุ": f"lag ผ่าน: {val.get('pass_lags', [])}"})

        # H4
        h4_pass = n_pass >= 4 if "n_pass" in dir() else False
        rows.append({"สมมติฐาน": "H4 (MAE ลด ≥5%)", "ตัวแปร": "M1 vs M0",
                     "ผล": "✅ ผ่าน" if h4_pass else "❌ ไม่ผ่าน",
                     "หมายเหตุ": f"{n_pass if 'n_pass' in dir() else '?'}/5 fold ผ่าน (เกณฑ์ ≥4/5)"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("**ข้อสรุป:** อากาศไม่เพิ่มประโยชน์ในการพยากรณ์ไข้เลือดออกในข้อมูลชุดนี้ (H1–H4 ไม่ผ่านทั้งหมด) ซึ่งเป็น outcome ที่ proposal ระบุว่ายอมรับได้")

    except Exception as e:
        st.warning(f"decision table: {e}")
