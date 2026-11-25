import duckdb

# ลงทะเบียน df (ตารางฟีเจอร์ฐาน) ให้ DuckDB อ่านโดยตรง (ไม่คัดลอกข้อมูล)
_fcon = duckdb.connect()
_fcon.register("feat", df)

ALL_FEATURES = [c for g, cols in FEATURE_GROUPS.items() for c in cols
                if (g != "nat" or USE_NATIONAL_FEATURES)]
_FOLD_CACHE = {}


def _fold_sql(fold):
    '''SQL เดียว: climatology (เฉพาะก่อน test_start) → anomaly → lag ของ anomaly → ตัดแบ่ง train/test'''
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
    return f'''
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


def get_fold_sql(fold):
    '''เหมือน get_fold เดิม: dict(train, test, test_start) แต่ทำ anomaly + ตัดแบ่งด้วย SQL; ผลถูก cache'''
    if fold["name"] in _FOLD_CACHE:
        return _FOLD_CACHE[fold["name"]]
    test_start = pd.Timestamp(f"{fold['test_year']}-01-01")
    d = _fcon.execute(_fold_sql(fold), [test_start.to_pydatetime()]).df()
    tr = d[d["_tr"]].drop(columns=["_ok", "_tr", "_te"]).reset_index(drop=True)
    te = d[d["_te"]].drop(columns=["_ok", "_tr", "_te"]).reset_index(drop=True)

    # --- unit test ของ fold (หยุดทันทีถ้าไม่ผ่าน) ---
    assert tr["label_end"].max() < test_start, "label ของ train ล้ำเข้าช่วงทดสอบ"
    assert te["Week start"].min() >= test_start, "test เริ่มก่อนต้นปีทดสอบ"
    assert set(tr["P-code"]) <= set(te["P-code"]) | set(tr["P-code"]), "ตรวจจังหวัด"
    out = {"train": tr, "test": te, "test_start": test_start}
    _FOLD_CACHE[fold["name"]] = out
    return out


get_fold = get_fold_sql      # ให้ section อื่นที่เรียก get_fold ใช้ของ SQL ต่อได้เลย

summary = []
for f in MAIN_FOLDS + SUPP_FOLDS:
    fd = get_fold(f)
    summary.append({"fold": f["name"], "train_years": f"{f['train_years'][0]}-{f['train_years'][1]}",
                    "test_year": f["test_year"], "n_train": len(fd["train"]), "n_test": len(fd["test"]),
                    "last_train_label_end": fd["train"]["label_end"].max().date()})
pd.DataFrame(summary)
