import math
import duckdb
import pandas as pd


def build_features_sql(source, con=None):
    '''build_features เวอร์ชัน DuckDB: คืน DataFrame คอลัมน์/ลำดับ/ค่าเท่ากับ build_features(raw)
    source = ชื่อ table/view ใน DuckDB (เช่น "mart.trends_weather_latest") หรือ DataFrame (raw)
    con    = duckdb connection (ไม่ใส่ = ใช้ in-memory ชั่วคราว)
    ใช้ค่าคงที่จากเซลล์ตั้งค่า: WX_BASE, MAX_WX_LAG, FOURIER_K, TARGET
    '''
    own = con is None
    if own:
        con = duckdb.connect()
    try:
        if isinstance(source, pd.DataFrame):
            con.register("_bf_src_df", source)
            src = "_bf_src_df"
        else:
            src = source
        cols = con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns
        drop = [c for c in ("data_version", "_ord") if c in cols]
        base = f"SELECT * EXCLUDE ({', '.join(drop)}) FROM {src}" if drop else f"SELECT * FROM {src}"

        # ---- ตรวจเหมือน _national_block: ทุกสัปดาห์ต้องมีจังหวัดครบ ----
        n_prov, bad = con.execute(f'''
            WITH s AS ({base}), t AS (SELECT count(DISTINCT "P-code") AS n FROM s)
            SELECT (SELECT n FROM t),
                   (SELECT count(*) FROM (SELECT "Week start" FROM s GROUP BY 1
                                          HAVING count(DISTINCT "P-code") <> (SELECT n FROM t)))''').fetchone()
        assert bad == 0, "บางสัปดาห์มีจังหวัดไม่ครบ"
        regions = [r[0] for r in con.execute(
            f'SELECT DISTINCT CAST("Health region" AS INTEGER) FROM ({base}) ORDER BY 1').fetchall()]

        W = 'OVER (PARTITION BY "P-code" ORDER BY "Week start")'
        W4 = ('OVER (PARTITION BY "P-code" ORDER BY "Week start" '
              'ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)')
        dd = lambda a, b: f"date_diff('day', CAST({a} AS DATE), CAST({b} AS DATE))"
        wx = list(WX_BASE.items())

        # s1: ค่า c / w รายสัปดาห์เต็ม, ผลรวมหน้าต่าง 4 สัปดาห์, วันที่ของแถวข้างเคียง
        s1 = ['*',
              'CAST("Population" AS DOUBLE) AS pop',
              'CASE WHEN week_days = 7 THEN CAST("Dengue total" AS DOUBLE) END AS c']
        s1 += [f'CASE WHEN week_days = 7 THEN CAST("{col}" AS DOUBLE) END AS w_{key}' for key, col in wx]
        _ws_needed = sorted(set(range(1, MAX_WX_LAG + 1)) | {3, 7})
        s1 += [f'lag("Week start", {k}) {W} AS ws_l{k}' for k in _ws_needed]
        s1 += [f'lead("Week start", 4) {W} AS ws_f4']
        s1 += [f'CASE WHEN count(CASE WHEN week_days = 7 THEN "Dengue total" END) {W4} = 4 '
               f'THEN sum(CASE WHEN week_days = 7 THEN CAST("Dengue total" AS DOUBLE) END) {W4} END AS s4']

        # s2: lag/lead ของ c, w, s4 พร้อมเงื่อนไขความต่อเนื่องของวันที่
        s2 = ['*']
        s2 += [f'CASE WHEN {dd(f"ws_l{k}", chr(34)+"Week start"+chr(34))} = {7*k} THEN lag(c, {k}) {W} END AS c_l{k}'
               for k in (1, 2, 3)]
        for key, _ in wx:
            s2 += [f'CASE WHEN {dd(f"ws_l{k}", chr(34)+"Week start"+chr(34))} = {7*k} '
                   f'THEN lag(w_{key}, {k}) {W} END AS w_{key}_l{k}' for k in range(1, MAX_WX_LAG + 1)]
        s2 += [f'CASE WHEN {dd("ws_l3", chr(34)+"Week start"+chr(34))} = 21 THEN s4 END AS s4_b3',
               f'CASE WHEN {dd("ws_l7", chr(34)+"Week start"+chr(34))} = 49 THEN lag(s4, 4) {W} END AS s4_b7',
               f'CASE WHEN {dd(chr(34)+"Week start"+chr(34), "ws_f4")} = 28 THEN lead(s4, 4) {W} END AS s4_f4',
               f'lead("Week end", 4) {W} AS label_end_']

        # s3: อัตรา R และ target
        s3 = ['*', 's4_b3 / pop * 1e5 AS R_prev', 's4_b7 / pop * 1e5 AS R_prev2',
              's4_f4 / pop * 1e5 AS R_next']

        out = ['p.*', f'ln(1 + R_next) - ln(1 + R_prev) AS "{TARGET}"',
               'label_end_ AS label_end', 'R_prev AS diag_R_prev', 'R_next AS diag_R_next',
               's4_b3 + s4_f4 AS diag_cases_8w',
               'ln(1 + c / pop * 1e5) AS hist_lograte_lag0']
        out += [f'ln(1 + c_l{k} / pop * 1e5) AS hist_lograte_lag{k}' for k in (1, 2, 3)]
        out += ['ln(1 + R_prev) AS hist_logR_prev', 'ln(1 + R_prev2) AS hist_logR_prev2',
                'ln(1 + R_prev) - ln(1 + R_prev2) AS hist_growth_prev']
        for key, _ in wx:
            out += [f'w_{key} AS wk_{key}', f'w_{key} AS wx_{key}_lag0']
            out += [f'w_{key}_l{k} AS wx_{key}_lag{k}' for k in range(1, MAX_WX_LAG + 1)]
        out += ['n.nat_lograte_lag0', 'n.nat_logR_prev', 'n.nat_growth_prev']
        ang = 'pi() * 2 * dayofyear(p."Week start") / 365.25'
        for k in range(1, FOURIER_K + 1):
            out += [f'sin({k} * {ang}) AS seas_sin{k}', f'cos({k} * {ang}) AS seas_cos{k}']
        out += [f'CAST(CAST(p."Health region" AS INTEGER) = {r} AS BIGINT) AS reg_{r}' for r in regions]

        # แยก p.* ออกจากคอลัมน์ช่วย (pop, c, w_*, ws_*, s4*, ...) ให้เหลือเฉพาะคอลัมน์เดิมของ raw
        helper = (['pop', 'c', 's4', 'ws_f4', 'label_end_', 's4_b3', 's4_b7', 's4_f4', 'R_prev', 'R_prev2', 'R_next']
                  + [f'ws_l{k}' for k in _ws_needed] + [f'c_l{k}' for k in (1, 2, 3)]
                  + [f'w_{key}' for key, _ in wx]
                  + [f'w_{key}_l{k}' for key, _ in wx for k in range(1, MAX_WX_LAG + 1)])
        out[0] = f'p.* EXCLUDE ({", ".join(helper)})'

        sql = f'''
        WITH src AS ({base}),
        s1 AS (SELECT {", ".join(s1)} FROM src),
        s2 AS (SELECT {", ".join(s2)} FROM s1),
        p  AS (SELECT {", ".join(s3)} FROM s2),
        n1 AS (SELECT "Week start" AS ws, sum(CAST("Dengue total" AS DOUBLE)) AS cases,
                      sum(CAST("Population" AS DOUBLE)) AS pop, min(week_days) AS wd
               FROM src GROUP BY 1),
        n2 AS (SELECT ws, pop, CASE WHEN wd = 7 THEN cases END AS c,
                      lag(ws, 3) OVER o AS ws_l3, lag(ws, 7) OVER o AS ws_l7 FROM n1
               WINDOW o AS (ORDER BY ws)),
        n3 AS (SELECT *, CASE WHEN count(c) OVER (ORDER BY ws ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                              THEN sum(c) OVER (ORDER BY ws ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) END AS s4
               FROM n2),
        n4 AS (SELECT ws,
                      CASE WHEN {dd("ws_l3", "ws")} = 21 THEN s4 END / pop * 1e5 AS R_prev,
                      CASE WHEN {dd("ws_l7", "ws")} = 49 THEN lag(s4, 4) OVER (ORDER BY ws) END / pop * 1e5 AS R_prev2,
                      c, pop FROM n3),
        n AS (SELECT ws AS "Week start", ln(1 + c / pop * 1e5) AS nat_lograte_lag0,
                     ln(1 + R_prev) AS nat_logR_prev, ln(1 + R_prev) - ln(1 + R_prev2) AS nat_growth_prev FROM n4)
        SELECT {", ".join(out)}
        FROM p LEFT JOIN n ON p."Week start" = n."Week start"
        ORDER BY p."P-code", p."Week start"
        '''
        return con.execute(sql).df()
    finally:
        if own:
            con.close()
