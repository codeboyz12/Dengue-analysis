# CLAUDE.md — dengue_pipeline

โปรเจกต์กลุ่มวิชา CSS 342 (Data Science and Data Engineering, KMUTT, 2569) เจ้าของ repo นี้คือ Nut ทำฝั่ง **data pipeline** คนเดียว ทีมอีก 4 คนทำฝั่ง data science จาก CSV ที่ pipeline ส่งออก

## วิธีทำงานกับ Nut (สำคัญ)

- **ถามยืนยันก่อนเขียนโค้ด** เสนอแผน/ออกแบบก่อน เขียนโค้ดเมื่อ Nut อนุมัติ (Nut ระบุเป็นความชอบถาวร)
- สอนแบบ concept-first: อธิบายภาพรวม/สัญชาตญาณก่อนรายละเอียดเชิงสูตร
- Nut จะโต้แย้งเมื่อเหตุผลไม่ตรงหรือไม่ชัด ตอบให้แม่นและบอกข้อจำกัดของตัวเลข อย่าอ้างผลที่ยังไม่ได้ตรวจ
- ตอบเป็นภาษาไทย (ศัพท์เทคนิคเป็นอังกฤษได้) README และ comment ในโค้ดเป็นภาษาไทย
- ตัวเลขเวลา/ประสิทธิภาพ: บอกเสมอว่าวัดบนเครื่องไหน ตัวเลขจาก sandbox ใช้เทียบอัตราส่วนเท่านั้น
- ก่อนอ้างว่า "ผ่าน/ตรงกัน" ให้รันจริงบนข้อมูลจริง (`trends_weather_merged.csv`) ไม่ใช่แค่ข้อมูลสังเคราะห์ในเทส

## ภาพรวมโปรเจกต์

- **คำถามวิจัย:** สภาพอากาศ (ฝน, tmax, tmin) ช่วยพยากรณ์การเติบโตของไข้เลือดออกรายจังหวัดล่วงหน้า 4 สัปดาห์ได้หรือไม่ (ผลที่ proposal ยอมรับได้: "อากาศไม่เพิ่มประโยชน์ในข้อมูลนี้")
- **ข้อมูล:** TRENDS 2016-2025 (ไข้เลือดออก/ชิคุนกุนยา/มือเท้าปาก รายสัปดาห์ 77 จังหวัด, TRENDS.xlsx 2 ชีท) + อากาศรายวัน ERA5 ผ่าน Open-Meteo ที่จุด centroid จังหวัด (283,822 แถว = 77 × 3,686 วัน; สคริปต์ `weather_fetch.py`)
- **หน่วยวิเคราะห์:** จังหวัด-สัปดาห์ = 40,579 แถว (77 จังหวัด × 527 สัปดาห์; 13 สัปดาห์สั้นช่วงรอยต่อปี ต้องใช้ `week_days == 7`)
- **Target D:** `log1p(R_next) − log1p(R_prev)` เมื่อ R = ผู้ป่วยไข้เลือดออกรวมในสัปดาห์เต็มต่อประชากร ×1e5; R_prev = สัปดาห์ t−3..t, R_next = t+1..t+4; H = 4 สัปดาห์; มีตรวจวันที่ต่อเนื่อง (21/28/49 วัน)
  - ค่าอ้างอิง: D ไม่ null 35,266 แถว; มีทั้ง D และ `hist_growth_prev` 32,802 แถว (ต้องเท่านี้เสมอ)
- **Folds:** main = test 2019, 2022, 2023, 2024, 2025 (expanding train, gap 4 สัปดาห์ผ่าน `label_end < test_start`); supplementary = 2020, 2021; seed = 342
- **สมมติฐาน (ดู `docs/roadmap_h1_h4.md`):** H1 ฝน / H2 อุณหภูมิ (partial Spearman ภายในจังหวัด, anomaly lag 0-4, |ρ| ≥ 0.10 และเครื่องหมายเดียวกัน ≥ 4/5 fold) · H3 สัญญาณอยู่ที่ lag ใช้งานได้จริง (ขยายถึง 8) · **H4 (หลัก)** M1 (M0 + อากาศ lag 0-4 + anomaly) ลด MAE ของ D ≥ 5% เทียบ M0 (ประวัติ + ฤดูกาล), ดีกว่า ≥ 4/5 fold, CI ไม่คร่อมศูนย์
- **หลักฐานเบื้องต้นจาก EDA:** อากาศอย่างเดียวลด MAE ≈ 1.4-2% (ต่ำกว่า 5%); partial ρ ที่ lag 4 ≈ 0; 66% ของความแปรปรวนเป็นผลร่วมทั้งประเทศรายปี; correlation ข้ามจังหวัดแทบไม่ลดตามระยะทาง; 2021 เกือบไม่มีการระบาด 2023 ใหญ่สุด

## โครงสร้าง repo

```
config/pipeline.toml        ค่าที่เปลี่ยนได้ทั้งหมด ([load] sink="duckdb", db_file, [export] contract)
config/contract_v2.toml     สัญญา export: 44 คอลัมน์, คีย์ P-code + Week start
src/dengue_pipeline/
  cli.py                    run | status | export | inspect-trends | migrate-sink
  pipeline.py, runstate.py  ตัวควบคุมขั้นตอนและสถานะต่อ run
  stages/                   ingest → validate_source → transform → validate_merged → load → export
  validation/               checks_source / checks_merged / checks_export (rule id เช่น EXP-01..03, LKG-02, MRG-05)
  sink_duckdb.py, sinks.py  DuckDBSink (schema core/mart/ops, append-only ด้วย data_version) และ FileSink
  feature_cache.py          get_or_build(...) แคชฟีเจอร์ตาม hash ของข้อมูล+โค้ด+params
  rules.py                  write_audit → ops.audit_log
  trends_io.py, transform.py  รวมชีท TRENDS (รองรับ Population YYYY แบบ wide)
tests/                      pytest (93 ผ่าน / 3 ข้าม บน host และใน Docker)
docs/roadmap_h1_h4.md       roadmap ของสมมติฐาน
docs/notebook_snippets/     build_features_sql.py, get_fold_sql.py (ให้วางในเซลล์ของ notebook ทีม)
notebooks/                  pipeline_main_duckdb.ipynb, pipeline_main_csv_timed.ipynb (+ %%time)
data/ input/ reports/       ข้อมูลจริงบนเครื่องผู้ใช้ (ไม่ commit)
```

## คำสั่ง

```
source .venv/bin/activate                      # Python 3.12 (uv venv --python 3.12); pyproject ต้องการ >=3.11
python -m dengue_pipeline run --trends-input input/TRENDS.xlsx --weather-input input/weather_daily.csv
python -m dengue_pipeline status               # exit 1 เมื่อยังไม่มี run (อย่าใช้ && ต่อท้าย)
python -m dengue_pipeline export               # -> data/exports/contract_v2/<run_id>/trends_weather_merged.csv + metadata.json
python -m dengue_pipeline inspect-trends input/TRENDS.xlsx
python -m dengue_pipeline migrate-sink         # คัดลอก FileSink versions เข้า DuckDB (idempotent)
pytest -q
docker compose run --rm pipeline <คำสั่งข้างบน>
docker compose --profile tests run --rm tests
```

## การตัดสินใจที่ล็อกแล้ว (อย่าเปลี่ยนโดยไม่ถาม)

- **Decision B:** pipeline ส่งออกเฉพาะ **ตารางฐาน 44 คอลัมน์** (contract_v2) ไม่มี D หรือฟีเจอร์ใดๆ; notebook เป็นที่เดียวที่นิยาม D, ฟีเจอร์ `hist_/wx_/nat_/seas_/reg_` และ anomaly ต่อ fold (fit บน train เท่านั้น)
- ตารางฐานใน pipeline ต้องเหมือน `trends_weather_merged.csv` ที่ทีมใช้ทุกค่า (เทียบที่ tolerance 1e-6)
- ตัดการตรวจข้ามแหล่งข้อมูลทิ้ง; ตรวจเมื่อ validation ล้มเหลว = หยุดทั้ง pipeline
- รวมชีท TRENDS ทำใน pipeline (transform)
- DuckDB: ผู้เขียนได้ทีละ process, ผู้อ่านใช้ `read_only=True`; ชนิดข้อมูล DOUBLE (float32 ทำให้เทียบ 1e-6 พัง); column `_ord` เก็บลำดับแถว; มี view `*_latest`; เขียนใน transaction พร้อม rollback
- Docker: python:3.12-slim multi-stage (base/dev), bind mount config(ro)/data/input/reports, non-root uid 1000, ENTRYPOINT `python -m dengue_pipeline`
- แคชฟีเจอร์: cache ได้เฉพาะฟีเจอร์ที่ไม่ต้อง fit; anomaly ต้องใช้ key เฉพาะ fold
- รันบนเครื่องของ Nut ตามต้องการ (อยากมีตัวเลือก schedule); Airflow เฉพาะถ้าคุ้ม; real-time "ถ้าได้ก็ดี" ไม่บังคับ

## สถานะตอนนี้

ทำแล้ว: ingest→export ครบ, DuckDB sink, feature cache, Docker, notebook เวอร์ชัน DuckDB, `build_features_sql` และ `get_fold_sql` (ผลเท่ากับ pandas เดิมทุกค่า ที่ 1e-9 บนข้อมูลจริง ทุก 7 folds)

ตัวเลขเวลาใน notebook บนเครื่อง Nut (ms; Pandas / DuckDB อ่าน / DuckDB+SQL): อ่านข้อมูล 116/333/378 · build_features 615/638/292 · leakage 2310/2430/2060 · folds 2870/2870/878 · โมเดล 14800/19700/15900 (รันครั้งเดียว; section 9 ผันผวน ไม่เกี่ยวกับ DB) → รวม section 2,3,6,7: 5,911 → 3,608 ms (−39%) ข้อสรุป: DuckDB ไม่ช่วยขั้นอ่านที่ข้อมูลขนาดนี้ ช่วยเมื่อย้ายตรรกะฟีเจอร์/fold เป็น SQL

## สิ่งที่ต้องทำ (เรียงตามลำดับ)

### A. ขั้น 0: ล็อกนิยามก่อนดูผล (เป็นการตัดสินใจ ไม่ใช่โค้ด — ช่วยร่างเอกสารหน้าเดียวและ config)
- [ ] M0 มีอะไร: ผมเสนอตาม proposal (ประวัติ + ฤดูกาล, ไม่มีจังหวัดข้างเคียงเพราะ EDA เห็นว่า correlation แทบไม่ลดตามระยะทาง) ข้างเคียงเป็น sensitivity — **ต้องให้ทีมตัดสิน**
- [ ] M1 = M0 + อากาศ lag 0-4 + anomaly
- [ ] เลือกโมเดลหลักของ H4 ล่วงหน้า (Huber หรือ EN) และล็อกฟีเจอร์ + hyperparameter ก่อนดูผล (F1 มี train แค่ 2016-2018 inner split บางเกินไป)
- [ ] H1/H2: ระบุว่า partial ρ คำนวณบนแถวไหน (train ของ fold ซ้อนทับกัน → เครื่องหมายเดียวกัน 4/5 ได้ง่ายโดยไม่เป็นหลักฐานอิสระ; แถวปีทดสอบ ~52 จุด/จังหวัด เสียงดัง) และควบคุมอะไร (ระดับโรค `hist_logR_prev` สำคัญเพราะ D ถดถอยเข้าหาค่ากลาง); วิธีรวม (มัธยฐานข้ามจังหวัดต่อ fold) และเกณฑ์ "ผ่าน" เมื่อทดสอบ 15 ครั้ง (3 ตัวแปร × 5 lag)
- [ ] H3: กำหนดว่าถ้า H1/H2 ไม่ผ่านให้ H3 เป็น "ไม่ผ่าน" หรือ "ตัดสินไม่ได้"
- [ ] bootstrap ของ H4: block ตามจังหวัดภายในปีน่าจะประเมินความไม่แน่นอนต่ำเกินไป (ผลร่วมทั้งประเทศ 66%, anomaly โรคข้ามจังหวัด ≈ 0.6) ผมเสนอบล็อกตามสัปดาห์ หรือรายงานทั้งสองแบบ และระบุว่าตัดสินจากแบบไหน
- [ ] การจัดการ 2020/2021 (แยกกัน; 2021 ตัดจาก test หรือรายงานแยก; ปีเหล่านี้ยังอยู่ใน train ของ fold หลัง — เป็น sensitivity)
- [ ] บันทึกทั้งหมดลง config เดียว ผูกกับ hash ของ run (เอกสารนิยามหน้าเดียว = จุดตรวจก่อนรันจริง)
- [ ] ตกลงความรับผิดชอบ: pipeline รับขั้น 1, ตาราง results, แคช, leakage test, dashboard; ทีม DS รับนิยาม M0/M1 และการรันขั้น 2-4

### B. ขั้น 1: ตารางฟีเจอร์ที่รองรับทุกสมมติฐาน (ฝั่ง pipeline)
- [ ] lag อากาศ 5-8 ของ rain/tmax/tmin: สร้างจากคอลัมน์รายสัปดาห์เดิม (`wk_*`) ด้วย shift + ตรวจวันที่ต่อเนื่อง ไม่ต้องใช้ข้อมูลรายวัน (รายวันจำเป็นเฉพาะฟีเจอร์สะสม เช่นฝน 12 สัปดาห์) ตั้ง `MAX_WX_LAG = 8`; แถวต้นของจังหวัดหายเพิ่ม จึงต้องเทียบ lag 0-8 บน **ชุดแถวเดียวกัน** และอัปเดตค่าอ้างอิง 32,802 ให้ตรงนิยามใหม่ (ตั้งชื่อค่าแยกกันไว้ ไม่ทับค่าเดิม)
- [ ] anomaly ต่อ fold: ทำแล้ว (`get_fold_sql`) → ตรวจว่า `ref_mask` มาจาก train เท่านั้น
- [ ] **ขยาย leakage test:** ปัจจุบัน `FEATURE_COLS_BASE` ไม่ครอบคลุม `wxa_*` (anomaly) เลย ต้องเพิ่มทดสอบว่าเปลี่ยนอากาศปีทดสอบแล้ว anomaly ของแถว train ต้องไม่เปลี่ยน และครอบคลุม lag 5-8
- [ ] แคชฟีเจอร์ผูกกับ config hash (ตอนนี้ key = ข้อมูล+โค้ด+params)
- [ ] test parity ใน repo: `build_features` (pandas ของทีม) เทียบ `build_features_sql` และ `get_fold` เทียบ `get_fold_sql` บนข้อมูลจริง กันโค้ดสองชุดเพี้ยนกัน และตัดสินว่าที่เดียวที่เป็นความจริงคือชุดไหน (Decision B บอกว่า notebook)

### C. ตาราง results และ reproducibility
- [ ] ตาราง results ใน DuckDB: run id, config hash, fold, โมเดล, ชุดฟีเจอร์, MAE, ผลต่างเทียบ baseline (schema `mart` หรือ `ops`)
- [ ] บันทึก config hash + seed + data_version ต่อ run
- [ ] ให้ notebook ของ DS อ่าน config เดียวกัน

### D. Dashboard (ยังไม่เริ่ม ต้องยืนยันรูปแบบก่อน)
สามหน้า: (1) สุขภาพ pipeline: สถานะ/เวลาแต่ละขั้น, จำนวนแถวเข้า-ออก, ผลเช็กต่อ rule, ประวัติ data_version (ข้อมูลจาก `ops.audit_log`, `ops.data_version`, `reports/`) (2) EDA พื้นฐาน: coverage จังหวัด×สัปดาห์, อัตรารายปีและฤดูกาลต่อปี, สัดส่วนศูนย์, การกระจาย D ต่อปี/เขต, การกระจายอากาศ, จำนวนแถวต่อ fold, null rate ของฟีเจอร์ (3) ติดตาม H1-H4: สถานะ+เกณฑ์+หลักฐาน, ρ ต่อ lag พร้อมแถบข้าม fold, ΔMAE ต่อ fold ของ M0 vs M1
- [ ] เลือกรูปแบบ: **ผมเสนอ** pipeline ส่งออก snapshot (JSON/Parquet) ท้าย run แล้ว dashboard (HTML เดียว) อ่าน snapshot — เลี่ยงปัญหา DuckDB ผู้เขียนได้ทีละ process; ทางเลือก Streamlit ใน Docker แบบ read-only
- [ ] ดูรูปแบบจริงของ `reports/` และ `ops.audit_log` ก่อนออกแบบ schema ของ snapshot

### E. ขั้น 2-6 ของ roadmap (ทีม DS นำ; ฝั่ง pipeline เตรียมของให้)
H4 (M0 vs M1 ต่อ fold + CI + ablation + sensitivity) → H1/H2 (partial ρ + placebo สลับอากาศข้ามปีภายในจังหวัด + ฝนคุมอุณหภูมิ) → H3 (โปรไฟล์ ρ ต่อ lag 0-8) → ตรวจความสอดคล้อง (เตรียม M0 แบบ Fourier ฤดูกาลไว้ล่วงหน้า) → รายงานและข้อจำกัด รายละเอียดทั้งหมดใน `docs/roadmap_h1_h4.md`

### F. งานค้างเล็กๆ
- [ ] pin `duckdb==2.0.0` ใน `pyproject.toml` (ตอนนี้ `>=1.0`) และ Dockerfile ให้ host / Docker / notebook ใช้รุ่นเดียวกัน (ไฟล์ .duckdb ที่เขียนด้วยรุ่นใหม่เปิดด้วยรุ่นเก่าไม่ได้ — Nut เคยเจอ error "version number 999")
- [ ] notebook kernel ต้องเป็น `.venv` (Python 3.12) ไม่ใช่ env 3.9 ตัวเก่า; ยืนยันว่าไฟล์ .duckdb ที่ container เขียนเปิดจาก host ได้
- [ ] (ไม่บังคับ) `pull_policy: build` ให้ service tests ใน docker-compose เพื่อกัน warning pull-access
- [ ] `notebooks/pipeline_main_duckdb.ipynb` ยังมี section 2b ที่อ่าน CSV เพื่อเทียบความเท่ากัน — ถ้าต้องการให้ไม่แตะ CSV เลย ให้ลบ 2b + `DATA_PATH` + แก้ข้อความหัว notebook (ถามก่อน)
- [ ] leakage test ใน notebook ยังเรียก `build_features` แบบ pandas 3 รอบ → สลับเป็น `build_features_sql` ได้ (ยังไม่ได้ทดสอบ)
- [ ] ตรวจ `reports/` ว่า run ครั้งก่อนมีข้อมูลครบ (เคยเจอ `data/validated` ว่างหลังย้ายเครื่อง)

## ข้อควรระวังเชิงเทคนิค

- notebook ทีมที่ผมมีเป็นการถอดจากไฟล์ที่ Nut ส่ง; ต้นฉบับคือ `pipeline_main.ipynb` ของทีม — โค้ด 14 เซลล์ตรงกับ `notebooks/pipeline_main_csv_timed.ipynb` ยกเว้นเซลล์อ่านที่เติม `%%time`
- `_lag` ตรวจว่าแถวก่อนหน้าห่าง k×7 วันพอดีถึงจะใช้ค่า (ไม่งั้น NaN) ทุกฟีเจอร์ lag ต้องผ่านตรรกะนี้ (SQL เวอร์ชันใช้ `date_diff('day', ...)`)
- `rolling(4).sum()` ของ pandas ให้ NaN ถ้ามี NaN ในหน้าต่าง; SQL เวอร์ชันเทียบเท่าด้วย `count(...) = 4`
- `ln(1+x)` แทน `log1p` ใน SQL (DuckDB ไม่มี `log1p`; ต่างกันระดับ 1e-13)
- ผล `build_features_sql`/`get_fold_sql` ทดสอบกับ DuckDB 2.0.0 + pandas 3.x เท่านั้น
- `status` คืน exit 1 เมื่อยังไม่มี run
- แหล่งอากาศ ERA5 เป็นจุด centroid (ยังไม่ได้ตรวจข้ามกับตารางฝนกรมอุตุฯ; หน่วยของ TMD AvgRain ไม่ทราบ — ตัดสินใจแล้วว่าไม่ทำการตรวจข้ามแหล่ง)
