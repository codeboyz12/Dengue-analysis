# สรุปผลเบื้องต้น — ฝั่ง Pipeline (Nut)

วันที่: 2026-11-20 | data_version: 20261118T054719Z | config_hash: 5f21320732c4

---

## ผลสมมติฐาน H1–H4

| สมมติฐาน | ผล | หลักฐาน |
|---|---|---|
| **H1** ฝน → D (partial ρ) | ❌ ไม่ผ่าน | \|ρ\| สูงสุด 0.075 ที่ lag 2 (เกณฑ์ 0.10); ทุก lag อยู่ใน null distribution ของ placebo |
| **H2** อุณหภูมิ → D (partial ρ) | ❌ ไม่ผ่าน | tmax/tmin \|ρ\| ≈ 0.10 ที่ lag 1-2 แต่เครื่องหมายสลับข้าม fold (ไม่ผ่าน same-sign ≥ 4/5) |
| **H3** lag profile | — ตัดสินไม่ได้ | H1/H2 ไม่ผ่าน → ตามนิยาม H3 = ตัดสินไม่ได้ |
| **H4** M1 ลด MAE ≥ 5% | ❌ ไม่ผ่าน | Huber M1 ดีกว่า M0 ใน 1/5 fold; ΔMAE range −19% ถึง +5.6% |

**ข้อสรุป:** อากาศ (rain/tmax/tmin anomaly lag 0-4) ไม่เพิ่มประโยชน์ในการพยากรณ์ไข้เลือดออกในข้อมูลชุดนี้ — ผลนี้สอดคล้องกับทีม DS

---

## ผล H4 ต่อ fold (HuberRegressor, primary model)

| fold | MAE M0 | MAE M1 | ΔMAE (%) | CI 95% | ผ่าน? |
|---|---|---|---|---|---|
| 2019 | 0.3021 | 0.3601 | **−19.2%** | −23.6 ถึง −14.2 | ❌ M1 แย่กว่า |
| 2022 | 0.2983 | 0.2900 | +2.8% | +1.4 ถึง +3.8 | ❌ < 5% |
| 2023 | 0.2943 | 0.2957 | −0.5% | −1.8 ถึง +0.8 | ❌ CI คร่อมศูนย์ |
| 2024 | 0.2914 | 0.2751 | **+5.6%** | +3.4 ถึง +7.4 | ✅ |
| 2025 | 0.2830 | 0.2795 | +1.2% | −0.3 ถึง +2.7 | ❌ CI คร่อมศูนย์ |

> CI = block-week bootstrap 100 replicates (final run ต้องการ 1,000)

---

## Sensitivity comparison (preliminary)

| โมเดล | fold ที่ M1 ดีกว่า M0 | ΔMAE range |
|---|---|---|
| HuberRegressor (primary) | 1/5 | −19% ถึง +5.6% |
| ElasticNetCV | 1/5 | −20% ถึง +4.7% |
| HistGradientBoosting | 2/5 | +0.1% ถึง +6.9% |
| XGBRegressor | (รันอยู่) | — |
| TabMMini | (รันอยู่) | — |

**ข้อสังเกต:**
- F1(2019) M1 แย่กว่าทุกโมเดล — training data บาง (2016-2018 เท่านั้น ~10K rows) อาจ overfit บน weather features
- F4(2024) เด่นทุกโมเดล — อาจเป็น anomaly ของปีนั้นโดยเฉพาะ ไม่ใช่สัญญาณทั่วไป
- HistGBR ดีกว่า Huber บางส่วน สอดคล้องกับที่ทีม DS พบว่า tree-based models ทำงานได้ดีกว่า linear

---

## สิ่งที่ตรงกับทีม DS

| ประเด็น | Pipeline (Nut) | ทีม DS |
|---|---|---|
| H4 ผ่านหรือไม่ | ❌ ΔMAE ~1-6% | ❌ XGBoost 4.24%, TabM 2.44% |
| Lag ที่มีสัญญาณ | lag 0-4: อ่อนมาก | lag 5-8 (biological): แรงกว่า |
| Raw weather vs anomaly | anomaly อ่อนกว่า raw ใน HistGBR | XGBoost: custom lag 5-8 ดีกว่า raw lag 0-4 |
| 2021 | supplementary fold เท่านั้น | TabM ตัด 2021 ออกจาก test |

**ข้อสรุปร่วม:** สัญญาณอากาศอยู่ที่ biological lag 5-8 สัปดาห์ (อายุยุง/ระยะฟักตัว) ไม่ใช่ lag 0-4 ที่ H1/H2 ทดสอบ ถ้าจะ unblind H3 ด้วย lag 5-8 ต้องตัดสินก่อนว่า scope ของ H3 ครอบคลุมถึงไหม

---

## ไฟล์และข้อมูลที่ทีม DS ใช้ได้

```bash
# ดูผลทั้งหมดใน DuckDB
import duckdb
con = duckdb.connect("data/pipeline.duckdb", read_only=True)

# H4 results
con.execute("SELECT * FROM mart.results WHERE model='HuberRegressor' ORDER BY fold").df()

# H1/H2 ρ profile
con.execute("SELECT * FROM mart.h_results ORDER BY var, lag, fold").df()

# H1-H4 decisions
import json
dec = json.loads(con.execute("SELECT decisions_json FROM mart.h_decisions ORDER BY created_at DESC LIMIT 1").fetchone()[0])

# Dashboard
streamlit run dashboard/app.py
```

---

## งานที่เหลือก่อน final run

- [ ] รัน train ให้เสร็จ (XGBoost + TabMMini กำลังรันอยู่ ~2 ชั่วโมง)
- [ ] เปลี่ยน `n_replicates = 1000` ใน `config/analysis.toml` แล้วรัน final
- [ ] ตัดสินใจว่าจะทดสอบ H3 ด้วย lag 5-8 หรือไม่ (ต้องตกลงกับทีม DS)
