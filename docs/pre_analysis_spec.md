# Pre-Analysis Specification — ไข้เลือดออก × อากาศ (H1–H4)

วันที่ล็อก: 2026-11-19 | config: `config/analysis.toml`

**กฎ:** เอกสารนี้และ config ต้องไม่เปลี่ยนหลังดูผลบน test set  
เปลี่ยนได้เฉพาะ `epsilon_locked` / `alpha_locked` (กรอกจาก inner CV ก่อน unblind)

---

## 1. Target D

```
R      = dengue_cases_sum / population × 1e5   (เฉพาะ week_days == 7)
R_prev = sum(cases t-3..t) / pop × 1e5          (4 สัปดาห์เต็ม)
R_next = sum(cases t+1..t+4) / pop × 1e5        (4 สัปดาห์เต็ม)
D      = log1p(R_next) − log1p(R_prev)
```

ค่าอ้างอิง: D ไม่ null = **35,266** แถว; มีทั้ง D และ `hist_growth_prev` = **32,802** แถว

---

## 2. โมเดล

### M0 — Baseline (ประวัติ + ฤดูกาล)

| กลุ่ม | คอลัมน์ | หมายเหตุ |
|---|---|---|
| hist | `hist_lograte_lag0..3`, `hist_logR_prev`, `hist_logR_prev2`, `hist_growth_prev` | lag 0-3 รายสัปดาห์; rolling 4w |
| nat | `nat_lograte_lag0`, `nat_logR_prev`, `nat_growth_prev` | ระดับประเทศ |
| seas | `seas_sin1..3`, `seas_cos1..3` | Fourier K=3 ต่อ day-of-year |
| reg | `reg_1..reg_13` | one-hot 12 เขตสุขภาพ |

**ไม่มี:** จังหวัดข้างเคียง (sensitivity analysis เท่านั้น เพราะ EDA: correlation ไม่ลดตามระยะทาง)

### M1 — M0 + อากาศ

เพิ่มจาก M0:
- `wx_{rain,tmax,tmin}_lag0..4` — อากาศดิบ lag 0-4 (ERA5 centroid)
- `wxa_{rain,tmax,tmin}_lag0..4` — anomaly ต่อ fold (mean/std จาก **train** เท่านั้น)

รูปแบบอื่น (ระยะยาว lag 5-8, ฝนสะสม 12 สัปดาห์) เป็น sensitivity ของ H3

---

## 3. โมเดลหลักและ Hyperparameter (H4)

**โมเดลหลัก:** `HuberRegressor` (เลือกก่อนดูผล test)

Hyperparameter เลือกด้วย inner 5-fold CV บน train ของแต่ละ fold:
- grid: `epsilon` ∈ {1.10, 1.35, 1.50}, `alpha` ∈ {1e-4, 1e-3, 1e-2}
- metric: MAE ของ D

**lock ก่อน unblind:** กรอกใน `config/analysis.toml` ที่ `epsilon_locked` / `alpha_locked`

โมเดลเสริม (ตรวจความทนทาน ไม่ตัดสิน H4): ElasticNet, HistGradientBoosting

---

## 4. Folds

| fold | test year | train | หมายเหตุ |
|---|---|---|---|
| F1 | 2019 | 2016-2018 | inner split บางมาก |
| F2 | 2022 | 2016-2021 | — |
| F3 | 2023 | 2016-2022 | — |
| F4 | 2024 | 2016-2023 | — |
| F5 | 2025 | 2016-2024 | — |
| S1 | 2020 | supplementary | ระบาดน้อยผิดปกติ |
| S2 | 2021 | supplementary | แทบไม่มีระบาด |

gap = 4 สัปดาห์ (`label_end < test_start`), expanding train, seed = 342

**2020/2021:** รายงานแยกเป็น fold เสริม ยังอยู่ใน train ของ F2-F5 (sensitivity)

---

## 5. เกณฑ์ตัดสิน

### H1 (ฝน) และ H2 (tmax, tmin)

วิธีคำนวณ:
1. คำนวณ partial Spearman ระหว่าง `wxa_{var}_lag{k}` กับ D ภายในจังหวัด คุม `hist_logR_prev` บน **train** ของแต่ละ fold
2. รวมด้วย median |ρ| ข้ามจังหวัดต่อ fold (จังหวัดไม่อิสระ anomaly โรคสัมพันธ์ ≈ 0.6)

เกณฑ์ผ่าน **ต่อตัวแปร:**
- มี **≥ 2 จาก 5 lag** (lag 0-4) ที่ทั้งสองเงื่อนไขนี้เป็นจริงพร้อมกัน:
  - median |ρ| ≥ 0.10 ข้ามจังหวัดใน fold นั้น
  - เครื่องหมาย median เดียวกัน ≥ 4/5 fold หลัก
- เหตุผลที่ต้อง ≥ 2 lag: lag เดียวอาจบังเอิญ และ lag 0-4 ไม่อิสระต่อกัน

| สมมติฐาน | ตัวแปร | จำนวนการทดสอบ |
|---|---|---|
| H1 | rain | 5 (lag 0-4) |
| H2 | tmax, tmin | 10 (2 ตัว × 5 lag) |

รายงาน pattern ด้วย (กี่ lag ผ่านจากทั้งหมด)

### H3 (lag ที่ใช้ได้จริง)

- ขยาย lag เป็น 0-8 (ต้องรอ lag 5-8 จากขั้น B)
- ผ่าน: มี lag ≥ 1 ตัวผ่านเกณฑ์ H1/H2 **และ** lag ที่ดีที่สุดไม่อยู่ที่ขอบ (ไม่ใช่ lag 0 หรือ 8)
- ถ้า H1 **และ** H2 ไม่ผ่าน → H3 = "ตัดสินไม่ได้"

### H4 (ผลต่าง MAE)

| เงื่อนไข | ค่า |
|---|---|
| ลด MAE ≥ | 5% |
| M1 ดีกว่า ≥ | 4 จาก 5 fold หลัก |
| CI ของผลต่าง | ไม่คร่อมศูนย์ |
| ตัวตัดสิน bootstrap | block-week (หลัก) |

---

## 6. Bootstrap CI ของ H4

| แบบ | บล็อก | หมายเหตุ |
|---|---|---|
| **หลัก** | สัปดาห์ (n ≈ 52/fold) | สะท้อน correlation ข้ามจังหวัดได้ดีกว่า (66% ผลร่วมรายปี) |
| เสริม | จังหวัดภายในปี | มาตรฐานในวรรณกรรม แต่น่าจะประเมิน SE ต่ำเกินไป |

n_replicates = 1,000, seed = 342, CI = 95%  
รายงานทั้งสองแบบในตาราง sensitivity

---

## 7. ความรับผิดชอบ

| งาน | รับผิดชอบ |
|---|---|
| ตารางฐาน 44 คอลัมน์, lag 0-8, leakage test | Pipeline (Nut) |
| ตาราง results ใน DuckDB, feature cache | Pipeline (Nut) |
| นิยาม M0/M1, รัน H1-H4 | ทีม DS |
| เลือก primary lag (H3) | Nut |
| รายงานและ interpretation | ทีม DS + Nut ตรวจ |

---

## 8. จุดตรวจก่อน unblind

- [x] `epsilon_locked = 1.35` และ `alpha_locked = 1.0` กรอกใน `config/analysis.toml` แล้ว
- [ ] lag 5-8 ใน pipeline ผ่าน leakage test แล้ว
- [ ] anomaly คำนวณจาก train เท่านั้น (ตรวจ `ref_mask`)
- [ ] config hash ของ run ตรงกับที่บันทึกใน `ops.audit_log`
