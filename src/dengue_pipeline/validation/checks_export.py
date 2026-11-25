"""กฎของไฟล์ส่งออก (EXP / LKG) ตรวจก่อนเขียนไฟล์ ถ้าไม่ผ่านจะไม่เขียนอะไร

EXP-01 คอลัมน์ตรงกับ contract ทั้งชื่อและลำดับ
EXP-02 ไม่มีค่าว่างในทุกคอลัมน์ (notebook assert ไว้)
EXP-03 คีย์ไม่ซ้ำ และแถวเรียงตามคีย์
LKG-02 ไม่มีคอลัมน์ที่คำนวณจากหลายปี (เทียบกับรูปแบบชื่อที่ห้ามใน contract)
"""
from __future__ import annotations

import re

import pandas as pd

from dengue_pipeline.contract import Contract
from dengue_pipeline.rules import RuleResult, check


def check_export(table: pd.DataFrame, contract: Contract) -> list[RuleResult]:
    out: list[RuleResult] = []

    same = list(table.columns) == contract.output_columns
    out.append(check("EXP-01", same, f"{table.shape[1]} คอลัมน์",
                     "" if same else f"ต่างจาก contract: ขาด {[c for c in contract.output_columns if c not in table.columns]}, "
                                     f"เกิน {[c for c in table.columns if c not in contract.output_columns]}"))

    nulls = table.isna().sum()
    nulls = nulls[nulls > 0]
    out.append(check("EXP-02", nulls.empty, f"คอลัมน์ที่มีค่าว่าง {len(nulls)} คอลัมน์",
                     "" if nulls.empty else str(nulls.to_dict())))

    dup = int(table.duplicated(contract.keys).sum()) if set(contract.keys) <= set(table.columns) else -1
    ordered = dup == 0 and table[contract.keys].equals(table[contract.keys].sort_values(contract.keys))
    out.append(check("EXP-03", ordered, f"คีย์ซ้ำ {dup} แถว",
                     "" if ordered else "คีย์ซ้ำหรือแถวไม่เรียงตามคีย์"))

    bad = [c for c in table.columns if any(re.search(p, c) for p in contract.forbidden_patterns)]
    out.append(check("LKG-02", not bad, f"คอลัมน์ต้องห้าม {len(bad)} คอลัมน์", f"{bad}" if bad else ""))
    return out
