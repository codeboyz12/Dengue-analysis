"""ข้อยกเว้นที่ตัวรันใช้แยก 3 สถานะ: สำเร็จ / ยังไม่ครบ / ตรวจไม่ผ่าน"""
from __future__ import annotations


class StageError(Exception):
    """ขั้นตอนทำต่อไม่ได้ด้วยเหตุที่ไม่ใช่ข้อมูลผิด (ไฟล์หาย, ยังไม่ได้เขียนโค้ดส่วนนั้น ฯลฯ) -> FAILED"""


class IncompleteRun(Exception):
    """ดึงไม่จบด้วยเหตุภายนอก เช่น โควตารายวันหมด -> INCOMPLETE รันซ้ำเพื่อต่อได้"""


class ValidationFailed(Exception):
    """กฎระดับ BLOCK อย่างน้อยหนึ่งข้อไม่ผ่าน -> FAILED + quarantine"""

    def __init__(self, stage: str, results: list):
        self.stage = stage
        self.results = results
        failed = [r.rule_id for r in results if r.blocking_failure]
        super().__init__(f"{stage}: กฎ BLOCK ไม่ผ่าน {failed}")
