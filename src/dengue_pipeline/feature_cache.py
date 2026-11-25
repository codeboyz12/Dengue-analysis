"""แคชผลของฟังก์ชันสร้างฟีเจอร์ (build_features ฯลฯ) ลงดิสก์ เพื่อไม่ต้องคำนวณซ้ำเมื่อข้อมูลและโค้ดไม่เปลี่ยน

ใช้จาก notebook โดยไม่ย้ายนิยามฟีเจอร์ออกจาก notebook (นิยามยังอยู่ที่เดียว):

    from dengue_pipeline.feature_cache import get_or_build
    df = get_or_build(raw, build_features, name="base")

cache key = sha256 ของ
  1. เนื้อหาของตารางอินพุต (หรือ data_version ถ้าระบุ จะเร็วกว่า)
  2. ซอร์สโค้ดของฟังก์ชัน + ฟังก์ชันที่มันเรียกต่อ (เช่น _province_block, _lag ที่นิยามในไฟล์/notebook เดียวกัน)
  3. ค่าคงที่ที่ฟังก์ชันอ่านจาก global (เช่น FOURIER_K, MAX_WX_LAG, WX_BASE, TARGET) ค่าเปลี่ยน key เปลี่ยน
  4. params ที่ส่งมาเอง และชื่อแคช
เปลี่ยนอะไรก็ตามข้างต้น = key ใหม่ = คำนวณใหม่ ไม่มีทางได้ผลเก่าโดยไม่รู้ตัว
ข้อจำกัด: จับได้เฉพาะสิ่งที่ฟังก์ชันอ้างถึงโดยตรงผ่านชื่อ global (ไม่ตามไปในไฟล์ที่อ่านเอง เช่น config ภายนอก) ถ้าฟังก์ชันอ่านสิ่งอื่น
ให้ใส่ไว้ใน params=

แคชใช้กับสิ่งที่ "ไม่ต้อง fit" เท่านั้น ถ้าผลขึ้นกับ train ของ fold (anomaly, scaler) ให้ใส่นิยาม fold ไว้ใน params
(หรือส่งเฉพาะแถว train เป็นอินพุต) เพื่อให้แต่ละ fold ได้ key ของตัวเอง ห้ามใช้แคชข้าม fold
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
import types
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_DIR = "feature_cache"
_PRIMITIVES = (int, float, str, bool, type(None))


def _is_const(v) -> bool:
    if isinstance(v, _PRIMITIVES):
        return True
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_is_const(x) for x in v)
    if isinstance(v, dict):
        return all(_is_const(k) and _is_const(x) for k, x in v.items())
    return False


def _code_objects(code: types.CodeType):
    yield code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            yield from _code_objects(c)


def fingerprint_callable(fn: Callable) -> dict:
    """ลายนิ้วมือของฟังก์ชัน: ซอร์สของมันและของฟังก์ชันที่ผู้ใช้นิยามเองที่มันเรียกต่อ + ค่าคงที่ global ที่อ่าน"""
    parts: dict[str, str] = {}
    seen: set[int] = set()

    def visit(f: Callable, label: str) -> None:
        if id(f) in seen:
            return
        seen.add(id(f))
        try:
            src = inspect.getsource(f)
        except (OSError, TypeError) as ex:
            raise ValueError(
                f"อ่านซอร์สของ {label} ไม่ได้ ({ex}) จึงสร้าง cache key ที่เชื่อถือได้ไม่ได้ "
                "ให้ส่ง code_version=\"...\" เองและเปลี่ยนเมื่อแก้โค้ด") from ex
        parts[f"fn:{label}"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
        glb = getattr(f, "__globals__", {})
        names = {n for co in _code_objects(f.__code__) for n in co.co_names}
        for n in sorted(names):
            if n not in glb:
                continue
            v = glb[n]
            if isinstance(v, types.FunctionType) and v.__module__ == f.__module__:
                visit(v, n)
            elif _is_const(v):
                parts[f"const:{n}"] = json.dumps(v, sort_keys=True, default=sorted, ensure_ascii=False)

    visit(fn, getattr(fn, "__name__", "fn"))
    return parts


def fingerprint_frame(df: pd.DataFrame) -> str:
    """เนื้อหาของตาราง: ค่า ลำดับแถว ชื่อและชนิดคอลัมน์ (ไม่ใช่แค่ขนาด)"""
    h = hashlib.sha256()
    h.update(json.dumps([[str(c), str(t)] for c, t in df.dtypes.items()], ensure_ascii=False).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def cache_key(df: pd.DataFrame, builder: Callable, *, name: str = "features", params: dict | None = None,
              data_version: str | None = None, code_version: str | None = None) -> tuple[str, dict]:
    parts = {
        "name": name,
        "data": data_version if data_version is not None else fingerprint_frame(df),
        "params": json.dumps(params or {}, sort_keys=True, default=str, ensure_ascii=False),
        "pandas": pd.__version__,
    }
    if code_version is not None:
        parts["code_version"] = code_version
    else:
        parts.update(fingerprint_callable(builder))
    key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
    return key, parts


def get_or_build(df: pd.DataFrame, builder: Callable[[pd.DataFrame], pd.DataFrame], *, name: str = "features",
                 params: dict | None = None, cache_dir: str | os.PathLike | None = None,
                 data_version: str | None = None, code_version: str | None = None,
                 gate: Callable[[pd.DataFrame], None] | None = None, verbose: bool = True) -> pd.DataFrame:
    """คืน builder(df) จากแคชถ้ามี ไม่งั้นคำนวณแล้วเก็บ

    gate = ฟังก์ชันตรวจผลที่คำนวณใหม่ก่อนเก็บ (เช่น leakage test) ถ้ายก exception จะไม่เก็บลงแคช
    ผลที่คืนเป็น DataFrame ใหม่ทุกครั้ง แก้ไขได้โดยไม่กระทบแคช
    """
    root = Path(cache_dir if cache_dir is not None else DEFAULT_DIR)
    t0 = time.perf_counter()
    key, parts = cache_key(df, builder, name=name, params=params, data_version=data_version, code_version=code_version)
    t_key = time.perf_counter() - t0
    path = root / f"{name}-{key[:16]}.parquet"
    if path.exists():
        t1 = time.perf_counter()
        out = pd.read_parquet(path)
        if verbose:
            print(f"[feature_cache] hit  {path.name}  key {t_key:.2f}s + โหลด {time.perf_counter() - t1:.2f}s")
        return out

    t1 = time.perf_counter()
    out = builder(df)
    t_build = time.perf_counter() - t1
    if gate is not None:
        gate(out)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp-{os.getpid()}-{path.name}"
    out.to_parquet(tmp, index=False)
    meta = {"key": key, "name": name, "rows": len(out), "columns": int(out.shape[1]), "build_seconds": round(t_build, 3),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "inputs": parts}
    (root / f"{name}-{key[:16]}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    if verbose:
        print(f"[feature_cache] miss {path.name}  key {t_key:.2f}s + คำนวณ {t_build:.2f}s → เก็บแล้ว")
    return out


def cache_info(cache_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """รายการแคชที่มี: ชื่อ ขนาดแถว/คอลัมน์ เวลาคำนวณตอนสร้าง ขนาดไฟล์"""
    root = Path(cache_dir if cache_dir is not None else DEFAULT_DIR)
    rows = []
    for j in sorted(root.glob("*.json")) if root.exists() else []:
        m = json.loads(j.read_text(encoding="utf-8"))
        pq = j.with_suffix(".parquet")
        rows.append({"file": pq.name, "name": m["name"], "rows": m["rows"], "columns": m["columns"],
                     "build_seconds": m["build_seconds"], "size_mb": round(pq.stat().st_size / 1024**2, 2) if pq.exists() else None,
                     "created_at": m["created_at"]})
    return pd.DataFrame(rows, columns=["file", "name", "rows", "columns", "build_seconds", "size_mb", "created_at"])


def prune(name: str | None = None, keep: int = 1, cache_dir: str | os.PathLike | None = None) -> list[str]:
    """ลบแคชเก่า เหลือไว้ keep ไฟล์ล่าสุดต่อชื่อ (ค่าตั้งต้น 1) คืนรายชื่อไฟล์ที่ลบ"""
    root = Path(cache_dir if cache_dir is not None else DEFAULT_DIR)
    removed = []
    info = cache_info(root)
    for n, g in info.groupby("name"):
        if name is not None and n != name:
            continue
        for f in g.sort_values("created_at", ascending=False)["file"].iloc[keep:]:
            for suffix in (".parquet", ".json"):
                p = (root / f).with_suffix(suffix)
                if p.exists():
                    p.unlink()
            removed.append(f)
    return removed
