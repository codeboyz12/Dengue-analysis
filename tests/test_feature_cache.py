"""แคชฟีเจอร์: hit/miss ที่ถูกต้อง และที่สำคัญคือ "ไม่ให้ผลเก่าเมื่อข้อมูลหรือโค้ดเปลี่ยน" """
import textwrap

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dengue_pipeline.feature_cache import (cache_info, cache_key, fingerprint_callable, get_or_build, prune)

SCALE = 2.0


def _helper(s):
    return s * SCALE


def builder(df):
    out = df.copy()
    out["y"] = _helper(out["x"])
    return out


@pytest.fixture
def df():
    return pd.DataFrame({"P-code": ["A", "A", "B"], "x": [1.0, 2.5, 3.0], "d": pd.to_datetime(["2019-01-06"] * 3)})


def test_second_call_is_a_hit_and_returns_an_identical_frame(df, tmp_path, capsys):
    a = get_or_build(df, builder, cache_dir=tmp_path)
    b = get_or_build(df, builder, cache_dir=tmp_path)
    assert_frame_equal(a, b)
    assert_frame_equal(a, builder(df))
    out = capsys.readouterr().out
    assert "miss" in out and "hit" in out


def test_hit_does_not_call_the_builder(df, tmp_path):
    calls = []

    def counting(d):
        calls.append(1)
        return d.assign(z=1)

    get_or_build(df, counting, cache_dir=tmp_path, code_version="v1", verbose=False)
    get_or_build(df, counting, cache_dir=tmp_path, code_version="v1", verbose=False)
    assert len(calls) == 1


def test_changed_data_gives_a_new_key(df):
    k1, _ = cache_key(df, builder)
    changed = df.copy()
    changed.loc[1, "x"] = 2.6
    assert cache_key(changed, builder)[0] != k1
    assert cache_key(df.iloc[::-1], builder)[0] != k1                    # ลำดับแถวต่างก็คือข้อมูลต่าง
    assert cache_key(df.rename(columns={"x": "x2"}), builder)[0] != k1
    assert cache_key(df.astype({"x": "float32"}), builder)[0] != k1
    assert cache_key(df.copy(), builder)[0] == k1                        # สำเนาเหมือนเดิม key เหมือนเดิม


def test_changed_params_or_name_give_a_new_key(df):
    base, _ = cache_key(df, builder)
    assert cache_key(df, builder, params={"k": 2})[0] != base
    assert cache_key(df, builder, params={"k": 2})[0] == cache_key(df, builder, params={"k": 2})[0]
    assert cache_key(df, builder, name="other")[0] != base


def test_data_version_replaces_hashing_the_frame(df):
    k1, _ = cache_key(df, builder, data_version="20261101T000000Z")
    changed = df.assign(x=0.0)
    assert cache_key(changed, builder, data_version="20261101T000000Z")[0] == k1        # เชื่อ data_version
    assert cache_key(df, builder, data_version="20261102T000000Z")[0] != k1


def test_editing_the_builder_changes_the_key(tmp_path):
    ns1, ns2 = {}, {}
    src1 = textwrap.dedent("""
        def build(df):
            return df.assign(y=df["x"] + 1)
    """)
    src2 = src1.replace("+ 1", "+ 2")
    for src, ns in ((src1, ns1), (src2, ns2)):
        f = tmp_path / f"m{len(ns1)}{len(ns2)}.py"
        f.write_text(src)
        exec(compile(src, str(f), "exec"), ns)
        import linecache
        linecache.updatecache(str(f))
    k1 = fingerprint_callable(ns1["build"])
    k2 = fingerprint_callable(ns2["build"])
    assert k1 != k2


def test_changing_a_helper_function_or_a_global_constant_changes_the_key(df, monkeypatch):
    import sys
    base, parts = cache_key(df, builder)
    assert "const:SCALE" in parts and "fn:_helper" in parts           # ตามไปถึงฟังก์ชันช่วยและค่าคงที่ที่ใช้
    mod = sys.modules[builder.__module__]
    monkeypatch.setattr(mod, "SCALE", 3.0)
    assert cache_key(df, builder)[0] != base                          # ค่าคงที่เปลี่ยน -> key เปลี่ยน
    monkeypatch.setattr(mod, "SCALE", 2.0)
    assert cache_key(df, builder)[0] == base


def test_unreadable_source_requires_an_explicit_code_version(df, tmp_path):
    fn = eval("lambda d: d")                                           # ไม่มีซอร์สในไฟล์ให้อ่าน
    with pytest.raises(ValueError, match="code_version"):
        get_or_build(df, fn, cache_dir=tmp_path)
    get_or_build(df, fn, cache_dir=tmp_path, code_version="1", verbose=False)


def test_gate_failure_is_not_cached(df, tmp_path):
    def gate(out):
        raise AssertionError("leakage")
    with pytest.raises(AssertionError):
        get_or_build(df, builder, cache_dir=tmp_path, gate=gate, verbose=False)
    assert not list(tmp_path.glob("*.parquet")) and cache_info(tmp_path).empty
    get_or_build(df, builder, cache_dir=tmp_path, verbose=False)        # ครั้งต่อมาที่ผ่านก็เก็บได้


def test_returned_frame_can_be_modified_without_touching_the_cache(df, tmp_path):
    a = get_or_build(df, builder, cache_dir=tmp_path, verbose=False)
    a["y"] = -1
    b = get_or_build(df, builder, cache_dir=tmp_path, verbose=False)
    assert_frame_equal(b, builder(df))


def test_dtypes_and_nan_survive_the_round_trip(tmp_path):
    big = pd.DataFrame({"P-code": ["A", "B"], "d": pd.to_datetime(["2019-01-06", "2019-01-13"]),
                        "n": np.array([1, 2], dtype="int64"), "f": [np.nan, 0.1 + 0.2]})
    ident = lambda d: d.copy()                                           # noqa: E731
    a = get_or_build(big, ident, cache_dir=tmp_path, code_version="1", verbose=False)
    b = get_or_build(big, ident, cache_dir=tmp_path, code_version="1", verbose=False)
    assert_frame_equal(a, b) and assert_frame_equal(b, ident(big))


def test_cache_info_and_prune_keep_the_newest(df, tmp_path):
    for i in range(3):
        get_or_build(df, builder, cache_dir=tmp_path, params={"i": i}, verbose=False)
        import time; time.sleep(1.05)                                   # created_at ละเอียดระดับวินาที
    info = cache_info(tmp_path)
    assert len(info) == 3 and (info["rows"] == 3).all()
    removed = prune(keep=1, cache_dir=tmp_path)
    assert len(removed) == 2 and len(cache_info(tmp_path)) == 1
    assert len(list(tmp_path.glob("*.parquet"))) == 1 and len(list(tmp_path.glob("*.json"))) == 1
