"""Faults the independent audit (2026-11-18) found in decoding the same file again: the repeat cache's light
check, coefficient handles and the copy-kernel check. Each test fails on a54cd05 and passes with the fix."""
import io
import os

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import RepeatCache  # noqa: E402
from exact_gpu_jpeg import _copycheck  # noqa: E402
from conftest import reference  # noqa: E402


def _photo(w, h, seed=1, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


def test_a_light_hit_whose_copy_cannot_be_decoded_reads_the_file(tmp_path, monkeypatch):
    data = _photo(640, 480)
    p = tmp_path / "a.jpg"
    p.write_bytes(data)
    dec = egj.Decoder(cache=RepeatCache())
    dec.decode(str(p), check="light")                    # decoded, copy kept, path recorded
    monkeypatch.setattr(dec, "decode_from_handle", lambda *a, **k: None)     # e.g. memory short for the copy
    img = dec.decode(str(p), check="light")              # was: TypeError (decode(None))
    assert img is not None and np.array_equal(img.get(), reference(data))


def _same_head_and_size(a, b):
    """Two JPEG files with the same size and the same first 64 KiB, and different pixels."""
    pad = b"\xff\xfe" + (65533).to_bytes(2, "big") + b"\x00" * 65531           # a 64 KiB comment after SOI
    x, y = a[:2] + pad + a[2:], b[:2] + pad + b[2:]
    n = max(len(x), len(y))
    return x + b"\x00" * (n - len(x)), y + b"\x00" * (n - len(y))              # (bytes after EOI are ignored)


def test_light_compares_with_the_file_last_read_in_full(tmp_path):
    x, y = _same_head_and_size(_photo(640, 480, seed=1), _photo(640, 480, seed=2))
    p = tmp_path / "f.jpg"
    p.write_bytes(x)
    st = os.stat(p)
    c = RepeatCache()
    dec = egj.Decoder(cache=c)
    assert np.array_equal(dec.decode(str(p), check="light").get(), reference(x))
    p.write_bytes(y)                                     # the documented miss of the light check: same size, time
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))     # and first 64 KiB
    assert np.array_equal(dec.decode(str(p), check="full").get(), reference(y))
    got = dec.decode(str(p), check="light")              # was: x's copy again, after a full read returned y
    assert np.array_equal(got.get(), reference(y))


def test_a_handle_decode_applies_max_pixels():
    data = _photo(640, 480)
    dec = egj.Decoder(keep_coefs="packed")
    dec.decode(data)
    h = dec.take_coefs()
    small = egj.Decoder(max_pixels=1000)
    assert small.decode_from_handle(h) is None and small.reason == "more than max_pixels (1000)"


def test_no_older_handle_after_a_call_that_raised():
    dec = egj.Decoder(keep_coefs="raw")
    dec.decode(_photo(320, 240, seed=1))
    with pytest.raises(ValueError):
        dec.decode(_photo(320, 240, seed=2), dtype="int8")
    assert dec.take_coefs() is None
    dec.decode(_photo(320, 240, seed=1))
    h = dec.take_coefs()
    with pytest.raises(ValueError):
        dec.decode_from_handle(h, dtype="int8")
    assert dec.take_coefs() is None


def test_memory_short_during_the_copy_check_is_not_a_failed_check(monkeypatch):
    from exact_gpu_jpeg import _packed
    dev = cp.cuda.Device().id
    monkeypatch.setattr(_copycheck, "_state", {})
    real, calls = _packed.pack, []

    def pack(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 0, 0)
        return real(*a, **k)
    monkeypatch.setattr(_packed, "pack", pack)
    r = _copycheck.state(dev)
    assert r["packed"] == _copycheck.NO_MEMORY and dev not in _copycheck._state
    r = _copycheck.state(dev)                            # the next use checks again, and it passes
    assert r["packed"] is None and _copycheck._state[dev] is r


def test_a_cache_copy_does_not_hash_the_file_twice(monkeypatch):
    from exact_gpu_jpeg import _repeat
    calls = []
    real = _repeat.check_value
    monkeypatch.setattr(_repeat, "check_value", lambda d: calls.append(1) or real(d))
    dec = egj.Decoder(cache=RepeatCache())
    dec.decode(_photo(640, 480, seed=6))
    assert len(calls) == 1


def test_a_staged_buffer_changed_before_a_cached_decode_is_kept_under_its_own_identity():
    """(second audit, part 1.1) upload_ahead(b), then b changed, then a cached decode of b: the image is b's new
    content's, and no copy is kept under one file's identity with the other's coefficients."""
    from exact_gpu_jpeg import RepeatCache
    a, b = _photo(640, 480, seed=11), _photo(640, 480, seed=12)
    c = RepeatCache(min_pixels=0)
    dec = egj.Decoder(cache=c)
    buf = bytearray(a)
    dec.upload_ahead(buf)                                # a's bytes staged
    buf[:] = b                                           # the program reuses its buffer
    got = dec.decode(buf)
    assert got is not None and np.array_equal(got.get(), reference(b))
    other = egj.Decoder(cache=c).decode(bytes(b))        # a later decode of b, from the copy if one was kept
    assert other is not None and np.array_equal(other.get(), reference(b))


def test_memory_short_during_the_copy_check_is_asked_again_by_the_same_decoder(monkeypatch):
    """(second audit, part 1.2) The decoder does not remember a check that memory was too short to run: its next
    copy is made in the form asked for."""
    from exact_gpu_jpeg import _packed
    monkeypatch.setattr(_copycheck, "_state", {})
    real, calls = _packed.pack, []

    def pack(*a, **k):
        calls.append(1)
        if len(calls) == 1:                              # the check's own first pack
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 0, 0)
        return real(*a, **k)
    monkeypatch.setattr(_packed, "pack", pack)
    dec = egj.Decoder(keep_coefs="packed")
    data = _photo(320, 240, seed=3)
    dec.decode(data)
    h = dec.take_coefs()
    assert h is not None and h.form == "raw"             # not checked this time: the raw form
    dec.decode(data)
    h = dec.take_coefs()
    assert h is not None and h.form == "packed"          # was: raw for the decoder's whole life


def test_decode_from_handle_that_raises_early_leaves_no_older_handle():
    """(second audit, part 1.5) An argument error in decode_from_handle: take_coefs() then gives None, not the
    previous decode's handle."""
    dec = egj.Decoder(keep_coefs="raw")
    dec.decode(_photo(320, 240, seed=1))
    h = dec.take_coefs()
    dec.decode(_photo(320, 240, seed=2))
    with pytest.raises(ValueError):
        dec.decode_from_handle(h, "XYZ")
    assert dec.take_coefs() is None


def test_a_handle_is_in_gpu_memory_only_and_hashes_nothing(monkeypatch):
    """(0.1.0 scope) The forms are "raw" and "packed", both in GPU memory: no form that leaves the process, nothing
    that turns a handle into bytes or bytes into a handle, and keeping a handle does not hash the file."""
    from exact_gpu_jpeg import _repeat
    with pytest.raises(ValueError, match="keep_coefs"):
        egj.Decoder(keep_coefs="bytes").decode(_photo(64, 48))
    assert not hasattr(egj.CoefHandle, "to_bytes") and not hasattr(egj.CoefHandle, "from_bytes")
    calls = []
    real = _repeat.check_value
    monkeypatch.setattr(_repeat, "check_value", lambda d: calls.append(1) or real(d))
    dec = egj.Decoder(keep_coefs="packed")
    data = _photo(640, 480, seed=7)
    dec.decode(data)
    h = dec.take_coefs()
    assert h.form == "packed" and calls == []
    assert np.array_equal(egj.Decoder().decode_from_handle(h).get(), reference(data))
