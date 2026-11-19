"""Faults found by the independent audit of the release candidate (2026-11-18), one test each: every test here
fails on a54cd05 and passes with the fix. Grouped by where the fault was."""
import io
import os
import random
import sys
import time

import numpy as np
import pytest
from PIL import Image, ImageFile

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from exact_gpu_jpeg import _header  # noqa: E402
from exact_gpu_jpeg import _selfcheck  # noqa: E402
from conftest import reference  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fuzz import mutate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _photo(w, h, seed=1, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=kw.pop("quality", 90), **kw)
    return b.getvalue()


def _host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


# ---- a queued job and the shared device tables ----

def test_a_queued_job_holds_the_device_tables_its_kernels_read(monkeypatch):
    """The tables of an image layout live in a cache shared by every decoder; another decoder can clear it while
    a job's kernels are still queued. The job keeps them until result() has waited."""
    data = _photo(1600, 1200, seed=2)
    want = reference(data)
    other = egj.Decoder()
    dec = egj.Decoder(canary=True)
    for k in range(6):
        with D._dev_lock:
            D._dev_cache.clear()
        other.decode(data)                               # every round: the layout built on the other's stream
        job = dec.decode_async(data)
        assert job._q.tables is not None and job._q.tables["pix"] is not None
        with D._dev_lock:
            D._dev_cache.clear()                         # what a 65th layout does, from any decoder
        for j in range(4):                               # the other decoder takes memory and writes it meanwhile
            other.decode(_photo(64 + 8 * j + k, 48, seed=j))
        img = job.result()
        assert img is not None and np.array_equal(img.get(), want), (k, dec.reason)
        assert job._q.tables is None                     # let go once the job has finished


def test_the_fast_idct_table_is_the_one_the_layout_keeps(monkeypatch):
    """Two threads that both make a layout's fast-IDCT table: the one that finishes second uses the table the
    layout kept (the first one's), not its own, which nothing else holds. Made to happen: while this call builds
    its table, "another thread" stores one."""
    data = _photo(640, 480, seed=3)
    egj.Decoder(fast_dct=True).decode(data)
    h = D.parse(data)
    T = D._device_layout(h, True)
    T.pop("qf", None)
    theirs = cp.zeros(3 * 64, cp.int32)
    real = D.ifast_table

    def racing(q):
        r = real(q)
        T["qf"] = theirs                                 # the other thread got there first
        return r
    monkeypatch.setattr(D, "ifast_table", racing)
    assert D._ifast_q(T, h) is theirs and T["qf"] is theirs     # (was: its own table, while T kept the other)
    T.pop("qf")


# ---- a queued job's own outcome ----

def _declined_after_queueing(seed=5):
    """A damaged file whose header is fine and whose data is found bad only by the GPU work."""
    rng = random.Random(seed)
    base = _photo(320, 240, seed=4)
    dec = egj.Decoder()
    scan = D.parse(base).scan_start
    for _ in range(3000):
        d = bytearray(base)
        for _ in range(3):
            d[rng.randrange(scan + 8, len(d) - 8)] = rng.randrange(256)
        d = bytes(d)
        if dec.decode(d) is None and dec.reason and dec.reason.split("; ")[0] in D.ERRORS.values():
            return d, dec.reason
    pytest.skip("no damaged file declined by the GPU work was found")


def test_a_job_keeps_its_own_reason_when_the_next_call_concludes_it():
    bad, why = _declined_after_queueing()
    dec = egj.Decoder()
    job = dec.decode_async(bad)
    assert job is not None
    good = dec.decode(_photo(64, 48))                    # concludes the job, and has a reason of its own (None)
    assert good is not None and dec.reason is None
    assert job.result() is None and job.reason is not None


def test_an_exception_from_result_is_raised_again(monkeypatch):
    dec = egj.Decoder()
    job = dec.decode_async(_photo(320, 240))

    def boom(*a, **k):
        raise _selfcheck.GuardBandError("test")
    monkeypatch.setattr(dec, "_conclude", boom)
    with pytest.raises(_selfcheck.GuardBandError):
        job.result()
    with pytest.raises(_selfcheck.GuardBandError):       # not a quiet None the second time
        job.result()


# ---- upload_ahead ----

def test_a_staged_relative_path_is_the_file_it_named_when_staged(tmp_path, monkeypatch):
    a, b = _photo(320, 240, seed=7), _photo(320, 240, seed=8)
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    (tmp_path / "x" / "p.jpg").write_bytes(a)
    (tmp_path / "y" / "p.jpg").write_bytes(b)
    dec = egj.Decoder()
    monkeypatch.chdir(tmp_path / "x")
    assert dec.upload_ahead("p.jpg")
    monkeypatch.chdir(tmp_path / "y")
    got = dec.decode("p.jpg")                            # names y/p.jpg now: not the staged x/p.jpg
    assert np.array_equal(got.get(), reference(b))
    assert dec.stats.get("read") != "staged"


def test_upload_ahead_says_why_it_declines():
    dec = egj.Decoder(max_pixels=1000)
    assert dec.upload_ahead(_photo(320, 240)) is False
    assert dec.reason == "more than max_pixels (1000)"


# ---- reading files: limits before buffers ----

def test_a_header_of_many_tiny_segments_is_declined_at_once(tmp_path):
    good = _photo(320, 240)
    junk = b"\xff\xfe\x00\x02" * 300_000                 # 1.2 MB of empty comments, then a real header
    data = good[:2] + junk + good[2:]
    p = tmp_path / "many.jpg"
    p.write_bytes(data)
    dec = egj.Decoder(read="streamed")
    t = time.perf_counter()
    assert dec.decode(str(p)) is None and "marker segments before the scan" in dec.reason
    assert dec.decode(data) is None and "marker segments before the scan" in dec.reason
    assert time.perf_counter() - t < 2.0
    info = {}
    img = egj.decode(data, info=info)                    # the CPU decoder still reads it (libjpeg-turbo skips COM)
    assert info["path"] == "cpu" and np.array_equal(_host(img), reference(data))


def test_a_long_run_of_fill_bytes_costs_what_reading_it_costs():
    good = _photo(320, 240)
    i = D.parse(good).scan_start
    s = good.index(b"\xff\xda")
    data = good[:s] + b"\xff" * (8 << 20) + good[s:]     # 8 MB of fill bytes before SOS: valid
    t = time.perf_counter()
    h = _header.parse(data)
    assert time.perf_counter() - t < 1.0 and h.scan_start == i + (8 << 20)
    tail = good[:-2] + b"\xff" * (8 << 20) + b"\xff\xd9"
    t = time.perf_counter()
    assert _header.check_trailer(tail, len(good) - 2) is None
    assert time.perf_counter() - t < 1.0


def test_a_header_longer_than_the_first_piece_is_still_read_in_pieces(tmp_path):
    """A real header of several MB (many APP segments): the streamed read reads on until the header is whole."""
    good = _photo(640, 480, seed=9)
    app = b"\xff\xe2" + (65535).to_bytes(2, "big") + b"\x00" * 65533
    data = good[:2] + app * 60 + good[2:]                # about 3.9 MB of APP2 before the frame
    p = tmp_path / "big_header.jpg"
    p.write_bytes(data)
    dec = egj.Decoder(read="streamed")
    img = dec.decode(str(p))
    assert img is not None and dec.stats["read"] == "streamed", dec.reason
    assert np.array_equal(img.get(), reference(good))


def test_a_file_over_the_limit_is_declined_before_any_buffer(tmp_path, monkeypatch):
    egj.self_check()                                     # (the first-use check is not what is tested here)
    data = _photo(320, 240)
    p = tmp_path / "f.jpg"
    p.write_bytes(data)
    monkeypatch.setattr(D, "MAX_FILE", len(data) - 1)
    for read in ("streamed", "whole"):
        dec = egj.Decoder(read=read)
        assert dec.decode(str(p)) is None and "larger than 1 GB" in dec.reason
        assert dec._pin is None                          # nothing was page-locked for it
    dec = egj.Decoder()
    assert dec.decode(data) is None and "larger than 1 GB" in dec.reason
    assert dec.decode_batch([data, data]) == [None, None] and all("larger than 1 GB" in r for r in dec.reasons)


def test_page_locked_memory_that_cannot_be_had_declines_the_file(monkeypatch):
    egj.self_check()                                     # (the first-use check is not what is tested here)
    data = _photo(320, 240)

    def no(self, size):
        raise cp.cuda.memory.OutOfMemoryError(size, 0, 0)
    monkeypatch.setattr(D.Decoder, "_pin_new", no)
    dec = egj.Decoder()
    assert dec.decode(data) is None and "page-lock" in dec.reason
    import threading
    res = {}

    def top():                                           # a new thread: a decoder with no page-locked buffer yet
        info = {}
        res["img"] = _host(egj.decode(data, info=info, cpu_below=0))
        res["path"] = info["path"]
    t = threading.Thread(target=top)
    t.start()
    t.join()
    assert res["path"] == "cpu" and np.array_equal(res["img"], reference(data))   # the top level uses the CPU


def test_a_large_file_that_is_not_a_jpeg_is_declined_before_it_is_page_locked(tmp_path, monkeypatch):
    """(second audit, part 3.5) Its header is read into ordinary memory first."""
    egj.self_check()
    monkeypatch.setattr(D.Decoder, "PEEK_ABOVE", 1 << 19)
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00ftypmp42" + bytes(3 << 20))
    for read in ("streamed", "whole"):
        dec = egj.Decoder(read=read)
        assert dec.decode(str(p)) is None and "not a JPEG" in dec.reason
        assert dec._pin is None                          # nothing was page-locked for it
    q = tmp_path / "big.jpg"
    q.write_bytes(_photo(2000, 1500))                    # a JPEG above the limit still decodes
    assert q.stat().st_size > 1 << 19
    dec = egj.Decoder()
    img = dec.decode(str(q))
    assert img is not None and np.array_equal(img.get(), reference(q.read_bytes()))


def test_a_page_locked_buffer_made_for_a_large_file_is_given_back(tmp_path, monkeypatch):
    """(second audit, part 3.5) The buffer only grew; one far larger than the file at hand is now given back."""
    egj.self_check()
    monkeypatch.setattr(D.Decoder, "KEEP_PINNED", 1 << 20)
    big, small = _photo(2000, 1500), _photo(320, 240)
    dec = egj.Decoder()
    assert dec.decode(big) is not None
    grown = dec._pin.size
    assert grown > 4 * len(small)
    img = dec.decode(small)
    assert img is not None and np.array_equal(img.get(), reference(small)) and dec._pin.size < grown


# ---- the first-use self-check and GPU memory ----

def test_out_of_memory_during_the_self_check_declines_that_call_only(monkeypatch):
    dev = cp.cuda.Device().id
    monkeypatch.setattr(_selfcheck, "FAILED", {})
    monkeypatch.setattr(_selfcheck, "_done", set())
    calls = []
    real = _selfcheck._run

    def run(d, part):
        calls.append(part)
        if len(calls) == 1:
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 0, 0)
        return real(d, part)
    monkeypatch.setattr(_selfcheck, "_run", run)
    why = _selfcheck.ensure(dev, "decode")
    assert why is not None and why.startswith("out of GPU memory")
    assert _selfcheck.failed(dev) is None and (dev, "decode") not in _selfcheck._done
    assert _selfcheck.ensure(dev, "decode") is None       # the next call runs the check again, and it passes
    assert calls == ["decode", "decode"]


def test_a_memory_decline_inside_the_self_check_is_not_a_failed_check(monkeypatch):
    dev = cp.cuda.Device().id
    monkeypatch.setattr(_selfcheck, "FAILED", {})
    monkeypatch.setattr(_selfcheck, "_done", set())
    monkeypatch.setattr(D.Decoder, "_decode", lambda self, *a, **k: setattr(self, "reason", "out of GPU memory"))
    why = _selfcheck.ensure(dev, "decode")
    assert why.startswith("out of GPU memory") and _selfcheck.failed(dev) is None


# ---- the CPU stand-in ----

def test_the_cpu_decoder_takes_jpeg_files_only():
    b = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(b, "PNG")
    for strict in (True, False):                         # (the documented kinds of error; was KeyError('MPO'))
        with pytest.raises((ValueError, OSError)):
            egj.decode_cpu(b.getvalue(), strict=strict)
        with pytest.raises((ValueError, OSError)):
            egj.decode(b.getvalue(), strict=strict)


def test_pillows_truncated_image_switch_does_not_change_what_the_cpu_decoder_accepts():
    bad = open(os.path.join(HERE, "regressions", "dqt_after_scan.jpg"), "rb").read()
    was = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True           # as training code often sets it
        with pytest.raises(Exception):
            egj.decode_cpu(bad)
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True   # put back
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = was


def test_cpu_route_output_options_are_the_gpu_routes():
    data = _photo(200, 150, seed=11)                     # below the CPU-route threshold
    kw = dict(dtype="float16", layout="CHW", scale=[0.017, 0.018, 0.019], bias=[-2.1, -2.0, -1.8])
    info = {}
    a = egj.decode(data, info=info, **kw)
    assert info["path"] == "cpu"
    b = egj.decode(data, cpu_below=0, **kw)
    assert a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())


def test_decode_cpu_checks_the_colour_space():
    with pytest.raises(ValueError):
        egj.decode_cpu(_photo(64, 48), "CMYK")


# ---- lists ----

def test_a_list_call_takes_a_list_for_info():
    with pytest.raises(TypeError):
        egj.decode([_photo(64, 48)], info={})


def test_a_file_the_cpu_cannot_read_is_named_and_info_is_filled():
    b = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(b, "PNG")
    files = [_photo(320, 240, seed=1), b.getvalue(), _photo(320, 240, seed=2)]
    info = []
    with pytest.raises(Exception, match="file 1"):
        egj.decode(files, info=info)
    assert len(info) == 3 and info[0]["path"] == "gpu" and info[2]["path"] == "gpu"


def test_decode_batch_checks_the_colour_space():
    with pytest.raises(ValueError):
        egj.Decoder().decode_batch([_photo(64, 48)], "RGBA")


def test_a_batch_does_not_use_or_fill_the_repeat_cache():
    from exact_gpu_jpeg import RepeatCache
    c = RepeatCache()
    dec = egj.Decoder(cache=c)
    a = _photo(640, 480, seed=1)
    b = _photo(640, 480, seed=2, subsampling=0)          # other tables: each goes one by one
    got = dec.decode_batch([a, b])
    assert np.array_equal(got[0].get(), reference(a)) and np.array_equal(got[1].get(), reference(b))
    s = c.stats() if callable(c.stats) else c.stats
    assert s["made"] == 0 and s["used"] == 0


def test_out_of_memory_for_a_groups_tables_sends_it_one_by_one(monkeypatch):
    from exact_gpu_jpeg import _batch
    files = [_photo(320, 240, seed=s) for s in range(4)]

    def no(*a, **k):
        raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)
    monkeypatch.setattr(_batch, "_group_tables", no)
    dec = egj.Decoder()
    got = dec.decode_batch(files)
    assert dec.stats["batched"] == 0
    for d, g in zip(files, got):
        assert g is not None and np.array_equal(g.get(), reference(d))


def test_a_misaligned_out_array_is_refused():
    """A float32 array at an address that is not a multiple of 4 would make the colour kernel fault (and switch
    the GPU path off for the process): it is refused like a wrong shape."""
    data = _photo(64, 48)
    raw = cp.empty(64 * 48 * 3 * 4 + 4, cp.uint8)
    mis = cp.ndarray((48, 64, 3), cp.float32, cp.cuda.MemoryPointer(raw.data.mem, raw.data.ptr - raw.data.mem.ptr + 1))
    with pytest.raises(ValueError, match="aligned"):
        egj.Decoder().decode_async(data, out=mis, dtype="float32").result()
    ok = cp.empty((48, 64, 3), cp.float32)
    img = egj.Decoder().decode_async(data, out=ok, dtype="float32").result()
    assert img is ok and np.array_equal(ok.get(), reference(data).astype(np.float32))


def test_a_missing_file_in_a_list_is_named_and_info_is_filled(tmp_path):
    """(second audit, part 3.12) One path that cannot be read: Decoder.decode_batch declines that file; the
    top-level call raises naming its index and path, with info filled for every file."""
    good = tmp_path / "a.jpg"
    good.write_bytes(_photo(320, 240))
    missing = str(tmp_path / "missing.jpg")
    dec = egj.Decoder()
    got = dec.decode_batch([str(good), missing, str(good)])
    assert got[0] is not None and got[1] is None and "could not read the file" in dec.reasons[1]
    info = []
    with pytest.raises(FileNotFoundError, match="file 1"):
        egj.decode([str(good), missing], info=info)
    assert len(info) == 2 and info[0]["path"] == "gpu" and info[1]["path"] is None
