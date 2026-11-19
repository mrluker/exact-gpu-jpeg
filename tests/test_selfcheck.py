"""The start-up self-check and what follows a GPU that cannot be trusted: a failed check, a kernel that does not
compile, a missing NVRTC or a CUDA error must end in the CPU decoder with one warning (decode with fallback), in a
decline with the same reason (Decoder.decode, fallback=False) or in a RuntimeError carrying it (encode); running out
of GPU memory and the guard-band test mode keep their own behaviour. Also: the memory estimate counts float output
and the guard bands."""
import base64
import hashlib
import io
import os
import sys
import warnings

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _batch as B  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from exact_gpu_jpeg import _encoder as E  # noqa: E402
from exact_gpu_jpeg import _selfcheck as S  # noqa: E402
from exact_gpu_jpeg import _selfcheck_data as DATA  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))


def _jpeg(w=160, h=120, seed=0):
    a = np.random.default_rng(seed).integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90)
    return b.getvalue()


@pytest.fixture(autouse=True)
def fresh_state():
    """Every test starts with no part checked and the GPU path on, and leaves the process as it found it."""
    saved = [(d, type(d)(d)) for d in (S.FAILED, S._done, S._props, S._checked)]
    for d, _ in saved:
        d.clear()
    yield
    for d, old in saved:
        d.clear()
        d.update(old)


def _boom(exc):
    def k(*a, **kw):
        raise exc
    return k


def test_the_check_passes_on_this_gpu():
    r = egj.self_check()
    assert r["ok"] and r["reason"] is None, r
    assert set(r["ms"]) == {"decode", "batch", "encode", "fast_dct", "fast_upsample", "output"}
    assert all(v is not None for v in r["ms"].values())
    assert ", driver " in r["gpu"] or r["gpu"].startswith("CUDA device")
    assert egj.self_check()["ok"]                      # again: nothing runs twice
    print("self-check:", r)


def test_the_stored_cases_are_what_the_reference_libraries_give():
    simplejpeg = pytest.importorskip("simplejpeg")
    for name, cs, sha, b64 in DATA.DECODE:
        a = simplejpeg.decode_jpeg(base64.b64decode(b64), colorspace=cs, fastdct=False, fastupsample=False)
        assert hashlib.sha256(a.tobytes()).hexdigest() == sha, name
    for sha, b64 in DATA.BATCH[1]:
        a = simplejpeg.decode_jpeg(base64.b64decode(b64), colorspace="RGB", fastdct=False, fastupsample=False)
        assert hashlib.sha256(a.tobytes()).hexdigest() == sha
    for name, w, h, q, ss, sha in DATA.ENCODE:
        b = io.BytesIO()
        Image.fromarray(S.synthetic(w, h)).save(b, "JPEG", quality=q, subsampling={"444": 0, "422": 1, "420": 2}[ss])
        assert hashlib.sha256(b.getvalue()).hexdigest() == sha, name
    # the shared image rule, by hand: v = (x*7 + y*13 + c*61 + ((x*y) >> 3)) & 255, +-40 on a 16-pixel checker, clamped
    a = S.synthetic(40, 40)
    for x, y, c in ((0, 0, 0), (3, 2, 1), (17, 5, 2), (33, 20, 0), (39, 39, 2)):
        v = (x * 7 + y * 13 + c * 61 + ((x * y) >> 3)) & 255
        v += 40 if (((x >> 4) + (y >> 4)) & 1) == 0 else -40
        assert a[y, x, c] == min(max(v, 0), 255)
    assert np.array_equal(S.synthetic(40, 40, gray=True), a[..., 0])


def test_the_committed_data_file_is_current():
    import PIL
    simplejpeg = pytest.importorskip("simplejpeg")
    if not (PIL.__version__ == "13.0.0" and getattr(simplejpeg, "__version__", "") == "1.9.0"):
        pytest.skip("the data file is written with Pillow 13.0.0 and simplejpeg 1.9.0")
    import make_selfcheck
    assert open(make_selfcheck.OUT, encoding="utf-8").read() == make_selfcheck.build(), "run tools/make_selfcheck.py"


def test_a_failed_check_ends_in_the_cpu_decoder_with_one_warning(monkeypatch):
    monkeypatch.setenv("EGJ_SELFCHECK_FAIL", "1")
    data = _jpeg()
    ref = egj.decode_cpu(data)
    info = {}
    with pytest.warns(RuntimeWarning, match=r"self-check failed on .+: decode 4:2:0 with restart markers") as w:
        img = egj.decode(data, cpu_below=0, info=info)
    assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 1
    assert np.array_equal(img.get(), ref) and info["path"] == "cpu"
    reason = info["reason"]
    assert reason.startswith("self-check failed on ") and reason.endswith("(forced by EGJ_SELFCHECK_FAIL)")
    with warnings.catch_warnings():                    # no second warning, whatever is called next
        warnings.simplefilter("error")
        assert np.array_equal(egj.decode(data, cpu_below=0).get(), ref)
        info = {}
        assert egj.decode(data, cpu_below=0, fallback=False, info=info) is None and info == dict(path=None, reason=reason, warning=None, warnings_checked=None)
        dec = egj.Decoder()
        assert dec.decode(data) is None and dec.reason == reason
        assert dec.upload_ahead(data) is False and dec.reason == reason
        assert dec.decode_async(data) is None and dec.reason == reason
        infos = []
        imgs = egj.decode([data, data], info=infos)
        assert all(np.array_equal(i.get(), ref) for i in imgs) and infos == [dict(path="cpu", reason=reason, warning=None, warnings_checked=True)] * 2
        assert egj.decode([data, data], fallback=False) == [None, None]
        with pytest.raises(RuntimeError, match="exact_gpu_jpeg: self-check failed on "):
            egj.encode(np.zeros((16, 16, 3), np.uint8))
        r = egj.self_check()
        assert r["ok"] is False and r["reason"] == reason


@pytest.mark.parametrize("exc", [
    cp.cuda.compiler.CompileException("error: no such intrinsic", "source", "kernels.cu", ("--std=c++14",)),
    RuntimeError("CuPy failed to load nvrtc64_120_0.dll: FileNotFoundError"),
    ImportError("DLL load failed while importing nvrtc"),
], ids=["kernel does not compile", "NVRTC missing", "NVRTC import fails"])
def test_a_kernel_that_cannot_be_built_fails_the_check(monkeypatch, exc):
    monkeypatch.setattr(D, "_k", _boom(exc))
    data = _jpeg()
    info = {}
    # one prefix for everything that goes wrong inside the check: the case, then the error in brackets
    with pytest.warns(RuntimeWarning, match=rf"self-check failed on .+: decode 4:2:0 with restart markers "
                                             rf"\({type(exc).__name__}: ") as w:
        img = egj.decode(data, cpu_below=0, info=info)
    assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 1
    assert np.array_equal(img.get(), egj.decode_cpu(data)) and info["path"] == "cpu"
    dec = egj.Decoder()
    assert dec.decode(data) is None and dec.reason == info["reason"]
    assert egj.self_check()["ok"] is False and egj.device_report()["gpu_path"] == "off: " + info["reason"]


def test_a_cuda_error_after_the_check_turns_the_gpu_path_off(monkeypatch):
    assert egj.self_check()["ok"]
    data = _jpeg()
    ref = egj.decode_cpu(data)
    assert np.array_equal(egj.decode(data, cpu_below=0).get(), ref)
    monkeypatch.setattr(D, "_k", _boom(cp.cuda.runtime.CUDARuntimeError(700)))      # cudaErrorIllegalAddress
    info = {}
    with pytest.warns(RuntimeWarning, match=r"GPU error on .+: CUDARuntimeError") as w:
        img = egj.decode(data, cpu_below=0, info=info)
    assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 1
    assert np.array_equal(img.get(), ref) and info["path"] == "cpu" and info["reason"].startswith("GPU error on ")
    monkeypatch.undo()                                 # the kernels work again, but the GPU path stays off
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dec = egj.Decoder()
        assert dec.decode(data) is None and dec.reason == info["reason"]
        assert egj.decode(data, cpu_below=0, fallback=False) is None
        with pytest.raises(RuntimeError, match="exact_gpu_jpeg: GPU error on "):
            egj.encode(np.zeros((16, 16, 3), np.uint8))


def test_a_cuda_error_in_a_batch_and_in_an_encode(monkeypatch):
    assert egj.self_check()["ok"]
    datas = [_jpeg(seed=k) for k in range(3)]
    refs = [egj.decode_cpu(d) for d in datas]
    monkeypatch.setattr(B, "_bk", _boom(cp.cuda.runtime.CUDARuntimeError(700)))
    infos = []
    with pytest.warns(RuntimeWarning, match=r"GPU error on .+: CUDARuntimeError"):
        imgs = egj.decode(datas, info=infos)
    assert all(np.array_equal(i.get(), r) for i, r in zip(imgs, refs))
    assert all(i["path"] == "cpu" and i["reason"].startswith("GPU error on ") for i in infos)
    monkeypatch.undo()
    S.FAILED.clear()
    monkeypatch.setattr(E, "_k", _boom(cp.cuda.runtime.CUDARuntimeError(700)))
    with pytest.warns(RuntimeWarning, match=r"GPU error on "):
        with pytest.raises(RuntimeError, match="exact_gpu_jpeg: GPU error on .+: CUDARuntimeError"):
            egj.encode(np.zeros((16, 16, 3), np.uint8))


def test_out_of_memory_and_guard_bands_keep_their_behaviour(monkeypatch):
    assert egj.self_check()["ok"]
    data = _jpeg()
    dec = egj.Decoder()
    monkeypatch.setattr(D.Decoder, "_decode", _boom(cp.cuda.memory.OutOfMemoryError(1 << 30, 1 << 20)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dec.decode(data) is None and dec.reason == "out of GPU memory"
    assert not S.FAILED                                 # one file declined; the GPU path stays on
    monkeypatch.undo()
    assert np.array_equal(dec.decode(data).get(), egj.decode_cpu(data))
    monkeypatch.setattr(D.Decoder, "_decode", _boom(S.GuardBandError("exact_gpu_jpeg: a kernel wrote outside device buffer #3")))
    with pytest.raises(S.GuardBandError):
        dec.decode(data)
    assert not S.FAILED


def test_the_memory_estimate_counts_float_output_and_guard_bands():
    assert egj.self_check()["ok"]
    big = _jpeg(1024, 1024)
    ref = egj.decode_cpu(big)
    dec = egj.Decoder(max_gpu_bytes=10 << 20, canary=False)   # room for uint8 output (3 MB): float16 is 6, float32 12
    assert np.array_equal(dec.decode(big).get(), ref)
    assert dec.decode(big, dtype="float32") is None and dec.reason.startswith("needs about")
    assert dec.decode(big, dtype="float16") is None and dec.reason.startswith("needs about")
    roomy = egj.Decoder()
    assert np.array_equal(roomy.decode(big, dtype="float32").get(), ref.astype(np.float32))
    small = _jpeg(64, 48)
    assert egj.Decoder(max_gpu_bytes=2 << 20, canary=False).decode(small) is not None
    guarded = egj.Decoder(max_gpu_bytes=2 << 20, canary=True)      # the guard bands alone are about 3 MB
    assert guarded.decode(small) is None and guarded.reason.startswith("needs about")
    assert np.array_equal(egj.Decoder(canary=True).decode(small).get(), egj.decode_cpu(small))


# ---- the device check and the settings derived from what the device reports ----

GB = 1 << 30


def _pretend(monkeypatch, total_gb, capability=(6, 1), nvrtc="12.9", free=None):
    """Make the library see another card: what it reports is replaced, the real GPU does the work."""
    real_free = S.free_memory(cp.cuda.Device().id)
    total = int(total_gb * GB)
    free = int(0.9 * total) if free is None else free
    monkeypatch.setattr(S, "detect", lambda dev: dict(compute_capability=capability, total_memory=total,
                                                      free_memory=free, cuda_runtime="12.9", nvrtc=nvrtc))
    monkeypatch.setattr(S, "free_memory", lambda dev: free)
    for d in (S._props, S._checked, S._done, S.FAILED):
        d.clear()
    return real_free


def test_the_device_report():
    r = egj.device_report()
    for k in ("device", "gpu", "compute_capability", "total_memory_mb", "free_memory_mb", "cuda_runtime", "nvrtc",
              "minimum_compute_capability", "derived_settings", "self_check", "gpu_path"):
        assert k in r, k
    assert r["gpu_path"] == "on" and r["self_check"] == dict(decode="not run yet", batch="not run yet", encode="not run yet",
                                                             fast_dct="not run yet", fast_upsample="not run yet",
                                                             output="not run yet")
    assert egj.self_check()["ok"]
    r2 = egj.device_report()
    assert r2["self_check"] == dict(decode="passed", batch="passed", encode="passed", fast_dct="passed",
                                    fast_upsample="passed", output="passed")
    assert r2["derived_settings"] == r["derived_settings"]            # the same card: the same settings, every time
    print("device report:", r2)


@pytest.mark.parametrize("total_gb, batch_mb", [(2, 16), (4, 32), (8, 64), (24, 64), (48, 64), (0.5, 8)])
def test_every_pretended_card_gives_the_same_bytes(monkeypatch, total_gb, batch_mb):
    _pretend(monkeypatch, total_gb)
    dev = cp.cuda.Device().id
    assert S.settings(dev) == dict(memory_budget_share=0.9, batch_data_bytes=batch_mb << 20)
    assert S.settings(dev) == S.settings(dev)
    assert egj.self_check()["ok"]                                    # the built-in cases, under these settings
    datas = [_jpeg(160 + 16 * k, 120 + 8 * k, seed=k) for k in range(6)] + [_jpeg(1024, 768, seed=9)]
    refs = [egj.decode_cpu(d) for d in datas]
    infos = []
    imgs = egj.decode(datas, info=infos)                             # a batch, split by the derived limits
    assert all(np.array_equal(i.get(), r) for i, r in zip(imgs, refs)) and all(i["path"] == "gpu" for i in infos)
    for d, r in zip(datas, refs):                                    # and one at a time
        info = {}
        assert np.array_equal(egj.decode(d, cpu_below=0, info=info).get(), r) and info["path"] == "gpu"
    a = np.random.default_rng(3).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=75, subsampling=2)
    assert egj.encode(a) == b.getvalue()


def test_a_card_with_little_free_memory_declines_large_images_to_the_cpu(monkeypatch):
    _pretend(monkeypatch, 2, free=24 << 20)                          # 24 MB free: the budget is 21.6 MB
    import gc
    gc.collect()                                                     # (nothing an earlier test left may be freed in here:
    cp.get_default_memory_pool().free_all_blocks()                   # the budget counts what the pool holds)
    big, small = _jpeg(3072, 2304), _jpeg(320, 240)
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                               # a decline for memory is not a warning
        img = egj.decode(big, info=info)
    assert np.array_equal(img.get(), egj.decode_cpu(big)) and info["path"] == "cpu"
    assert info["reason"].startswith("needs about") and "more than allowed" in info["reason"]
    assert egj.decode(big, fallback=False) is None
    info = {}
    assert np.array_equal(egj.decode(small, cpu_below=0, info=info).get(), egj.decode_cpu(small)) and info["path"] == "gpu"
    infos = []
    imgs = egj.decode([small, big, small], info=infos)               # in a batch: the small ones on the GPU
    assert [i["path"] for i in infos] == ["gpu", "cpu", "gpu"]
    assert all(np.array_equal(i.get(), egj.decode_cpu(d)) for i, d in zip(imgs, [small, big, small]))


@pytest.mark.parametrize("kw, what", [
    (dict(capability=(3, 7)), "compute capability 3.7 is below 5.0"),
    (dict(nvrtc=None), r"the CUDA kernel compiler \(NVRTC\) could not be loaded"),
], ids=["below the minimum", "no kernel compiler"])
def test_a_card_that_does_not_meet_the_requirements(monkeypatch, kw, what):
    _pretend(monkeypatch, 8, **kw)
    data = _jpeg()
    ref = egj.decode_cpu(data)
    info = {}
    with pytest.warns(RuntimeWarning, match=rf"device check failed on .+: {what}") as w:
        img = egj.decode(data, cpu_below=0, info=info)
    assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 1
    assert np.array_equal(img.get(), ref) and info["path"] == "cpu" and info["reason"].startswith("device check failed on ")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dec = egj.Decoder()
        assert dec.decode(data) is None and dec.reason == info["reason"]
        assert egj.decode(data, cpu_below=0, fallback=False) is None
        assert [np.array_equal(i.get(), ref) for i in egj.decode([data, data])] == [True, True]
        with pytest.raises(RuntimeError, match="exact_gpu_jpeg: device check failed on "):
            egj.encode(np.zeros((16, 16, 3), np.uint8))
        r = egj.device_report()
        assert r["gpu_path"] == "off: " + info["reason"] and egj.self_check()["ok"] is False


def test_a_device_that_cannot_be_queried(monkeypatch):
    def broken(dev):
        raise cp.cuda.runtime.CUDARuntimeError(100)                  # cudaErrorNoDevice
    monkeypatch.setattr(S, "detect", broken)
    data = _jpeg()
    with pytest.warns(RuntimeWarning, match=r"device check failed on .+: the device could not be queried"):
        dec = egj.Decoder()
        assert dec.decode(data) is None
    assert "detect_error" in egj.device_report()


def test_an_error_inside_the_batch_and_encode_parts_of_the_check(monkeypatch):
    """The same prefix when the error comes while the batch or the encode cases run."""
    err = cp.cuda.compiler.CompileException("error: no such intrinsic", "source", "kernels.cu", ("--std=c++14",))
    assert S.ensure(cp.cuda.Device().id, "decode") is None
    monkeypatch.setattr(B, "_bk", _boom(err))
    with pytest.warns(RuntimeWarning, match=r"self-check failed on .+: decode batch of two \(CompileException: "):
        imgs = egj.decode([_jpeg(), _jpeg(seed=1)])
    assert all(i is not None for i in imgs)                          # the CPU decoder's pixels
    monkeypatch.undo()
    for d in (S.FAILED, S._done, S._props, S._checked):
        d.clear()
    monkeypatch.setattr(E, "_k", _boom(err))
    with pytest.warns(RuntimeWarning, match=r"self-check failed on .+: encode few-launch \(CompileException: "):
        with pytest.raises(RuntimeError, match="exact_gpu_jpeg: self-check failed on "):
            egj.encode(np.zeros((16, 16, 3), np.uint8))


def test_the_warm_up_call_leaves_nothing_for_a_first_large_encode_to_prepare(monkeypatch):
    """An encode of more than SCAN_BLOCKS blocks uses CuPy's own running sum, which CuPy prepares the first time
    it is used (about 45 ms, once per process). egj.self_check() has already used it in the same form (its batch
    part), so a program that calls it at start-up does not pay that in its first large encode; small work alone
    never touches it."""
    sums = []
    real = cp.cumsum

    def cumsum(a, *args, **kw):
        sums.append((str(a.dtype), str(np.dtype(kw["dtype"])) if kw.get("dtype") else None, kw.get("out") is not None))
        return real(a, *args, **kw)
    monkeypatch.setattr(cp, "cumsum", cumsum)
    assert egj.decode(_jpeg(640, 480), cpu_below=0) is not None      # the automatic check: decode part only
    assert len(egj.encode(cp.asarray(np.zeros((64, 64, 3), np.uint8)))) > 100
    assert not sums                                                  # small work has not touched the running sum
    assert egj.self_check()["ok"]
    prepared = set(sums)
    assert prepared, "the self-check no longer uses the running sum: prepare it there"
    del sums[:]
    big = cp.zeros((1000, 1504, 3), cp.uint8)                        # 70,500 blocks at 4:4:4: above SCAN_BLOCKS
    data = egj.encode(big, quality=90, subsampling="444")
    assert sums and set(sums) <= prepared, (sums, prepared)          # a real large encode needs nothing new
    b = io.BytesIO()
    Image.fromarray(np.zeros((1000, 1504, 3), np.uint8)).save(b, "JPEG", quality=90, subsampling=0)
    assert data == b.getvalue()


def test_the_large_image_routes_give_the_same_bytes(corpus):
    """The routes that only large inputs take (a file of more than SMALL_TABLE_CHUNKS chunks, an encode of more
    than SCAN_BLOCKS blocks), forced on ordinary inputs: every corpus file the GPU decodes, and encodes of several
    shapes."""
    from conftest import reference
    dec, plain = egj.Decoder(), egj.Decoder()
    dec._large_route = True
    n = 0
    for name, data, _ in corpus:
        want = plain.decode(data)
        got = dec.decode(data)
        assert (got is None) == (want is None), name
        if got is not None:
            assert np.array_equal(got.get(), reference(data)), name
            n += 1
    assert n >= 100
    enc = egj.Encoder()
    enc._scan_blocks = 0
    rng = np.random.default_rng(5)
    for (w, h), ss, sub in (((600, 592), "444", 0), ((1600, 1200), "420", 2), ((1224, 816), "422", 1), ((1001, 701), "420", 2)):
        a = np.asarray(Image.fromarray(rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC))
        b = io.BytesIO()
        Image.fromarray(a).save(b, "JPEG", quality=85, subsampling=sub)
        assert enc.encode(cp.asarray(a), quality=85, subsampling=ss) == b.getvalue(), (w, h, ss)

