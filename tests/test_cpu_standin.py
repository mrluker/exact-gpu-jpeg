"""A CPU decoder standing in for the GPU never changes a byte and never changes the mode: every way the CPU decoder
can be used (the small-image route, a declined file, a failed self-check, a failed device check, too little GPU
memory, a batch, a decoder of the caller's own) against every option that changes the output (fast_dct,
fast_upsample, colourspace, dtype / layout / scale / bias), with simplejpeg installed and without it. Either the
bytes equal libjpeg-turbo's in the mode asked for, or no pixels are returned and the caller is told."""
import io
import sys

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")
simplejpeg = pytest.importorskip("simplejpeg")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _selfcheck as S  # noqa: E402

MODES = [dict(), dict(fast_dct=True), dict(fast_upsample=True)]
OUTPUTS = [dict(colorspace="RGB"), dict(colorspace="BGR"), dict(colorspace="GRAY"),
           dict(colorspace="RGB", dtype="float32", layout="CHW", scale=(1 / 58.4, 1 / 57.1, 1 / 57.4), bias=(-2.1, -2.0, -1.8)),
           dict(colorspace="GRAY", dtype="float16", scale=1 / 255)]


def _jpeg(w, h, seed=0, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


SMALL, LARGE = _jpeg(200, 136), _jpeg(640, 480, 1)                   # 27,200 and 307,200 pixels
PROGRESSIVE = _jpeg(320, 240, 2, progressive=True)                   # the GPU path declines it


def want(data, mode, out):
    """libjpeg-turbo's pixels in the mode asked for, with the output options applied as NumPy does."""
    a = simplejpeg.decode_jpeg(data, colorspace=out["colorspace"], fastdct=bool(mode.get("fast_dct")),
                               fastupsample=bool(mode.get("fast_upsample")))
    if "dtype" not in out:
        return a
    v = a.astype(np.float32)
    if out.get("scale") is not None:
        v = v * np.asarray(out["scale"], np.float32)
    if out.get("bias") is not None:
        v = v + np.asarray(out["bias"], np.float32)
    v = v.astype(out["dtype"])
    return np.ascontiguousarray(v.transpose(2, 0, 1)) if out.get("layout") == "CHW" else v


@pytest.fixture
def fresh():
    saved = [(d, type(d)(d)) for d in (S.FAILED, S._done, S._props, S._checked)]
    yield
    for d, old in saved:
        d.clear()
        d.update(old)


@pytest.fixture(params=["simplejpeg", "Pillow only"])
def cpu(request, monkeypatch):
    if request.param == "Pillow only":
        monkeypatch.setitem(sys.modules, "simplejpeg", None)         # "import simplejpeg" now raises ImportError
    return request.param


def _check(img, data, mode, out, where):
    assert img is not None, where
    assert np.array_equal(cp.asnumpy(img), want(data, mode, out)), where


@pytest.mark.parametrize("mode", MODES, ids=["accurate", "fast_dct", "fast_upsample"])
def test_small_image_route_and_declined_files(cpu, mode):
    fast = bool(mode)
    for k, out in enumerate(OUTPUTS):
        for data, name in ((SMALL, "small"), (LARGE, "large")):
            for lim in (0, 100_000, 400_000):
                info = {}
                img = egj.decode(data, cpu_below=lim, info=info, **mode, **out)
                _check(img, data, mode, out, (cpu, mode, k, name, lim, info))
                px = 27_200 if name == "small" else 307_200
                if cpu == "Pillow only":
                    # the small-image route takes only files simplejpeg decodes (Pillow does not report
                    # libjpeg-turbo's warnings, nor has it the fast modes): without it, the GPU decodes them
                    assert info["path"] == "gpu"
                elif px <= lim:
                    assert info["path"] == "cpu"
                elif not fast:                                   # (a fast mode may decline a block; then the CPU)
                    assert info["path"] == "gpu"
        info = {}
        if cpu == "Pillow only" and fast:                            # declined, and no CPU decoder has the mode
            with pytest.raises(egj.FastModeUnavailable, match="needs|only simplejpeg"):
                egj.decode(PROGRESSIVE, info=info, **mode, **out)
            with pytest.raises(egj.FastModeUnavailable):
                egj.decode([SMALL, PROGRESSIVE], **mode, **out)
        else:
            img = egj.decode(PROGRESSIVE, info=info, **mode, **out)
            _check(img, PROGRESSIVE, mode, out, (cpu, mode, k, "declined"))
            assert info["path"] == "cpu" and info["reason"].startswith("progressive JPEG")
            assert info["warnings_checked"] is (cpu != "Pillow only")      # Pillow's pixels say so
            imgs = egj.decode([SMALL, PROGRESSIVE, LARGE], **mode, **out)
            for im, d in zip(imgs, (SMALL, PROGRESSIVE, LARGE)):
                _check(im, d, mode, out, (cpu, mode, k, "batch"))
        info = {}
        assert egj.decode(PROGRESSIVE, fallback=False, info=info, **mode, **out) is None       # told, no pixels
        assert info == dict(path=None, reason="progressive JPEG", warning=None, warnings_checked=None)


@pytest.mark.parametrize("mode", MODES, ids=["accurate", "fast_dct", "fast_upsample"])
@pytest.mark.parametrize("why", ["self-check", "device check", "memory"])
def test_when_the_gpu_path_is_off_or_out_of_memory(cpu, mode, why, monkeypatch, fresh):
    fast = bool(mode)
    for d in (S.FAILED, S._done, S._props, S._checked):
        d.clear()
    if why == "self-check":
        monkeypatch.setenv("EGJ_SELFCHECK_FAIL", "1")
    elif why == "device check":
        monkeypatch.setattr(S, "detect", lambda dev: dict(compute_capability=(3, 0), total_memory=8 << 30,
                                                          free_memory=7 << 30, cuda_runtime="12.9", nvrtc="12.9"))
    else:
        assert egj.self_check()["ok"]
        monkeypatch.setattr(S, "budget", lambda dev: 1 << 16)       # 64 kB: every image "needs about ..."
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # the one warning is tested in test_selfcheck.py
        for k, out in enumerate(OUTPUTS[:4]):
            for data in (SMALL, LARGE):
                info = {}
                if cpu == "Pillow only" and fast:
                    with pytest.raises(egj.FastModeUnavailable):
                        egj.decode(data, cpu_below=0, info=info, **mode, **out)
                    with pytest.raises(egj.FastModeUnavailable):
                        egj.decode([data, data], **mode, **out)
                else:
                    img = egj.decode(data, cpu_below=0, info=info, **mode, **out)
                    _check(img, data, mode, out, (cpu, mode, why, k))
                    assert info["path"] == "cpu"
                    for im in egj.decode([data, data], **mode, **out):
                        _check(im, data, mode, out, (cpu, mode, why, k, "batch"))
                assert egj.decode(data, cpu_below=0, fallback=False, **mode, **out) is None


def test_a_decoder_of_the_callers_own(monkeypatch):
    calls = []

    def plain(data, colorspace):                                     # takes no fast-mode keywords
        calls.append(("plain", colorspace))
        return simplejpeg.decode_jpeg(data, colorspace=colorspace)

    def aware(data, colorspace, fast_dct=False, fast_upsample=False):
        calls.append(("aware", colorspace, fast_dct, fast_upsample))
        return simplejpeg.decode_jpeg(data, colorspace=colorspace, fastdct=fast_dct, fastupsample=fast_upsample)

    try:
        egj.set_cpu_decoder(plain)
        assert "custom" in egj.device_report()["cpu_decoder"] and "responsibility" in egj.device_report()["cpu_decoder"]
        info = {}
        img = egj.decode(SMALL, info=info)                           # the small-image route is not used with one set
        assert info["path"] == "gpu" and not calls
        _check(img, SMALL, {}, dict(colorspace="RGB"), "custom, small")
        _check(egj.decode(PROGRESSIVE, colorspace="BGR"), PROGRESSIVE, {}, dict(colorspace="BGR"), "custom, declined")
        assert calls == [("plain", "BGR")]
        for mode in MODES[1:]:                                       # it cannot be given the mode: no pixels
            with pytest.raises(egj.FastModeUnavailable, match="set_cpu_decoder"):
                egj.decode(PROGRESSIVE, **mode)
            assert egj.decode(PROGRESSIVE, fallback=False, **mode) is None
        egj.set_cpu_decoder(aware)                                   # this one takes the keywords: it gets them
        for mode in MODES:
            _check(egj.decode(PROGRESSIVE, **mode), PROGRESSIVE, mode, dict(colorspace="RGB"), ("aware", mode))
        assert calls[-1] == ("aware", "RGB", False, True)
    finally:
        egj.set_cpu_decoder(None)
    assert "simplejpeg" in egj.device_report()["cpu_decoder"]


def test_decode_cpu_itself(cpu):
    for out in OUTPUTS[:3]:
        assert np.array_equal(egj.decode_cpu(LARGE, out["colorspace"]), want(LARGE, {}, out))
    for mode in MODES[1:]:
        if cpu == "Pillow only":
            with pytest.raises(egj.FastModeUnavailable, match="simplejpeg"):
                egj.decode_cpu(LARGE, **mode)
        else:
            assert np.array_equal(egj.decode_cpu(LARGE, **mode), want(LARGE, mode, dict(colorspace="RGB")))
    r = egj.device_report()["cpu_decoder"]
    assert ("accurate mode only" in r) if cpu == "Pillow only" else ("accurate and fast modes" in r)
