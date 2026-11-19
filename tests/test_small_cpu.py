"""decode() sends one small image to the CPU decoder (cpu_below / EGJ_SMALL_CPU_PIXELS): the same pixels as the GPU
path, for every file, option and colour space; everything else is untouched."""
import numpy as np
import pytest

pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import reference  # noqa: E402


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())


def _turbojpeg_takes(data):
    """The small-image CPU route takes only files simplejpeg (TurboJPEG) decodes: it reports libjpeg-turbo's
    warnings for strict; Pillow, the stand-in for other sampling layouts, would not."""
    import simplejpeg
    try:
        simplejpeg.decode_jpeg_header(data)
        return True
    except Exception:
        return False


def _cpu_reads(data):
    """Does the CPU decoder decode it as it stands? (It refuses a file without its end marker, for one; such a file
    stays on the GPU path, which decodes it.)"""
    try:
        egj.decode_cpu(data)
        return True
    except Exception:
        return False


def _pixels(data):
    h = egj.inspect(data)
    return h["width"] * h["height"] if h.get("supported") else None


def test_small_images_take_the_cpu_route_with_the_same_pixels(corpus):
    routed = gpu = 0
    for name, data, expected in corpus:
        ref = reference(data)
        if ref is None:
            continue
        if not _cpu_reads(data):                          # libjpeg-turbo warns about it (strict): both routes raise
            for cb in (0, 10 ** 9):                       # (tests/test_warnings_rule.py)
                with pytest.raises(Exception):
                    egj.decode(data, cpu_below=cb)
            continue
        a, b = {}, {}
        x = egj.decode(data, info=a)                      # the default: small ones on the CPU
        y = egj.decode(data, info=b, cpu_below=0)         # always the GPU path (CPU fallback only when it declines)
        assert _same(x, y), name
        assert np.array_equal(x.get(), ref), name
        px = _pixels(data)
        small = px is not None and px <= egj.SMALL_CPU_PIXELS and _cpu_reads(data) and _turbojpeg_takes(data)
        assert a["path"] == ("cpu" if small or not expected else "gpu"), (name, a)
        assert b["path"] == ("gpu" if expected else "cpu"), (name, b)
        if small and expected:
            assert a["reason"] == egj.SMALL_REASON
            routed += 1
        elif expected:
            gpu += 1
    assert routed > 100 and gpu > 20


def test_options_colourspaces_and_paths(corpus, tmp_path):
    small = [(n, d) for n, d, e in corpus if e and (_pixels(d) or 10 ** 9) <= egj.SMALL_CPU_PIXELS][::9]
    assert len(small) > 8
    for k, (name, data) in enumerate(small):
        for kw in (dict(colorspace="BGR"), dict(colorspace="GRAY"), dict(dtype="float32"),
                   dict(dtype="float16", layout="CHW", scale=[0.017, 0.018, 0.019], bias=[-2.1, -2.0, -1.8]),
                   dict(fast_upsample=True), dict(fast_dct=True)):
            assert _same(egj.decode(data, **kw), egj.decode(data, cpu_below=0, **kw)), (name, kw)
        p = tmp_path / f"s{k}.jpg"
        p.write_bytes(data)
        info = {}
        assert _same(egj.decode(str(p), info=info), egj.decode(data, cpu_below=0))
        assert info["path"] == "cpu"
        assert _same(egj.decode(bytearray(data)), egj.decode(memoryview(data), cpu_below=0))


def test_when_it_does_not_apply(corpus):
    big = next(d for n, d, e in corpus if e and (_pixels(d) or 0) > egj.SMALL_CPU_PIXELS)
    small = next(d for n, d, e in corpus if e and 1000 < (_pixels(d) or 0) <= egj.SMALL_CPU_PIXELS)
    info = {}
    egj.decode(big, info=info)
    assert info["path"] == "gpu"                                       # above the limit
    egj.decode(big, info=info, cpu_below=10 ** 9)
    assert info["path"] == "cpu" and _same(egj.decode(big, cpu_below=10 ** 9), egj.decode(big))   # a caller's limit
    egj.decode(small, info=info, fallback=False)
    assert info["path"] == "gpu"                                       # fallback=False means GPU only
    assert all(i["path"] == "gpu" for i in _batch_info([small, small, small]))   # lists are batched on the GPU
    try:
        egj.set_cpu_decoder(lambda data, colorspace: np.zeros((2, 2, 3), np.uint8))
        egj.decode(small, info=info)
        assert info["path"] == "gpu"                                   # only the default CPU decoder is trusted
    finally:
        egj.set_cpu_decoder(None)
    with pytest.raises(ValueError):
        egj.decode(small, "XYZ")


def _batch_info(files):
    info = []
    egj.decode(files, info=info)
    return info


def test_bad_small_files_behave_as_before(corpus):
    """Truncated and damaged small files: the same image or the same kind of error with and without the CPU route."""
    small = [d for n, d, e in corpus if e and 1000 < (_pixels(d) or 0) <= egj.SMALL_CPU_PIXELS][:25]
    for data in small:
        for bad in (data[:len(data) // 2], data[:len(data) - 2], data[:300] + bytes(len(data) - 300)):
            out = []
            for kw in ({}, dict(cpu_below=0)):
                try:
                    out.append(egj.decode(bad, **kw))
                except (ValueError, OSError) as e:
                    out.append(type(e))
            if isinstance(out[0], type) or isinstance(out[1], type):
                assert isinstance(out[0], type) and isinstance(out[1], type)
            else:
                assert _same(out[0], out[1])


def test_frame_pixels_agrees_with_the_parser(corpus):
    """The cheap size check that lets a larger image skip the routing: right wherever the header parses."""
    seen = 0
    for name, data, expected in corpus:
        px = _pixels(data)
        got = egj._frame_pixels(data)
        if px is not None:
            assert got == px, name
            assert egj._frame_pixels(bytearray(data)) == px and egj._frame_pixels(memoryview(data)) == px
            assert egj._frame_pixels(np.frombuffer(data, np.uint8)) == px
            seen += 1
        assert got is None or isinstance(got, int)
    assert seen > 200
    for bad in (b"", b"\xff\xd8", b"not a jpeg at all", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\x00"):
        assert egj._frame_pixels(bad) is None
