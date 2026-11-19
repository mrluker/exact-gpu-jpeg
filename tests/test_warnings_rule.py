"""Files libjpeg-turbo decodes only with a warning (audit of 2026-11-18: area 1's finding and the strict rule).

Default strict=True on every entry point: such a file is declined by the GPU path and the CPU decoder raises,
naming the warning. strict=False: both routes give libjpeg-turbo's non-strict image (as djpeg gives it) and say
which warning occurred. The two routes agree in both modes. The expected outcome comes from libjpeg-turbo itself
(simplejpeg: strict raises on a warning, strict=False decodes); for the two header warnings TurboJPEG refuses even
non-strict, Pillow (also libjpeg-turbo) is the reference image. Also: CMYK / YCCK files are outside the claim, and
float output needs finite scale and bias."""
import io
import struct

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")
simplejpeg = pytest.importorskip("simplejpeg")

import exact_gpu_jpeg as egj  # noqa: E402


def _good(w=320, h=240, seed=1, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


def _host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def _segments(d):
    i, out = 2, []
    while d[i] == 0xFF and d[i + 1] != 0xDA:
        L = struct.unpack(">H", d[i + 2:i + 4])[0]
        out.append((i, d[i + 1], L))
        i += 2 + L
    return out


def _before_rst(d):
    """Three data bytes inserted before the first restart marker of a file with restart markers."""
    s = d.index(b"\xff\xda")
    k = d.index(b"\xff\xd0", s)
    return d[:k] + b"\x11\x22\x33" + d[k:]


def _adobe2(d):
    """No JFIF segment, an Adobe segment with transform 2 (libjpeg-turbo warns and assumes YCbCr)."""
    i, m, L = next(s for s in _segments(d) if s[1] == 0xE0)
    body = d[:i] + d[i + 2 + L:]
    adobe = b"\xff\xee\x00\x0eAdobe\x00\x64\x00\x00\x00\x00\x02"
    return body[:2] + adobe + body[2:]


def _jfif(d, major):
    i = d.index(b"JFIF\x00")
    return d[:i + 5] + bytes([major]) + d[i + 6:]


def _app0(major):
    return b"\xff\xe0\x00\x10JFIF\x00" + bytes([major, 1]) + b"\x00\x00\x01\x00\x01\x00\x00"


def _second_app0(d, major):
    i, m, L = next(s for s in _segments(d) if s[1] == 0xE0)
    return d[:i + 2 + L] + _app0(major) + d[i + 2 + L:]


def _bad_icc(d):
    """An ICC_PROFILE APP2 chunk numbered 5 of 2 (libjpeg-turbo's jpeg_read_icc_profile warns: bad ICC marker)."""
    body = b"ICC_PROFILE\x00" + bytes([5, 2]) + bytes(32)
    return d[:2] + b"\xff\xe2" + struct.pack(">H", len(body) + 2) + body + d[2:]


def _junk_between_header_markers(d):
    i, m, L = next(s for s in _segments(d) if s[1] == 0xDB)
    return d[:i + 2 + L] + b"\x01\x02\x03" + d[i + 2 + L:]


g, rst = _good(), _good(restart_marker_blocks=4)
CASES = {
    "no end marker": (g[:-2], True),                 # (file, the GPU path decodes it when strict=False)
    "data that ends early": (g[:len(g) * 2 // 3], False),
    # unused bytes in the scan: libjpeg-turbo warns for some lengths only, so the GPU path declines them in both
    # modes and the CPU decoder decides (test_a_short_tail_libjpeg_turbo_does_not_report)
    "extraneous bytes before the end marker": (g[:-2] + bytes(range(1, 21)) + g[-2:], False),
    "extraneous bytes before a restart marker": (_before_rst(rst), False),
    "an unknown JFIF major version": (_jfif(g, 3), False),
    "JFIF major version 2": (_jfif(g, 2), False),
    "a second JFIF segment with major version 3": (_second_app0(g, 3), False),
    "a JFIF segment with major version 3 after the scan": (g[:-2] + _app0(3) + g[-2:], True),
    "a bad ICC profile segment": (_bad_icc(g), False),
    "extraneous bytes between header markers": (_junk_between_header_markers(g), False),
    "an unknown Adobe transform": (_adobe2(g), False),
}


def _refused(data):
    """TurboJPEG refuses the file even non-strict (a header warning): only the libjpeg API (Pillow) decodes it."""
    try:
        simplejpeg.decode_jpeg(data, colorspace="RGB", strict=False)
        return False
    except ValueError:
        return True


def _libjpeg_turbo(data):
    """(warned: libjpeg-turbo's strict decode raises, its non-strict image)."""
    try:
        simplejpeg.decode_jpeg(data, colorspace="RGB")
        warned = False
    except ValueError:
        warned = True
    try:
        img = simplejpeg.decode_jpeg(data, colorspace="RGB", strict=False)
    except ValueError:                                # the header warnings: TurboJPEG refuses; Pillow decodes
        with Image.open(io.BytesIO(data)) as im:
            img = np.asarray(im.convert("RGB"))
    return warned, img


@pytest.mark.parametrize("name", list(CASES))
def test_libjpeg_turbo_warns_about_each_case(name):
    data, _ = CASES[name]
    warned, img = _libjpeg_turbo(data)
    assert warned, name                                        # (strict raises, also where TurboJPEG refuses outright)
    assert img is not None and img.shape[:2] == (240, 320)


@pytest.mark.parametrize("name", list(CASES))
def test_strict_by_default_on_every_route(name):
    data, _ = CASES[name]
    dec = egj.Decoder()
    assert dec.decode(data) is None, name
    assert dec.reason, name
    assert dec.decode_batch([data, g])[0] is None and dec.reasons[0], name
    job = dec.decode_async(data)
    assert job is None or job.result() is None, name
    for cpu_below in (0, 10 ** 9):                              # the GPU route, then the CPU route
        with pytest.raises(ValueError, match="libjpeg-turbo"):  # (the error names libjpeg-turbo's warning)
            egj.decode(data, cpu_below=cpu_below)
    with pytest.raises(Exception):
        egj.decode([data, g])
    with pytest.raises(ValueError, match="libjpeg-turbo"):
        egj.decode_cpu(data)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast_dct"])
def test_strict_false_gives_libjpeg_turbos_image_and_names_the_warning(name, fast):
    data, gpu_decodes = CASES[name]
    if fast and _refused(data):
        pytest.skip("only TurboJPEG has the fast modes, and it refuses this file")
    want = (simplejpeg.decode_jpeg(data, colorspace="RGB", strict=False, fastdct=fast) if fast
            else _libjpeg_turbo(data)[1])
    for cpu_below in (0, 10 ** 9):
        info = {}
        img = egj.decode(data, cpu_below=cpu_below, strict=False, fast_dct=fast, info=info)
        assert np.array_equal(_host(img), want), (name, cpu_below)
        assert info["warning"], (name, cpu_below, info)
    info = []
    got = egj.decode([data, g], strict=False, fast_dct=fast, info=info)
    assert np.array_equal(_host(got[0]), want) and info[0]["warning"], (name, info)
    dec = egj.Decoder(strict=False, fast_dct=fast)
    img = dec.decode(data)
    if gpu_decodes:                                             # the GPU path decodes it and names the warning
        assert img is not None and np.array_equal(img.get(), want) and dec.warning, (name, dec.reason)
        job = dec.decode_async(data)
        assert np.array_equal(job.result().get(), want) and job.warning, name
        out = dec.decode_batch([data, data, g])
        assert np.array_equal(out[0].get(), want) and dec.warnings[0] and dec.warnings[2] is None, name
    else:
        assert img is None, name


@pytest.mark.parametrize("name", [n for n, (d, gpu) in CASES.items() if gpu])
@pytest.mark.parametrize("mode", ["BGR", "GRAY", "fast_upsample"])
def test_strict_false_in_the_other_modes(name, mode):
    """(second audit, part 5.7) The files the GPU path decodes with strict=False, in BGR, GRAY and with fast
    upsampling: libjpeg-turbo's non-strict image in that mode on both routes, the warning named; strict declines."""
    data = CASES[name][0]
    cs = mode if mode != "fast_upsample" else "RGB"
    fu = mode == "fast_upsample"
    want = simplejpeg.decode_jpeg(data, colorspace=cs, strict=False, fastupsample=fu)
    for cpu_below in (0, 10 ** 9):
        info = {}
        img = egj.decode(data, cs, cpu_below=cpu_below, strict=False, fast_upsample=fu, info=info)
        assert np.array_equal(_host(img), want) and info["warning"], (name, mode, cpu_below, info)
        with pytest.raises(ValueError, match="libjpeg-turbo"):
            egj.decode(data, cs, cpu_below=cpu_below, fast_upsample=fu)
    dec = egj.Decoder(strict=False, fast_upsample=fu)
    img = dec.decode(data, cs)
    assert img is not None and np.array_equal(img.get(), want) and dec.warning, (name, mode)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_short_tail_libjpeg_turbo_does_not_report(n):
    """(second audit, part 2.6) A few unused bytes before the end marker: libjpeg-turbo's bit buffer has read them
    already and it does not warn. The GPU path cannot tell such a tail from a longer one: it declines it in both
    modes, and the CPU decoder gives libjpeg-turbo's image with no warning."""
    data = g[:-2] + bytes(range(1, n + 1)) + g[-2:]
    want = simplejpeg.decode_jpeg(data, colorspace="RGB")      # (strict: no warning, the premise of the test)
    for strict in (True, False):
        dec = egj.Decoder(strict=strict)
        assert dec.decode(data) is None and "the CPU decoder decides" in dec.reason
        assert dec.decode_batch([data, g])[0] is None and "the CPU decoder decides" in dec.reasons[0]
        info = {}
        img = egj.decode(data, cpu_below=0, strict=strict, info=info)
        assert np.array_equal(_host(img), want) and info["path"] == "cpu" and info["warning"] is None


def test_a_cmyk_file_with_an_unknown_adobe_transform_is_strict_too():
    """(second audit, part 2.4) Four components: libjpeg-turbo knows Adobe transforms 0 and 2 only."""
    a = np.random.default_rng(2).integers(0, 256, (48, 64, 4), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "CMYK").save(b, "JPEG", quality=90)
    d = b.getvalue()
    k = d.index(b"Adobe")
    bad = d[:k + 11] + b"\x01" + d[k + 12:]                   # transform 1
    with pytest.raises(ValueError, match="Adobe"):
        egj.decode_cpu(bad)
    egj.decode_cpu(bad, strict=False)
    assert egj.decode_cpu(d) is not None


def test_a_good_file_has_no_warning():
    for strict in (True, False):
        info = {}
        egj.decode(g, cpu_below=0, strict=strict, info=info)
        assert info["path"] == "gpu" and info["warning"] is None
        info = {}
        egj.decode(g, cpu_below=10 ** 9, strict=strict, info=info)
        assert info["path"] == "cpu" and info["warning"] is None


def test_a_file_libjpeg_turbo_rejects_is_rejected_by_both_routes_in_both_modes():
    bad = g[:-2] + b"\xff\xdb\x00\x43\x0b" + bytes(64) + g[-2:]       # a DQT with table index 11 after the scan
    for strict in (True, False):
        for cpu_below in (0, 10 ** 9):
            with pytest.raises(Exception):
                egj.decode(bad, cpu_below=cpu_below, strict=strict)


def test_a_warned_file_leaves_no_cache_copy():
    from exact_gpu_jpeg import RepeatCache
    c = RepeatCache()
    lenient = egj.Decoder(strict=False, cache=c)
    assert lenient.decode(CASES["no end marker"][0]) is not None
    s = c.stats() if callable(c.stats) else c.stats
    assert s["made"] == 0                                       # a strict decoder sharing the cache cannot use it


def test_cmyk_files_are_outside_the_claim_and_say_so():
    a = np.random.default_rng(2).integers(0, 256, (48, 64, 4), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "CMYK").save(b, "JPEG", quality=90)
    dec = egj.Decoder()
    assert dec.decode(b.getvalue()) is None and "outside the identical-pixels claim" in dec.reason
    info = {}
    egj.decode(b.getvalue(), info=info, cpu_below=0)
    assert info["path"] == "cpu" and "CMYK" in info["reason"]


@pytest.mark.parametrize("bad", [dict(scale=float("nan")), dict(bias=float("inf")), dict(scale=[1.0, 1e39, 1.0])])
def test_scale_and_bias_must_be_finite(bad):
    """(audit 2.2) A NaN result is not NumPy's bit pattern on the GPU; a value that overflows float32 is not finite."""
    with pytest.raises(ValueError, match="finite"):
        egj.Decoder().decode(_good(64, 48), dtype="float32", **bad)


def test_a_queued_job_keeps_the_strict_rule_it_was_queued_with():
    """(second audit, parts 2.9 and 3.10) Changing dec.strict before result() does not change the job's verdict."""
    data = CASES["no end marker"][0]
    dec = egj.Decoder()
    job = dec.decode_async(data)
    dec.strict = False
    assert job is None or job.result() is None
    dec = egj.Decoder(strict=False)
    job = dec.decode_async(data)
    dec.strict = True
    assert job.result() is not None and job.warning


@pytest.mark.parametrize("name", ["no end marker", "extraneous bytes before the end marker"])
def test_without_simplejpeg_a_warning_the_gpu_path_found_still_counts(monkeypatch, name):
    """(second audit, part 5.1 / 2.5) Pillow, standing in when simplejpeg is not installed, decodes past every
    warning: what the GPU path found when it declined the file still raises under strict=True. strict=False:
    Pillow's image (libjpeg-turbo's), with a warning."""
    import sys
    data = CASES[name][0]
    want = _libjpeg_turbo(data)[1]
    monkeypatch.setitem(sys.modules, "simplejpeg", None)          # import simplejpeg now fails
    with pytest.raises(ValueError, match="libjpeg-turbo"):
        egj.decode(data, cpu_below=0)
    info = {}
    img = egj.decode(data, cpu_below=0, strict=False, info=info)
    assert np.array_equal(_host(img), want) and info["warning"], info
    assert info["warnings_checked"] is (info["path"] == "gpu"), info      # Pillow's pixels: not every warning seen


def test_pillows_pixels_say_that_not_every_warning_was_checked(monkeypatch):
    """(the coordinator's rule for the strict gap) A file the GPU path declines for another reason than a warning,
    decoded by Pillow because simplejpeg is not there: the image comes back, under strict too, and
    info["warnings_checked"] is False, with the route reason saying Pillow decoded it. With simplejpeg: True."""
    import sys
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(4).integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(
        b, "JPEG", quality=90, progressive=True)
    prog = b.getvalue()
    info = {}
    egj.decode(prog, cpu_below=0, info=info)
    assert info["path"] == "cpu" and info["warnings_checked"] is True, info
    monkeypatch.setitem(sys.modules, "simplejpeg", None)
    for strict in (True, False):
        info = {}
        img = egj.decode(prog, cpu_below=0, strict=strict, info=info)
        assert img is not None and info["path"] == "cpu" and info["warnings_checked"] is False, info
        assert "Pillow" in info["reason"], info
        got = []
        egj.decode([prog, g], strict=strict, info=got)
        assert got[0]["warnings_checked"] is False and got[1]["warnings_checked"] is True, got
