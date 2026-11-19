"""Malformed input: every decode returns an exact image or None; it never raises and never differs from
libjpeg-turbo. tools/fuzz.py runs the same checks at scale, with a watchdog for hangs and crashes."""
import os
import random
import sys

import numpy as np
import pytest

pytest.importorskip("cupy")

from conftest import reference, strict_refuses
from jpegwriter import grid, write_jpeg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fuzz import mutate  # noqa: E402


def _check(dec, data):
    out = dec.decode(data)                 # must not raise
    if out is not None:
        ref = reference(data)
        assert ref is not None and np.array_equal(out.get(), ref)
        if dec.strict:                     # nor accept, under strict=True, a file libjpeg-turbo warns about
            assert not strict_refuses(data)
    return out


def test_truncation(corpus, dec):
    for name, data, exp in corpus:
        if name.startswith(("pil_97x203_s1_rst1", "sj_97x203_noise_420", "craft_3_")):
            for n in list(range(0, len(data), 37)) + [len(data) - 3, len(data) - 1]:
                _check(dec, data[:n])


def test_random_mutations(corpus, dec):
    rng = random.Random(4)
    seeds = [d for n, d, e in corpus if len(d) < 60000]
    for _ in range(600):
        _check(dec, mutate(rng.choice(seeds), rng))


def test_restart_markers_out_of_order(dec):
    rng = np.random.default_rng(0)
    samp = [(2, 2), (1, 1), (1, 1)]
    blocks = [rng.integers(-5, 6, grid(samp, 40, 40, c) + (64,)) for c in range(3)]
    q = [np.full(64, 4)] * 3
    assert _check(dec, write_jpeg(blocks, q, samp, 40, 40, dri=1)) is not None
    swapped = write_jpeg(blocks, q, samp, 40, 40, dri=1, rst_numbers=lambda r: (r + 1) & 7)
    assert dec.decode(swapped) is None and "restart" in dec.reason


def test_huge_dimensions_little_data(dec):
    rng = np.random.default_rng(1)
    d = bytearray(write_jpeg([rng.integers(-3, 4, (1, 1, 64))], [np.full(64, 2)], [(1, 1)], 8, 8))
    i = d.find(b"\xff\xc0")
    d[i + 5:i + 9] = b"\xff\xdc\xff\xdc"          # 65500 x 65500
    assert dec.decode(bytes(d)) is None


def test_extreme_coefficients_are_declined_for_the_range(dec):
    """Coefficients far outside what an 8-bit encoder produces: libjpeg-turbo's SIMD and C IDCTs disagree there,
    so the GPU path must decline, with the range reason; tools/check_simd.py checks against all three code
    paths. Files whose coefficients stay inside the range must decode, to libjpeg-turbo's pixels."""
    rng = np.random.default_rng(3)
    samp = [(2, 1), (1, 1), (1, 1)]
    for k in range(40):
        blocks = [rng.integers(-2000, 2001, grid(samp, 24, 16, c) + (64,)) for c in range(3)]
        data = write_jpeg(blocks, [rng.integers(1, 255, 64) for _ in range(3)], samp, 24, 16)
        assert dec.decode(data) is None, k
        assert "IDCTs agree" in dec.reason, (k, dec.reason)
    for k in range(20):
        blocks = [rng.integers(-60, 61, grid(samp, 24, 16, c) + (64,)) * (rng.random(grid(samp, 24, 16, c) + (64,)) < 0.2)
                  for c in range(3)]
        data = write_jpeg(blocks, [rng.integers(1, 9, 64) for _ in range(3)], samp, 24, 16)
        img = dec.decode(data)
        assert img is not None, (k, dec.reason)
        assert np.array_equal(img.get(), reference(data)), k


def test_regressions(dec):
    """Inputs the fuzzer once caught (synthetic seeds, mutated). Each must give None or libjpeg-turbo's image.
    dqt_after_scan.jpg: a corrupt DQT after the scan, which libjpeg-turbo parses and rejects."""
    import glob
    files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "regressions", "*.jpg"))
    assert files
    for p in files:
        _check(dec, open(p, "rb").read())


def test_canary_checker_catches_overwrite():
    """The guard-band test mode (EGJ_CANARY=1) really detects a write past a buffer's end."""
    import pytest
    import exact_gpu_jpeg as egj
    d = egj.Decoder(canary=True)
    d._guards = []
    buf = d._alloc(100, np.uint8)
    raw, nbytes = d._guards[0]
    raw[d.GUARD + nbytes] = 0                       # one byte past the end
    with pytest.raises(RuntimeError, match="outside device buffer"):
        d._check_guards()


def test_canary_mode_decodes_exactly(corpus):
    import exact_gpu_jpeg as egj
    d = egj.Decoder(canary=True)
    for name, data, expected in corpus[::17]:
        _check(d, data)


def test_memory_budget(corpus):
    import exact_gpu_jpeg as egj
    d = egj.Decoder(max_gpu_bytes=1 << 20)
    data = next(x for n, x, e in corpus if n.startswith("sj_1080x1920"))
    assert d.decode(data) is None and "GPU memory" in d.reason


def test_data_after_eoi(dec):
    """Phone JPEGs often carry more after the end-of-image marker (MPF multi-picture files, HDR gain maps: a second
    JPEG; or padding). libjpeg-turbo stops at EOI, so the GPU path must give exactly its image for the first one."""
    import io

    from PIL import Image
    rng = np.random.default_rng(5)

    def jpg(w, h, ss):
        b = io.BytesIO()
        Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8)).save(b, "JPEG", quality=90, subsampling=ss)
        return b.getvalue()
    main = jpg(96, 64, 2)
    for tail in (jpg(32, 24, 0), b"\0" * 4096, b"trailer data", b"\xff\xd8\xff"):
        data = main + tail
        img = dec.decode(data)
        assert img is not None, dec.reason
        assert np.array_equal(img.get(), reference(data))
