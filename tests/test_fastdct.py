"""Opt-in fast IDCT (Decoder(fast_dct=True)): byte-identical to libjpeg-turbo's JDCT_IFAST (simplejpeg
fastdct=True) on every corpus file the GPU path accepts. tools/check_fastdct.py also compares against each x86-64
code path (SSE2 SIMD and C) in its own process."""
import numpy as np
import pytest

pytest.importorskip("cupy")
simplejpeg = pytest.importorskip("simplejpeg")


def fast_reference(data, colorspace="RGB"):
    try:
        return simplejpeg.decode_jpeg(data, colorspace=colorspace, fastdct=True, fastupsample=False, strict=False)
    except Exception:
        return None


@pytest.fixture(scope="module")
def fdec():
    import exact_gpu_jpeg as egj
    return egj.Decoder(fast_dct=True)


def test_corpus_exact_fastdct(corpus, fdec):
    """ifast keeps 16-bit intermediates with less headroom than islow: on hard edges libjpeg-turbo's SIMD and C
    versions differ (13 corpus files, all edges content) and the GPU declines those, with the range reason."""
    bad, wrong_path, declined = [], [], []
    for name, data, expected in corpus:
        out = fdec.decode(data)
        if out is None:
            if expected and "IDCTs agree" in fdec.reason and "_edges_" in name:
                declined.append(name)
            elif expected:
                wrong_path.append((name, fdec.reason))
            continue
        assert expected, name
        ref = fast_reference(data)
        g = out.get()
        if ref is None or g.shape != ref.shape or not np.array_equal(g, ref):
            bad.append(name)
    assert not bad, bad[:10]
    assert not wrong_path, wrong_path[:10]
    assert len(declined) <= 30, declined


@pytest.mark.parametrize("colorspace", ["BGR", "GRAY"])
def test_fastdct_colorspaces(corpus, fdec, colorspace):
    for name, data, expected in corpus[::7]:
        if not expected:
            continue
        out = fdec.decode(data, colorspace)
        if out is None and "IDCTs agree" in fdec.reason and "_edges_" in name:
            continue
        assert out is not None, (name, fdec.reason)
        assert np.array_equal(out.get(), fast_reference(data, colorspace)), name


def test_fastdct_differs_from_default(corpus, dec, fdec):
    """The option does something: on photo-like files the two IDCTs give different pixels."""
    name, data, _ = next(c for c in corpus if c[0].startswith("sj_480x640_smooth"))
    assert not np.array_equal(dec.decode(data).get(), fdec.decode(data).get()), name


def test_decode_fast_dct_and_fallback(corpus):
    import exact_gpu_jpeg as egj
    name, data, _ = next(c for c in corpus if c[0].startswith("sj_97x203"))
    info = {}
    out = egj.decode(data, fast_dct=True, info=info, cpu_below=0)       # (a file this small would go to the CPU)
    assert info["path"] == "gpu" and np.array_equal(out.get(), fast_reference(data))
    name, data, _ = next(c for c in corpus if c[0].endswith("_prog.jpg"))   # declined: the CPU fallback, also ifast
    out = egj.decode(data, fast_dct=True, info=info)
    assert info["path"] == "cpu" and np.array_equal(out.get(), fast_reference(data))


def test_ifast_table():
    from exact_gpu_jpeg._decoder import ifast_table
    q = np.arange(1, 65)
    t = ifast_table(q)
    assert t[0] == 4 and t[9] == (10 * 31521 + 2048) >> 12
    assert ifast_table(np.full(64, 65535))[9] == np.array((65535 * 31521 + 2048) >> 12).astype(np.int16)  # wraps
