"""fast_upsample=True (opt-in): byte-identical to libjpeg-turbo with do_fancy_upsampling = FALSE
(simplejpeg fastupsample=True) on every corpus file the GPU path accepts, in RGB, BGR and GRAY; the default is
unchanged (test_exact.py)."""
import numpy as np
import pytest

pytest.importorskip("cupy")
simplejpeg = pytest.importorskip("simplejpeg")

import exact_gpu_jpeg as egj


def fast_reference(data, colorspace="RGB"):
    return simplejpeg.decode_jpeg(data, colorspace=colorspace, fastdct=False, fastupsample=True, strict=False)


@pytest.mark.parametrize("colorspace", ["RGB", "BGR", "GRAY"])
def test_corpus_fast_upsample_exact(corpus, colorspace):
    dec = egj.Decoder(fast_upsample=True)
    bad, n = [], 0
    for name, data, expected in corpus[:: 1 if colorspace == "RGB" else 5]:
        if not expected:
            continue
        out = dec.decode(data, colorspace)
        assert out is not None, (name, dec.reason)
        ref = fast_reference(data, colorspace)
        if ref.ndim == 2:
            ref = ref[..., None]
        n += 1
        if not np.array_equal(out.get(), ref):
            bad.append(name)
    assert not bad, bad[:10]
    assert n > 50


def test_function_option(corpus):
    """On the GPU (the first 4:2:2 corpus file is 1x1, which the CPU route takes, and on which fast and fancy
    upsampling are the same)."""
    name, data, _ = next(c for c in corpus if c[2] and "422" in c[0] and len(c[1]) > 20000)
    info = {}
    a = egj.decode(data, fast_upsample=True, cpu_below=0, info=info).get()
    assert info["path"] == "gpu"
    assert np.array_equal(a, fast_reference(data)), name
    assert not np.array_equal(a, egj.decode(data, cpu_below=0).get())      # the option really changes the pixels
