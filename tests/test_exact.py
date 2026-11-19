"""Every corpus file the GPU path accepts decodes byte-identically to libjpeg-turbo; the others are declined."""
import numpy as np
import pytest

pytest.importorskip("cupy")

from conftest import reference


def test_corpus_exact(corpus, dec):
    bad, wrong_path = [], []
    for name, data, expected in corpus:
        out = dec.decode(data)
        if (out is not None) != expected:
            wrong_path.append((name, dec.reason))
            continue
        if out is None:
            continue
        ref = reference(data)
        g = out.get()
        if ref is None or g.shape != ref.shape or not np.array_equal(g, ref):
            bad.append(name)
    assert not wrong_path, wrong_path[:10]
    assert not bad, bad[:10]
    assert len(corpus) > 300


@pytest.mark.parametrize("colorspace", ["BGR", "GRAY"])
def test_other_colorspaces(corpus, dec, colorspace):
    for name, data, expected in corpus[::7]:
        if not expected:
            continue
        out = dec.decode(data, colorspace)
        assert out is not None, (name, dec.reason)
        ref = reference(data, colorspace)
        assert np.array_equal(out.get(), ref), name


def test_stable_over_chunk_sizes(corpus):
    """The fixed point of the synchronisation does not depend on where the chunks are cut. Very small chunks may
    need more passes than the limit (then the GPU path declines); they must never give a different image."""
    import exact_gpu_jpeg as egj
    for S in (64, 512, 8192):
        d = egj.Decoder(chunk_bits=S)
        for name, data, expected in corpus[::11]:
            if not expected:
                continue
            out = d.decode(data)
            if out is None and S < 512:
                assert "synchronise" in d.reason
                continue
            assert out is not None and np.array_equal(out.get(), reference(data)), (S, name, d.reason)


@pytest.mark.parametrize("canary", [False, True], ids=["plain", "guard bands"])
def test_files_that_are_decoded_a_second_time(redo_files, redos, canary):
    """The corpus files whose quick passes do not settle: one second decode each, and libjpeg-turbo's pixels, in
    every colour space, also with guard bands."""
    import exact_gpu_jpeg as egj
    d = egj.Decoder(canary=canary)
    for name, data in redo_files:
        for cs in ("RGB", "BGR", "GRAY"):
            del redos[:]
            got = d.decode(data, cs)
            assert len(redos) == 1, (name, cs, len(redos))
            assert got is not None and np.array_equal(got.get(), reference(data, cs)), (name, cs, d.reason)
