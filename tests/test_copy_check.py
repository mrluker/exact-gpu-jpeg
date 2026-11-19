"""The check of the copy kernels (_copycheck.py): it runs at the first use of the packed form on a GPU and not
before; it passes on this GPU; a failure changes the form copies are kept in (raw instead of packed) with one
warning, and never what a decode returns."""
import warnings

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _copycheck  # noqa: E402
from conftest import reference  # noqa: E402
import packref  # noqa: E402
from test_repeat_cache import _jpeg  # noqa: E402


def _same(img, jpeg):
    return img is not None and np.array_equal(cp.asnumpy(img), reference(jpeg))


@pytest.fixture
def unchecked(monkeypatch):
    """The process as it is before any copy form was used (and as it was again afterwards)."""
    monkeypatch.setattr(_copycheck, "_state", {})


def test_the_vector_the_check_uses_is_the_formats_first():
    for seed, B in ((1, 64), (3, 2000)):
        for a, b in zip(_copycheck.vector_arrays(seed, B), packref.arrays(seed, B, False)):
            assert np.array_equal(a, b) and a.dtype == b.dtype
    v = packref.VECTORS[0]
    assert (v[1], v[5], v[6]) == (_copycheck.BLOCKS, _copycheck.ARRAYS, _copycheck.PACKED_BODY)


def test_it_runs_at_the_first_use_of_a_copy_form_and_passes_here(unchecked):
    dev = cp.cuda.Device().id
    jpeg = _jpeg(640, 480)
    dec = egj.Decoder()
    assert _same(dec.decode(jpeg), jpeg)
    dec.keep_coefs = "raw"
    assert _same(dec.decode(jpeg), jpeg) and dec.take_coefs().form == "raw"
    cache = egj.RepeatCache(min_pixels=0)
    assert _copycheck._state == {}                             # decodes, raw handles, making a cache: not run
    assert set(egj.device_report()["copy_check"].values()) == {"not run yet"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dec.keep_coefs = "packed"
        assert _same(dec.decode(jpeg), jpeg) and dec.take_coefs().form == "packed"
        assert _same(egj.Decoder(cache=cache).decode(jpeg), jpeg)
    assert _copycheck._state[dev] == dict(packed=None)
    assert egj.device_report()["copy_check"] == dict(packed="passed")


def test_a_failed_check_changes_the_copies_and_not_the_decodes(unchecked, monkeypatch):
    jpeg = _jpeg(800, 600, seed=3)
    monkeypatch.setenv("EGJ_COPYCHECK_FAIL", "packed")
    dec = egj.Decoder(keep_coefs="packed")
    with pytest.warns(RuntimeWarning, match="check of the copy kernels failed") as w:
        assert _same(dec.decode(jpeg), jpeg)
        h = dec.take_coefs()
    assert len([x for x in w if "copy kernels" in str(x.message)]) == 1 and "Decodes are not affected" in str(w[0].message)
    assert h.form == "raw" and _same(dec.decode_from_handle(h), jpeg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # said once, whoever asks next
        cache = egj.RepeatCache(min_pixels=0)
        cd = egj.Decoder(cache=cache)
        assert _same(cd.decode(jpeg), jpeg) and _same(cd.decode(jpeg), jpeg)
    assert cache.stats["used"] == 1
    assert {c.form for c in cache._copies.values()} == {"raw"}
    assert egj.device_report()["copy_check"] == dict(packed="failed: forced by EGJ_COPYCHECK_FAIL")
    assert egj.device_report()["gpu_path"] == "on"
