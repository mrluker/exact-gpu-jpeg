"""Decoder.upload_ahead: staging a file and then decoding it gives decode()'s image; anything else behaves as before."""
import threading

import numpy as np
import pytest

pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import reference  # noqa: E402


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())


def _same_reason(a, b):
    """A reason from the header is the same text. For bad entropy-coded data both must name errors found in the
    data; which ones can differ (blocks that were never decoded hold whatever was in memory)."""
    from exact_gpu_jpeg._decoder import ERRORS
    known = set(ERRORS.values())

    def data_level(r):
        return r is not None and all(x in known for x in r.split("; "))
    return a == b or (data_level(a) and data_level(b))


def test_ahead_then_decode_equals_decode(corpus, dec):
    plain = egj.Decoder()
    staged = 0
    for name, data, expected in corpus[::3]:
        ok = dec.upload_ahead(data)
        staged += ok
        why = dec.reason
        got = dec.decode(data)
        want = plain.decode(data)
        assert _same(want, got), name
        assert _same_reason(dec.reason, plain.reason), name
        assert (got is not None) == expected, name
        if not ok and why:                                 # declined at the header: the same reason as decode()
            assert why == plain.reason, name
        if got is not None:
            assert ok, name
            assert np.array_equal(got.get(), reference(data)), name
    assert staged > 80


def test_paths(corpus, dec, tmp_path):
    plain = egj.Decoder()
    files = []
    for k, (name, data, expected) in enumerate(corpus[::17]):
        p = tmp_path / f"f{k}.jpg"
        p.write_bytes(data)
        files.append((p, data))
    for p, data in files:
        dec.upload_ahead(p)
        assert _same(dec.decode(str(p)), plain.decode(data))      # the same path, given as a str
        assert _same(dec.decode(p), plain.decode(data))           # not staged any more: read again


def test_other_file_drops_the_staged_one(corpus, dec):
    plain = egj.Decoder()
    good = [d for n, d, e in corpus if e]
    a, b, c = good[3], good[40], good[90]
    assert dec.upload_ahead(a)
    assert _same(dec.decode(b), plain.decode(b))               # a different file: decoded as usual
    assert _same(dec.decode(a), plain.decode(a))               # a is not staged any more
    assert dec.upload_ahead(a) and dec.upload_ahead(c)         # staging another replaces the first
    assert _same(dec.decode(c), plain.decode(c))
    assert dec.upload_ahead(a)
    assert _same(dec.decode(bytes(bytearray(a))), plain.decode(a))   # equal bytes, another object: not the staged one
    assert _same(dec.decode(a), plain.decode(a))


def test_async_and_output_options(corpus, dec):
    plain = egj.Decoder()
    for name, data, expected in corpus[::23]:
        if not expected:
            continue
        assert dec.upload_ahead(data)
        job = dec.decode_async(data)
        assert _same(job.result(), plain.decode(data)), name
        assert dec.upload_ahead(data)
        kw = dict(dtype="float16", layout="CHW", scale=[0.017, 0.018, 0.019], bias=[-2.1, -2.0, -1.8])
        assert _same(dec.decode(data, **kw), plain.decode(data, **kw)), name
        assert dec.upload_ahead(data)
        assert _same(dec.decode(data, "GRAY"), plain.decode(data, "GRAY")), name


def test_reuse_after_errors(corpus, dec):
    plain = egj.Decoder()
    good = [d for n, d, e in corpus if e and len(d) > 2000]
    a = good[5]
    for bad in (a[:len(a) // 2], a[:700] + bytes(len(a) - 700), b"", b"\xff\xd8\xff", a[:len(a) - 2] + b"\xff\xc4"):
        dec.upload_ahead(bad)
        assert _same(dec.decode(bad), plain.decode(bad))
        assert _same_reason(dec.reason, plain.reason)
        assert dec.upload_ahead(a)                             # the decoder is fine afterwards
        assert _same(dec.decode(a), plain.decode(a))
    progressive = [d for n, d, e in corpus if not e][0]
    assert dec.upload_ahead(progressive) is False and dec.reason
    assert plain.decode(progressive) is None
    assert dec.decode(progressive) is None and dec.reason == plain.reason


def test_guard_bands(corpus):
    d = egj.Decoder(canary=True)
    plain = egj.Decoder()
    for name, data, expected in corpus[::9]:
        d.upload_ahead(data)
        assert _same(d.decode(data), plain.decode(data)), name


def test_reader_thread_stages_the_next_file(corpus):
    """Two decoders take turns: while one decodes, a reader thread stages the next file on the other."""
    plain = egj.Decoder()
    files = [d for n, d, e in corpus if e][::4]
    decs = [egj.Decoder(), egj.Decoder()]
    decs[0].upload_ahead(files[0])
    for k, data in enumerate(files):
        t = None
        if k + 1 < len(files):
            t = threading.Thread(target=decs[(k + 1) % 2].upload_ahead, args=(files[k + 1],))
            t.start()
        got = decs[k % 2].decode(data)
        if t:
            t.join()
        assert _same(got, plain.decode(data)), k


def test_staged_files_that_are_decoded_a_second_time(redo_files, redos, tmp_path):
    """Staged files whose quick passes do not settle (the second decode starts from the staged copy, or reads the
    data again): decode() and decode_async() after upload_ahead(), from memory and from a path, output options
    with another decode in between."""
    plain, dec = egj.Decoder(), egj.Decoder()
    opts = dict(dtype="float32", layout="CHW", scale=1 / 255)
    for name, data in redo_files:
        want = reference(data)
        p = tmp_path / name
        p.write_bytes(data)
        ref = plain.decode(data, **opts).get()
        for src in (data, str(p)):
            del redos[:]
            assert dec.upload_ahead(src), (name, dec.reason)
            got = dec.decode(src)
            assert len(redos) == 1, name
            assert got is not None and np.array_equal(got.get(), want), (name, dec.reason)

            assert dec.upload_ahead(src)
            job = dec.decode_async(src)
            handed = job.out
            img = job.result()
            assert job.redone and img is handed and np.array_equal(handed.get(), want), name

            assert dec.upload_ahead(src)
            job = dec.decode_async(src, **opts)
            handed = job.out
            assert dec.upload_ahead(redo_files[0][1])       # staging the next file concludes the job
            other = dec.decode(redo_files[0][1], "GRAY")
            assert job.redone and other is not None
            r = job.result()
            assert r is handed and r.dtype == np.float32 and np.array_equal(r.get(), ref), name
            assert np.array_equal(other.get(), reference(redo_files[0][1], "GRAY")), name
