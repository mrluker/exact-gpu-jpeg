"""decode_batch: every image equals what decode() returns for it (and so libjpeg-turbo's), None where decode()
declines, whatever is decoded beside it."""
import io
import os
import random
import sys

import numpy as np
import pytest

pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import reference, sized  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fuzz import mutate  # noqa: E402


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())


def _same_reason(a, b):
    """A reason from the header is the same text. For bad entropy-coded data both must name errors found in the
    data; which ones can differ (blocks that were never decoded hold whatever was in memory, and after a damaged
    image's last bit come zeros in one path and the next image in the other; the image is declined either way)."""
    from exact_gpu_jpeg._decoder import ERRORS
    known = set(ERRORS.values())

    def data_level(r):
        return r is not None and all(x in known for x in r.split("; "))
    return a == b or (data_level(a) and data_level(b))


def _check_batch(dec, files, single=None, **kw):
    """Batch output against decode() one by one (same decoder settings); returns how many went through batch kernels."""
    one = single or egj.Decoder(canary=dec.canary, fast_upsample=dec.fast_upsample)
    got = dec.decode_batch(files, **kw)
    n = dec.stats["batched"]
    reasons = list(dec.reasons)
    assert len(got) == len(files) == len(reasons)
    for k, data in enumerate(files):
        want = one.decode(data, **kw)
        assert _same(want, got[k]), (k, one.reason, reasons[k])
        assert (got[k] is None) == (reasons[k] is not None), (k, reasons[k])
        if want is None:
            assert _same_reason(reasons[k], one.reason), (k, reasons[k], one.reason)
    return n


def _pil(rng, w, h, ss=2, q=90, **kw):
    from PIL import Image
    a = (rng.random((h // 8 + 1, w // 8 + 1, 3)) * 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=q, subsampling=ss, **kw)
    return b.getvalue()


def test_corpus_in_one_batch(corpus, dec):
    """The whole corpus in one call: every subsampling, size from 1x1, quality, restart interval, optimised tables,
    files the GPU declines; and the same against libjpeg-turbo."""
    files = [d for _, d, _ in corpus]
    n = _check_batch(dec, files)
    assert n > 150                                    # most of it really went through the batch kernels
    got = dec.decode_batch(files)
    for (name, data, expected), g in zip(corpus, got):
        assert (g is not None) == expected, name
        if g is not None:
            assert np.array_equal(g.get(), reference(data)), name


def test_orders_and_sizes(corpus, dec):
    rng = random.Random(7)
    files = [d for _, d, _ in corpus]
    for size in (2, 3, 5, 17, 64):
        for _ in range(4):
            _check_batch(dec, [rng.choice(files) for _ in range(size)])
    assert dec.decode_batch([]) == []
    one = files[0]
    assert _same(dec.decode_batch([one])[0], dec.decode(one))
    _check_batch(dec, [one] * 9)                      # the same file many times


@pytest.mark.parametrize("colorspace", ["BGR", "GRAY"])
def test_colorspaces(corpus, dec, colorspace):
    _check_batch(dec, [d for _, d, _ in corpus[::5]], colorspace=colorspace)


def test_output_options(corpus, dec):
    files = [d for _, d, _ in corpus[::9]]
    _check_batch(dec, files, dtype="float32")
    _check_batch(dec, files, dtype="float16", layout="CHW", scale=[1 / 58.4, 1 / 57.1, 1 / 57.4], bias=[-2.1, -2.0, -1.8])
    _check_batch(dec, files, colorspace="GRAY", dtype="float32", layout="CHW", scale=1 / 255)


def test_fast_upsample_batch(corpus):
    d = egj.Decoder(fast_upsample=True)
    assert _check_batch(d, [x for _, x, _ in corpus[::6]]) > 10


def test_fast_dct_goes_one_by_one(corpus):
    d = egj.Decoder(fast_dct=True)
    files = [x for _, x, _ in corpus[::25]]
    got = d.decode_batch(files)
    assert d.stats["batched"] == 0
    one = egj.Decoder(fast_dct=True)
    for data, g in zip(files, got):
        assert _same(one.decode(data), g)


def test_bad_files_do_not_touch_their_neighbours(corpus, dec):
    """Truncated and mutated files between good ones: the good ones are exact, the bad ones as decode() has them."""
    rng = random.Random(11)
    good = [d for n, d, e in corpus if e and len(d) < 60000]
    for _ in range(12):
        files = []
        for _ in range(24):
            d = rng.choice(good)
            r = rng.random()
            files.append(d if r < 0.4 else d[:rng.randrange(len(d))] if r < 0.6 else mutate(d, rng))
        _check_batch(dec, files)


def test_restart_markers_and_plain_files_together(dec):
    rng = np.random.default_rng(3)
    files = [_pil(rng, 640, 480), _pil(rng, 640, 480, restart_marker_rows=1), _pil(rng, 333, 211, restart_marker_blocks=5),
             _pil(rng, 17, 9), _pil(rng, 1024, 768, q=60, restart_marker_rows=2), _pil(rng, 8, 8), _pil(rng, 800, 600)]
    assert _check_batch(dec, files) == len(files)


def test_data_after_the_end_marker(dec):
    rng = np.random.default_rng(4)
    a, b, c = _pil(rng, 96, 64), _pil(rng, 200, 120), _pil(rng, 32, 24)
    files = [a + c, b, a + b"\0" * 5000, c + b"trailer", b + a, a]
    _check_batch(dec, files)


def test_guard_bands(corpus):
    d = egj.Decoder(canary=True)
    files = [x for _, x, _ in corpus[::3]]
    assert _check_batch(d, files) > 40
    rng = random.Random(2)
    good = [x for n, x, e in corpus if e and len(x) < 40000]
    _check_batch(d, [mutate(rng.choice(good), rng) for _ in range(200)])


def test_top_level_api(corpus):
    sub = [(n, d, e) for n, d, e in corpus[::4] if reference(d) is not None][:60]   # ones a CPU decoder reads
    files = [d for _, d, _ in sub]
    info = []
    got = egj.decode(files, info=info)                # a list: decode_batch
    assert len(got) == len(info) == len(files)
    for (name, data, expected), g, i in zip(sub, got, info):
        assert i["path"] == ("gpu" if expected else "cpu"), name
        assert g is not None and np.array_equal(g.get(), reference(data)), name
    got = egj.decode_batch(files, fallback=False)
    assert [g is not None for g in got] == [e for _, _, e in sub]


def test_images_stay_valid_after_the_next_batch(corpus, dec):
    files = [d for n, d, e in sized(corpus, 20)]
    first = dec.decode_batch(files)
    keep = [g.get() for g in first]
    dec.decode_batch(files[::-1])
    for g, k in zip(first, keep):
        assert np.array_equal(g.get(), k)


def test_same_header_different_pixels_and_input_types(dec):
    """Files whose headers are byte-identical (same encoder, size and settings) share one parsed header; their
    pixels differ. bytes, bytearray, memoryview and NumPy inputs."""
    rng = np.random.default_rng(8)
    files = [_pil(rng, 320, 240) for _ in range(12)] + [_pil(rng, 320, 240, q=70) for _ in range(5)]
    assert len({f[:f.find(b"\xff\xda")] for f in files}) == 2            # two distinct headers
    assert _check_batch(dec, files) == len(files)
    mixed = [files[0], bytearray(files[1]), memoryview(files[2]), np.frombuffer(files[3], np.uint8), files[4]]
    got = dec.decode_batch(mixed)
    one = egj.Decoder()
    for k in range(5):
        assert _same(got[k], one.decode(files[k])), k


def test_paths(corpus, dec, tmp_path):
    files = []
    for k, (name, data, expected) in enumerate(corpus[::13]):
        p = tmp_path / f"f{k}.jpg"
        p.write_bytes(data)
        files.append(p if k % 2 else str(p))
    got = dec.decode_batch(files)
    one = egj.Decoder()
    for p, g in zip(files, got):
        assert _same(g, one.decode(p))


def test_cached_headers_and_tables_equal_the_parsed_ones(corpus):
    """The batch path's header cache and its list-built geometry give exactly parse() and _geometry()."""
    from dataclasses import asdict

    from exact_gpu_jpeg import _batch, _decoder
    from exact_gpu_jpeg._header import Unsupported, parse
    rng = random.Random(3)
    good = [d for n, d, e in corpus if len(d) < 60000]
    files = [d for _, d, _ in corpus] * 2 + [mutate(rng.choice(good), rng) for _ in range(1500)]
    files += [d[:rng.randrange(len(d))] for d in good[:150]]
    hits = 0
    for data in files:
        try:
            want = parse(data)
        except Unsupported as e:
            with pytest.raises(Unsupported) as ex:
                _batch._parse(data)
            assert str(ex.value) == str(e)
            continue
        before = len(_batch._hdrs)
        got = _batch._parse(data)
        hits += len(_batch._hdrs) == before
        a, b = asdict(want), asdict(got)
        assert a.keys() == b.keys()
        for k in a:
            if k == "quant":
                assert all(np.array_equal(x, y) for x, y in zip(a[k], b[k]))
            else:
                assert a[k] == b[k], k
        for fancy in (True, False):
            T = _decoder._tables(want)
            g0, g1 = _decoder._geometry(want, T["blocks"], fancy), _batch._geom(got, T, fancy)
            assert g0["pix"].tolist() == g1["pix"] and g0["up"].tolist() == g1["up"]
            assert np.array_equal(g0["q"], g1["q"])
            assert (g0["nmcu"], g0["nblk"], g0["plane_bytes"]) == (g1["nmcu"], g1["nblk"], g1["plane_bytes"])
    assert hits > 300                                   # the cache really was used


def test_batch_drops_a_staged_file(corpus, dec):
    """upload_ahead() stages one file in the decoder's buffer; a batch reuses that buffer, so the staged file is
    dropped and decoded afresh afterwards."""
    good = [d for n, d, e in corpus if e and len(d) > 3000]
    a = good[7]
    one = egj.Decoder()
    assert dec.upload_ahead(a)
    _check_batch(dec, good[10:30])
    assert _same(dec.decode(a), one.decode(a))
    assert dec.upload_ahead(a)
    assert _same(dec.decode_batch([a, good[3], a])[2], one.decode(a))
    assert _same(dec.decode(a), one.decode(a))


def test_files_that_are_decoded_a_second_time(corpus, redo_files, redos, dec):
    """The corpus files that a single decode does a second time, in batches: libjpeg-turbo's pixels several of a
    kind, all kinds together, and among ordinary files (whose pixels they must not touch). In a batch every image
    starts from a known state and is followed by one, which can save a pass: the smallest file settles in the
    quick passes there (decoded as a batch, no second decode); the largest cannot (it fills more than three CUDA
    blocks of chunks), so its batch is decoded one by one, and each file then a second time by the single-image
    path. Which of the two the middle one takes depends on where its chunks fall in the blocks."""
    files = [d for n, d in redo_files]
    good = [d for n, d, e in corpus if e and 3000 < len(d) < 60000 and not n.startswith("redo_")][:6]
    seen = []
    for k, batch in enumerate(([files[0]] * 3, [files[1]] * 2, [files[2]] * 2, files * 2,
                               good[:3] + files + good[3:])):
        del redos[:]
        got = dec.decode_batch(batch)
        for d, g in zip(batch, got):
            assert g is not None and np.array_equal(g.get(), reference(d)), (k, dec.reasons)
        seen.append((dec.stats["batched"], len(redos)))
    assert seen[0] == (3, 0), seen                           # settles as a batch
    assert seen[1] in ((2, 0), (0, 2)), seen
    assert seen[2] == (0, 2), seen                           # one by one, each a second time
    again = egj.decode(good[:2] + files)                    # the top-level call with a list
    for d, g in zip(good[:2] + files, again):
        assert np.array_equal(np.asarray(g.get() if hasattr(g, "get") else g), reference(d))
