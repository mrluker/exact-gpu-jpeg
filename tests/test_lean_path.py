"""The kernels that replaced groups of small launches in the single-image paths give the numbers the replaced code
gave: dc_scan against cumulative sums (NumPy), first_blocks against the prefix-sum expression, the encoder's
enc_offsets against a cumulative sum; and the encoder's one-wait download gives Pillow's bytes on every route."""
import io

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from exact_gpu_jpeg import _encoder as E  # noqa: E402

BLOCKS = {
    "420": [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (2, 0, 0)],
    "444": [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
    "422": [(0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 0, 0)],
    "gray": [(0, 0, 0)],
    "411": [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 0, 0), (2, 0, 0)],
}


def dc_reference(d, blocks, nmcu, dri):
    """Per component, the wrapping 32-bit running sum over its blocks in scan order, restarting every dri MCUs."""
    bpm = len(blocks)
    x = d.reshape(nmcu, bpm).astype(np.uint32)
    out = np.empty_like(x)
    comp = [b[0] for b in blocks]
    step = dri or nmcu
    nseg = -(-nmcu // step)
    for ci in sorted(set(comp)):
        cols = [k for k in range(bpm) if comp[k] == ci]
        xp = np.zeros((nseg * step, len(cols)), np.uint32)
        xp[:nmcu] = x[:, cols]
        xp = np.cumsum(xp.reshape(nseg, step * len(cols)), axis=1, dtype=np.uint32).reshape(nseg * step, len(cols))
        out[:, cols] = xp[:nmcu]
    return out.view(np.int32).reshape(-1)


# (DC_ONE_BLOCK, DC_RUN, DC_BLOCKS): the library's values, then values that make small images take many CUDA blocks
@pytest.mark.parametrize("limits", [None, (0, 1, 3), (0, 1, 256), (0, 2, 7)])
@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_dc_scan_equals_the_cumulative_sums(name, limits, monkeypatch):
    if limits:
        for k, v in zip(("DC_ONE_BLOCK", "DC_RUN", "DC_BLOCKS"), limits):
            monkeypatch.setattr(D, k, v)
    blocks = BLOCKS[name]
    bpm = len(blocks)
    rng = np.random.default_rng(5)
    dec = egj.Decoder()
    sizes = (1, 2, 7, 1023, 1024, 1025, 2049, 4096, 4097, 40_000, 300_007) if limits is None else (1, 2, 1025, 5000, 9001)
    with dec.device:
        for nmcu in sizes:
            for dri in (0, 1, 3, 1000, 1024, 4096, nmcu, nmcu + 5):
                big = nmcu < 3000                      # also values that wrap 32 bits
                d = rng.integers(-2**31 if big else -2047, 2**31 if big else 2048, nmcu * bpm,
                                 dtype=np.int64).astype(np.int32)
                got = cp.asarray(d)
                dec._dc_prediction(got, dict(bpm=bpm, nmcu=nmcu, blocks=blocks), dri)
                assert np.array_equal(got.get(), dc_reference(d, blocks, nmcu, dri)), (name, nmcu, dri)


def test_first_blocks_equals_the_prefix_sum_expression():
    rng = np.random.default_rng(6)
    for nch, nseg, extra in ((1, 1, 0), (5, 1, 3), (1024, 1, 0), (1025, 3, 7), (5000, 40, 100), (60_000, 900, 1)):
        nmax = nch + extra
        cut = np.sort(rng.choice(np.arange(1, nch), nseg - 1, replace=False)) if nseg > 1 else np.array([], np.int64)
        cfirst = np.concatenate([[0], cut, [nch]]).astype(np.int32)           # first chunk of each interval; total
        cseg = np.zeros(nmax, np.int32)
        cseg[:nch] = np.searchsorted(cfirst[1:], np.arange(nch), side="right")
        cnt = rng.integers(0, 4097, nmax).astype(np.int32)                    # past nch: values that must not count
        seglen = int(rng.integers(1, 5000))
        c = cnt.copy()
        c[nch:] = 0
        excl = np.concatenate([[0], np.cumsum(c, dtype=np.int64)])
        want = cseg.astype(np.int64) * seglen + excl[:nmax] - excl[cfirst[cseg]]
        first = cp.full(nmax, -7, cp.int64)
        cfd = cp.asarray(cfirst)
        D._k("first_blocks")((1,), (1024,), (cp.asarray(cnt), cfd[nseg:], cp.asarray(cseg), cfd, np.int64(seglen), first,
                                              np.int32(nmax)))
        got = first.get()
        assert np.array_equal(got[:nch], want[:nch]), (nch, nseg)
        assert (got[nch:] == -7).all()                                        # nothing written past the chunks


def test_enc_offsets_equals_the_cumulative_sum():
    rng = np.random.default_rng(7)
    for n in (1, 2, 31, 1023, 1024, 1025, 5000, 65_536, 70_001):
        for hi in (2000, 2**31 - 1):                                          # sums past 32 bits too
            v = rng.integers(0, hi, n, dtype=np.int64).astype(np.int32)
            off = cp.full(n + 3, -7, cp.int64)
            E._k("enc_offsets")((1,), (1024,), (cp.asarray(v), np.int32(n), off))
            got = off.get()
            assert got[0] == 0 and np.array_equal(got[1:n + 1], np.cumsum(v, dtype=np.int64)), (n, hi)
            assert (got[n + 1:] == -7).all()


def _pil(a, q):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2)
    return b.getvalue()


@pytest.mark.parametrize("first", [None, 64, 1 << 12])
def test_one_wait_download_gives_pillows_bytes(first, monkeypatch):
    """Data shorter than, equal to and longer than the piece downloaded with the size words; a page-locked buffer
    that has to grow after that piece arrived; many encodes on one encoder."""
    if first:
        monkeypatch.setattr(E, "FIRST", first)
    rng = np.random.default_rng(8)
    enc = egj.Encoder()
    for w, h, q in ((16, 16, 75), (64, 48, 90), (8, 8, 1), (640, 480, 95), (33, 17, 50), (1600, 1200, 98), (16, 16, 75)):
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        assert enc.encode(a, quality=q) == _pil(a, q), (w, h, q, first)
    a = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    want = _pil(a, 85)
    for _ in range(50):
        assert enc.encode(cp.asarray(a), quality=85) == want


@pytest.mark.parametrize("ss", ["420", "422", "444"])
def test_small_image_encode_gives_pillows_bytes(ss, monkeypatch):
    """The few-launch route for small images (enc_plan, stuff_small): sizes around its limit, one block up to many
    tiles of output, flat and noisy content, and the same bytes as the general route."""
    rng = np.random.default_rng(9)
    SS = {"420": 2, "422": 1, "444": 0}
    enc = egj.Encoder()
    # the last four straddle the route's limit of 16384 blocks in 4:2:0 and in 4:4:4
    cases = [(8, 8), (1, 1), (17, 9), (64, 64), (200, 120), (416, 416), (640, 64), (832, 832), (848, 832), (584, 592),
             (592, 592)]
    for w, h in cases:
        for kind in ("noise", "flat", "smooth"):
            if kind == "noise":
                a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            elif kind == "flat":
                a = np.full((h, w, 3), 255, np.uint8)
            else:
                g = rng.integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
                a = np.asarray(Image.fromarray(g).resize((w, h), Image.BICUBIC))
            for q in (30, 90, 100):
                b = io.BytesIO()
                Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=SS[ss])
                got = enc.encode(a, quality=q, subsampling=ss)
                assert got == b.getvalue(), (w, h, kind, q, ss)
                monkeypatch.setattr(E, "SMALL_BLOCKS", 0)
                assert enc.encode(a, quality=q, subsampling=ss) == got, (w, h, kind, q, ss, "general route")
                monkeypatch.undo()
