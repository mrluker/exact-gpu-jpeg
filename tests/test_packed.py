"""The packed coefficient form (internal): the test vectors through the GPU packer and unpacker; the GPU packer's
bytes against tests/packref.py, a second implementation of the layout; the unpacked arrays against the Huffman decode's; a forged
copy for every validity rule (rejected, nothing read or written outside a buffer, the CUDA context alive); and
the "packed" handle form end to end."""
import hashlib
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _packed, _repeat  # noqa: E402
from conftest import reference, sized  # noqa: E402
import packref  # noqa: E402

RANGE, LAYOUT = _packed.ERR_PK_RANGE, _packed.ERR_PK_LAYOUT


def _jpeg(w, h, seed=0, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


class Gpu:
    """The packed kernels on one decoder's stream, with its allocator (guard bands when the decoder has them)."""

    def __init__(self, canary=True):
        self.dec = egj.Decoder(canary=canary)

    def pack(self, coef, nq, dc):
        d = self.dec
        with d.device, d.stream:
            c, n, x = cp.asarray(np.ascontiguousarray(coef, np.int16).reshape(-1)), cp.asarray(nq), cp.asarray(dc)
            made = _packed.pack(d._alloc, c, n, x, len(nq), d.stream)
        d._check_guards()
        return made

    def upload(self, idx, data):
        """A body from the host as a device copy: the index words, and the data with its zero tail."""
        d = self.dec
        with d.device, d.stream:
            i = d._alloc(len(idx), cp.uint32)
            i.set(np.ascontiguousarray(idx, np.uint32))
            buf = d._alloc(len(data) + _packed.TAIL, cp.uint8)
            buf.set(np.concatenate([np.frombuffer(data, np.uint8), np.zeros(_packed.TAIL, np.uint8)]))
            d.stream.synchronize()
        return i, buf, len(data)

    def unpack(self, idx, data, dlen, nblk):
        """(coef, nq, dc, error bits of the unpack, error bits of the layout check); guard bands checked."""
        d = self.dec
        with d.device, d.stream:
            coef, nq, dc = d._alloc(nblk * 64, cp.int16), d._alloc(nblk, cp.uint8), d._alloc(nblk, cp.int32)
            coef.fill(0)
            e1, e2 = d._alloc(1, cp.int32, zero=True), d._alloc(1, cp.int32, zero=True)
            _packed.unpack(idx, data, dlen, nblk, coef, dc, nq, e1)
            _packed.check_layout(idx, data, dlen, nblk, e2)
            d.stream.synchronize()
            out = cp.asnumpy(coef).reshape(-1, 64), cp.asnumpy(nq), cp.asnumpy(dc), int(e1.get()[0]), int(e2.get()[0])
        d._check_guards()
        cp.cuda.Device().synchronize()                         # a fault would be reported here at the latest
        return out


@pytest.fixture(scope="module")
def gpu():
    return Gpu()


def _arrays_hash(coef, nq, dc):
    return hashlib.sha256(np.asarray(coef, "<i2").tobytes() + np.asarray(nq, np.uint8).tobytes()
                          + np.asarray(dc, "<i4").tobytes()).hexdigest()


@pytest.mark.parametrize("vector", packref.VECTORS, ids=[f"seed{v[0]}" for v in packref.VECTORS])
def test_the_formats_test_vectors(gpu, vector):
    seed, B, wide, wides, nbytes, h_arrays, h_body, cv = vector
    coef, nq, dc = packref.arrays(seed, B, wide)
    assert _arrays_hash(coef, nq, dc) == h_arrays
    idx, data, nw = packref.pack(coef, nq, dc)                 # the packer written from the text
    ref_body = idx.astype("<u4").tobytes() + data
    assert nw == wides and len(ref_body) == nbytes and hashlib.sha256(ref_body).hexdigest() == h_body
    assert _repeat.check_value(ref_body).hex() == cv
    assert _packed.check_body(idx, data, B) is None
    made = gpu.pack(coef, nq, dc)                              # the GPU packer: the same bytes
    assert made is not None and _packed.body(*made) == ref_body
    c2, n2, d2, e1, e2 = gpu.unpack(made[0], made[1], made[2], B)
    assert (e1, e2) == (0, 0)                                  # the unpack and the layout check accept it
    assert _arrays_hash(packref.zero_past_nq(c2, n2), n2, d2) == h_arrays
    c3, n3, d3 = packref.unpack(idx, data, B)                  # and the unpacker written from the text
    assert _arrays_hash(c3, n3, d3) == h_arrays


def test_the_gpu_packer_and_unpacker_against_the_decodes_own_arrays(corpus, gpu):
    """For decoded files: the GPU packer writes the bytes the reference packer writes from the decode's arrays, and
    the unpack gives those arrays back (every DC, every quad count, every quad the IDCT reads)."""
    dec = egj.Decoder(keep_coefs="raw")
    n = 0
    for name, data, expected in corpus[::4]:
        if dec.decode(data) is None:
            continue
        h = dec.take_coefs()
        coef, nq, dc = (cp.asnumpy(a) for a in h._parts)
        B = h._nblk
        coef = packref.zero_past_nq(coef, nq)
        assert not coef[:, 0].any(), name                      # position 0 is the DC's place: never a stored value
        idx, body_data, _ = packref.pack(coef, nq, dc)
        made = gpu.pack(coef, nq, dc)
        assert made is not None and _packed.body(*made) == idx.astype("<u4").tobytes() + body_data, name
        assert _packed.check_body(idx, body_data, B) is None, name
        c2, n2, d2, e1, e2 = gpu.unpack(made[0], made[1], made[2], B)
        assert (e1, e2) == (0, 0), name
        assert np.array_equal(n2, nq) and np.array_equal(d2, dc) and np.array_equal(packref.zero_past_nq(c2, n2), coef), name
        n += 1
    assert n >= 60


def test_the_packed_handle_gives_the_same_image(corpus):
    dec, other = egj.Decoder(keep_coefs="packed"), egj.Decoder()
    raw = egj.Decoder(keep_coefs="raw")
    n = 0
    for name, data, expected in corpus[::3]:
        img = dec.decode(data)
        h = dec.take_coefs()
        if img is None:
            assert h is None, name
            continue
        assert h.form == "packed" and not h.freed and h.nbytes > 0, name
        want = img.get()
        assert np.array_equal(want, reference(data)), name
        for d in (dec, other):
            again = d.decode_from_handle(h)
            assert again is not None and np.array_equal(again.get(), want), (name, d.reason)
        for cs in ("BGR", "GRAY"):
            assert np.array_equal(other.decode_from_handle(h, cs).get(), reference(data, cs)), (name, cs)
        got = other.decode_from_handle(h, dtype="float16", layout="CHW", scale=1 / 255)
        assert np.array_equal(got.get().view(np.uint8), other.decode(data, dtype="float16", layout="CHW", scale=1 / 255).get().view(np.uint8))
        n += 1
    assert n >= 80
    photo = io.BytesIO()                                       # a photo-like image: the packed copy is far smaller
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
    from reproduce import synthetic_photo
    Image.fromarray(synthetic_photo(1200, 800, 1)).save(photo, "JPEG", quality=90)
    assert dec.decode(photo.getvalue()) is not None and raw.decode(photo.getvalue()) is not None
    h, hr = dec.take_coefs(), raw.take_coefs()
    assert h.form == "packed" and hr.form == "raw" and h.nbytes < hr.nbytes / 2, (h.nbytes, hr.nbytes)
    tm = {}
    other.profile = True
    assert other.decode_from_handle(h, timings=tm) is not None and {"unpack", "idct", "colour"} <= set(other.stats["profile_ms"])
    canary = egj.Decoder(canary=True, keep_coefs="packed")     # guard bands around the packer's and unpacker's buffers
    for name, data, expected in sized(corpus, 12):
        assert canary.decode(data) is not None
        assert np.array_equal(canary.decode_from_handle(canary.take_coefs()).get(), reference(data)), name


def _valid_body():
    """A body on the host with every kind of block in it: (index words, data bytes, blocks, coef, nq, dc), from the
    arrays of the second formula vector (blocks without values, with one value, with an odd number of bytes, wide)."""
    coef, nq, dc = packref.arrays(2, 4096, False)
    idx, data, _ = packref.pack(coef, nq, dc)
    return idx, bytearray(data), 4096, coef, nq, dc


def _fields(idx, b):
    h = int(idx[b])
    return h & 0x0FFFFFFF, (h >> 28) & 7, h >> 31


def _forgeries(idx, data, B, coef, nq):
    """(name, index words, data, the unpack must flag it, the layout check must flag it, host rule) per rule."""
    out = []
    dlen = len(data)
    k = B // 2
    cnt = (coef.reshape(B, 64) != 0).sum(1)

    def case(name, unpack_bits, layout_bits, rule, i=None, d=None):
        out.append((name, idx.copy() if i is None else i, bytes(data if d is None else d), unpack_bits, layout_bits, rule))

    i = idx.copy(); i[k] = (i[k] & 0xF0000000) | (dlen + 2)
    case("an offset past the data", RANGE, RANGE, "past the data", i)
    i = idx.copy(); i[k] |= 1                                   # the forgery that once faulted a kernel
    case("an odd offset inside the data", RANGE, RANGE, "odd", i)
    for n in (5, 6, 7):
        i = idx.copy(); i[k] = (i[k] & 0x8FFFFFFF) | (n << 28)
        case(f"{n} quads with a valid offset", RANGE, RANGE, "more than 4", i)
    i = idx.copy(); i[0] = (i[0] & 0xF0000000) | 2
    case("the first block not at 0", 0, LAYOUT, "start at 0", i)
    i = idx.copy(); i[k] += 2
    case("a block moved by two bytes", 0, LAYOUT, None, i)
    b = next(b for b in range(B) if nq[b] >= 1 and (int(idx[b]) >> 31) == 0)
    off, n, _ = _fields(idx, b)
    d = bytearray(data); d[off + 2 + 2 * (n - 1)] = 0; d[off + 3 + 2 * (n - 1)] = 0
    case("the last quad's mask emptied", 0, LAYOUT, None, None, d)
    d = bytearray(data); d[off + 2] |= 1
    case("position 0 stored as a value", 0, LAYOUT, None, None, d)
    z = next((b for b in range(B) if nq[b] == 0), None)
    if z is not None:
        i = idx.copy(); i[z] |= 0x80000000
        case("a block without values flagged wide", 0, LAYOUT, "without values", i)
    one = next((b for b in range(B) if cnt[b] == 1 and nq[b] >= 1 and 0 < coef.reshape(B, 64)[b].max() <= 127), None)
    if one is not None:                                        # same size either way: only the wide rule sees it
        i = idx.copy(); i[one] |= 0x80000000
        case("a narrow block with one value flagged wide", 0, LAYOUT, "no wide value", i)
    p = next((b for b in range(B) if (2 + 2 * int(nq[b]) + int(cnt[b]) * (2 if int(idx[b]) >> 31 else 1)) % 2 == 1), None)
    if p is not None:
        off, n, w = _fields(idx, p)
        d = bytearray(data); d[off + 2 + 2 * n + int(cnt[p]) * (2 if w else 1)] = 0x5A
        case("a pad byte that is not 0", 0, LAYOUT, "pad byte", None, d)
    return out


def test_a_forged_copy_for_every_rule_is_rejected_and_harms_nothing(gpu):
    idx, data, B, coef, nq, dc = _valid_body()
    assert _packed.check_body(idx, bytes(data), B) is None
    i, d, dlen = gpu.upload(idx, bytes(data))
    c, n, x, e1, e2 = gpu.unpack(i, d, dlen, B)                # the unchanged copy: accepted, the arrays back
    assert (e1, e2) == (0, 0) and np.array_equal(n, nq) and np.array_equal(x, dc)
    assert np.array_equal(packref.zero_past_nq(c, n), coef)
    cases = _forgeries(idx, data, B, coef, nq)
    names = [c[0] for c in cases]
    assert len(cases) == 12, names                             # every rule of the layout has its forgery
    for name, fi, fd, want_unpack, want_layout, rule in cases:
        why = _packed.check_body(fi, fd, B)                    # the host's validator names a rule for each
        assert why is not None and (rule is None or rule in why), (name, why)
        i, d, dlen = gpu.upload(fi, fd)
        c, n, x, e1, e2 = gpu.unpack(i, d, dlen, B)            # no fault, guard bands intact (checked inside)
        assert e1 & RANGE == want_unpack & RANGE, (name, e1)
        assert e2 & (RANGE | LAYOUT), (name, e2)
        assert (e2 & want_layout) == want_layout, (name, e2)
    # file-level rules, which only the host sees
    assert "odd" in _packed.check_body(idx, bytes(data) + b"\0", B)
    assert "blocks" in _packed.check_body(idx[:-1], bytes(data), B)
    assert "more data" in _packed.check_body(idx[:1], bytes(data) + bytes(200), 1)
    good = _jpeg(320, 240, 5)
    assert np.array_equal(gpu.dec.decode(good).get(), reference(good))      # the decoder and the context are fine


def test_a_damaged_packed_handle_is_declined_not_decoded(corpus):
    name, data, expected = next(c for c in corpus if c[2] and len(c[1]) > 4000)
    dec = egj.Decoder(keep_coefs="packed")
    want = dec.decode(data).get()
    h = dec.take_coefs()
    idx = h._parts[0]
    old = int(idx[1].get())
    idx[1] = old | 1                                           # an odd offset, written into the copy on the GPU
    cp.cuda.get_current_stream().synchronize()
    assert dec.decode_from_handle(h) is None and "out of range" in dec.reason
    idx[1] = old
    cp.cuda.get_current_stream().synchronize()
    assert np.array_equal(dec.decode_from_handle(h).get(), want)           # and usable again once it is whole
    assert np.array_equal(dec.decode(data).get(), want)


def test_the_cache_keeps_packed_copies_by_default(corpus):
    files = [c[1] for c in sized(corpus, 20)]
    sizes = {}
    for form in ("packed", "raw"):
        cache = egj.RepeatCache(min_pixels=0, form=form)
        dec = egj.Decoder(cache=cache)
        for f in files:
            for _ in range(2):
                assert np.array_equal(dec.decode(f).get(), reference(f))
        assert cache.stats["used"] >= len(files) and cache.form == form
        sizes[form] = cache.stats["bytes"]
    assert egj.RepeatCache().form == "packed"
    with pytest.raises(ValueError):
        egj.RepeatCache(form="zip")
