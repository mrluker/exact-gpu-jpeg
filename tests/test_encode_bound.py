"""When an image needs more room than the encoder's output bound, nothing may be read or written outside a device
buffer before the host notices and encodes again, and the second encode must give Pillow's bytes.

The bound is forced far too small (and just too small) through Encoder._queue's own `cap` argument. Every device
array the encoder allocates is given guard bands on both sides, filled with a poison byte; arrays the encoder
leaves uninitialised (cp.empty) are filled with a second poison byte. Then:
  - writes outside a buffer: the guard bands must be unchanged after every pass;
  - reads outside a buffer (or of uninitialised memory): everything the kernels compute in the pass that overflows
    (the block bit counts, the offsets, the tile counts and offsets, the flags and the four words the host reads)
    must be the same whatever the two poison bytes are. A kernel that read past a buffer's end would count or
    copy poison, and those values would change with it.
A read whose value reaches nothing cannot be seen this way; the code reading is the argument for those."""
import io

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _encoder as E  # noqa: E402

G = 8192                                               # guard bytes on each side (more than one 4096-byte tile)


class Arena:
    """cp.empty / cp.zeros replaced: the array asked for sits between two guard bands."""

    def __init__(self, guard_byte, fill_byte):
        self.guard, self.fill, self.made = guard_byte, fill_byte, []
        self.real_empty, self.real_zeros = cp.empty, cp.zeros

    def _make(self, shape, dtype, zero):
        dt = np.dtype(dtype)
        n = int(np.prod(shape)) if not np.isscalar(shape) else int(shape)
        g = -(-G // dt.itemsize)
        base = self.real_empty(n + 2 * g, dt)
        raw = base.view(cp.uint8)
        raw[:] = self.guard
        mid = base[g:g + n]
        mid.view(cp.uint8)[:] = 0 if zero else self.fill
        self.made.append((base, g, n, dt))
        return mid.reshape(shape) if not np.isscalar(shape) else mid

    def empty(self, shape, dtype=float, **kw):
        return self._make(shape, dtype, False)

    def zeros(self, shape, dtype=float, **kw):
        return self._make(shape, dtype, True)

    def guards_intact(self):
        for base, g, n, dt in self.made:
            raw = cp.asnumpy(base.view(cp.uint8))
            lo, hi = raw[:g * dt.itemsize], raw[(g + n) * dt.itemsize:]
            if (lo != self.guard).any() or (hi != self.guard).any():
                return False, (n, str(dt), int((lo != self.guard).sum()), int((hi != self.guard).sum()))
        return True, None


def run(a, quality, ss, first_cap, guard_byte, fill_byte, enc):
    """One encode with the first pass's bound forced to first_cap bytes, on the Encoder given (one for a whole
    test: CuPy's pool keeps freed blocks per stream, so a new Encoder for each of the 36 encodes of a case would
    leave 36 sets of buffers in the pool). Returns the file, the passes' computed values, and whether every guard
    band is intact."""
    arena = Arena(guard_byte, fill_byte)
    passes = []
    real_queue = E.Encoder._queue

    def queue(self, img, W, H, q, samp, timings, hdr, path, cap=None, attempt=0):
        forced = cap is None and first_cap is not None
        cp.empty, cp.zeros = arena.empty, arena.zeros          # (set and put back by hand: monkeypatch would keep
        try:                                                   # every arena, and all its arrays, until the test ends)
            st = real_queue(self, img, W, H, q, samp, timings, hdr, path, cap=first_cap if forced else cap,
                            attempt=attempt)
            self.stream.synchronize()
        finally:
            cp.empty, cp.zeros = arena.real_empty, arena.real_zeros
        info = cp.asnumpy(st["info"]).tolist()
        keep = st["keep"] if "keep" in st else ()
        vals = dict(info=info, cap=first_cap if forced else cap)
        over = bool(info[2] or info[3])
        for k, x in enumerate(keep):                   # coef, nb, off, buf, [tc, toff,] flags, tables
            if isinstance(x, cp.ndarray) and k != 3 and x.dtype != cp.int16:      # not buf (legitimately poison where unwritten), not coef (large)
                vals[f"keep{k}"] = cp.asnumpy(x).tobytes()
        if not over:
            vals["out"] = cp.asnumpy(st["out"][:info[0]]).tobytes()
        passes.append(vals)
        return st
    E.Encoder._queue = queue
    try:
        data = enc.encode(cp.asarray(a), quality=quality, subsampling=ss)
    finally:
        E.Encoder._queue = real_queue
    ok, where = arena.guards_intact()
    arena.made.clear()                                         # the arrays go back to the pool now
    return data, passes, ok, where


def busy(w, h, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)            # noise: the most bits a block can need


CASES = [  # (width, height, quality, subsampling as the library names it, as Pillow numbers it)
    (64, 64, 100, "444", 0), (488, 480, 100, "444", 0), (488, 480, 95, "420", 2),          # the few-launch route
    (600, 592, 100, "444", 0), (1500, 1000, 100, "444", 0), (1001, 701, 95, "422", 1),     # the general route
    (3840, 2560, 95, "444", 0),                                                            # above SCAN_BLOCKS
]


@pytest.mark.parametrize("w, h, q, ss, sub", CASES, ids=[f"{c[0]}x{c[1]} q{c[2]} {c[3]}" for c in CASES])
def test_a_bound_that_is_too_small(w, h, q, ss, sub, monkeypatch):
    a = busy(w, h, w + h)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=sub)
    want = b.getvalue()
    plain = egj.encode(cp.asarray(a), quality=q, subsampling=ss)
    assert plain == want
    enc = egj.Encoder()
    _, passes, ok, _ = run(a, q, ss, None, 0x00, 0x00, enc)                        # the library's own bound
    assert ok and len(passes) in (1, 2)                # (noise at this quality can exceed it: then it is two passes)
    need = (passes[0]["info"][1] + 7) // 8             # bytes of entropy-coded data before stuffing; the bound must be need + 1
    for first_cap in (64, 4096, need // 2, need - 1, need, need + 1, need + 4096):
        results = []
        for guard_byte, fill_byte in ((0x00, 0x00), (0xFF, 0xFF), (0xFF, 0x00), (0x00, 0xFF), (0xA5, 0x3C)):
            data, passes, ok, where = run(a, q, ss, first_cap, guard_byte, fill_byte, enc)
            assert ok, ("a guard band was written", first_cap, guard_byte, where)
            assert data == want, ("not Pillow's bytes", first_cap, guard_byte, fill_byte)
            results.append(passes)
        n_pass = {len(p) for p in results}
        assert len(n_pass) == 1, ("the number of passes depends on the poison", first_cap, n_pass)
        for k in range(len(results[0])):
            for other in results[1:]:
                assert other[k].keys() == results[0][k].keys()
                for key in results[0][k]:
                    assert other[k][key] == results[0][k][key], ("depends on the poison", first_cap, k, key)
        over = bool(results[0][0]["info"][2] or results[0][0]["info"][3])
        assert over == (first_cap < need + 1), (first_cap, need, results[0][0]["info"])
        assert len(results[0]) == (2 if over else 1)


def _pillow(a, q, sub):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=sub)
    return b.getvalue()


@pytest.mark.parametrize("w, h, q, ss, sub", [(488, 480, 95, "420", 2), (1001, 701, 95, "422", 1), (1500, 1000, 100, "444", 0)],
                         ids=["few-launch route", "general route", "general route, large"])
def test_an_output_overflow_is_retried_once_with_room_for_the_worst_case(w, h, q, ss, sub, monkeypatch, tmp_path):
    a = busy(w, h, 7)
    want = _pillow(a, q, sub)
    calls = []
    real = E._out_cap
    monkeypatch.setattr(E, "_out_cap", lambda cap, retry: calls.append(retry) or (real(cap, True) if retry else 64))
    assert egj.encode(cp.asarray(a), quality=q, subsampling=ss) == want and calls == [False, True]
    p = tmp_path / "x.jpg"
    assert egj.encode(cp.asarray(a), quality=q, subsampling=ss, path=str(p)) == len(want) and p.read_bytes() == want
    assert real(1000, True) >= 2 * 1000 + 1 and real(1000, False) < real(1000, True)


def test_no_input_can_make_the_encoder_retry_without_end(monkeypatch, tmp_path):
    """Whatever the data, an encode is two attempts at most: a second overflow raises, leaves no file behind, and
    the encoder works afterwards."""
    a = busy(488, 480, 9)
    want = _pillow(a, 90, 2)
    calls = []
    real = E._out_cap
    monkeypatch.setattr(E, "_out_cap", lambda cap, retry: calls.append(retry) or 64)        # never enough room
    with pytest.raises(RuntimeError, match="did not fit its buffer"):
        egj.encode(cp.asarray(a), quality=90, subsampling="420")
    assert calls == [False, True]
    p = tmp_path / "x.jpg"
    with pytest.raises(RuntimeError, match="did not fit its buffer"):
        egj.encode(cp.asarray(a), quality=90, subsampling="420", path=str(p))
    assert not p.exists() and calls == [False, True, False, True]
    monkeypatch.setattr(E, "_out_cap", real)
    assert egj.encode(cp.asarray(a), quality=90, subsampling="420") == want


@pytest.mark.parametrize("route", ["few-launch", "general"])
def test_data_that_is_all_ff_bytes_fits_the_retry_exactly(route):
    """The stuffing kernels driven directly with a bit buffer of nothing but 0xFF bytes, the most stuffing there can
    be: the retry's output size holds it, one byte less is reported as an overflow, and nothing is written or read
    outside the buffers either way."""
    k = E._k
    for n in (1, 15, 16, 17, 4095, 4096, 4097, 3 * 4096 + 100, 70_000):
        for out_cap in (E._out_cap(n, True), 2 * n + 1, 2 * n, max(1, n)):
            src = Arena(0x00, 0xFF)                    # the data: all 0xFF; around it: zeros (a read past it would count fewer)
            dst = Arena(0xA5, 0x3C)
            buf = src.empty(-(-n // 4) + 2, cp.uint32)
            buf.view(cp.uint8)[4 * (-(-n // 4)):] = 0  # the two spare words the encoder leaves behind the data
            out = dst.empty(out_cap, cp.uint8)
            total = cp.asarray(np.array([n * 8], np.int64))
            flags = cp.zeros(2, cp.int32)
            info = cp.empty(4, cp.int64)
            if route == "few-launch":
                k("stuff_small")((1,), (256,), (buf, total, np.int64(n), out, np.int64(out_cap), flags, info))
            else:
                ntile = n // E.TILE + 1
                tc = cp.empty(ntile, cp.int32)
                k("stuff_count")((ntile,), (256,), (buf, total, np.int64(n), tc))
                toff = cp.empty(ntile + 1, cp.int64)
                k("enc_offsets")((1,), (1024,), (tc, np.int32(ntile), toff))
                k("stuff_scatter")((ntile,), (256,), (buf, total, np.int64(n), toff, out, np.int64(out_cap), flags[1:]))
                k("enc_info")((1,), (32,), (toff[ntile:], total, flags, info))
            got = cp.asnumpy(info)
            fits = out_cap >= 2 * n + 1
            assert bool(got[3]) == (not fits), (route, n, out_cap, got)
            assert dst.guards_intact()[0] and src.guards_intact()[0], (route, n, out_cap)
            if fits:
                assert got[0] == 2 * n, (route, n, out_cap, got)
                assert cp.asnumpy(out[:2 * n]).tobytes() == b"\xff\x00" * n, (route, n, out_cap)

