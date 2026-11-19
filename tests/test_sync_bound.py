"""No file decides how long a kernel launch runs.

The synchronisation pass is the one decode kernel whose work per launch depends on the data: inside a CUDA block a
chunk decodes again each time the chunk before it changes its exit state. A real JPEG settles after a few symbols.
The file made here never does: both Huffman tables hold 127 codes of 7 bits; every DC code means "no value bits"
and every AC code "one coefficient, one value bit", so a block is 7 + 63 x 8 = 511 bits whatever the data says; the
data has a 0 before every code and 0 as every value bit (no 0xFF byte, no invalid code at any bit phase). A chunk
guesses that a block starts at its first bit, which is true for one chunk in 511, and a decode from a wrong phase
stays at that phase for ever. It is a valid JPEG: libjpeg-turbo decodes it (a 64x64 grey image; the data after its
blocks is ignored).

A second file has the most symbols a byte can hold: one code of one bit in each table, a block of 3 bits, all-zero
data. Its chunks' guesses repeat with period 3 and do not settle either.

With nothing limiting it, one launch on such a file took 3.5 ms per MB of data on a GTX 1070, and a launch that
runs for seconds makes Windows reset the display driver, which ends every process's CUDA context on the card. The
limit is on work, so that it is the same on every card: the bytes of entropy-coded data one launch can decode
(SYNC_LAUNCH_WORK; sync_budget gives a file the most rounds per launch that keep within it, one round at the
input limit), and a file is declined after the eager group of SYNC_GROUP passes and a redo of MAX_SYNC_PASSES:
at most SYNC_GROUP + MAX_SYNC_PASSES launches for any file. decode() returns libjpeg-turbo's pixels for a declined
file from the CPU decoder.

The tests count the work with a build of the kernels that counts it (EGJ_COUNT_SYNC: chunk decodes and bits per
launch); times are asserted only loosely, they are one card's."""
import io
import struct
import threading
import time

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from conftest import reference  # noqa: E402


def never_synchronising(nbytes, seed=0):
    """A valid JPEG (64x64, grey) with about nbytes of entropy-coded data whose chunks never synchronise."""
    def seg(marker, payload):
        return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload
    counts = bytes([0, 0, 0, 0, 0, 0, 127] + [0] * 9)
    dht = lambda tc, sym: seg(0xC4, bytes([tc << 4]) + counts + bytes([sym] * 127))      # noqa: E731
    head = (bytes([0xFF, 0xD8]) + seg(0xDB, bytes([0]) + bytes([1] * 64))
            + seg(0xC0, bytes([8]) + struct.pack(">HH", 64, 64) + bytes([1, 1, 0x11, 0]))
            + dht(0, 0x00) + dht(1, 0x01) + seg(0xDA, bytes([1, 1, 0x00, 0, 63, 0])))
    rng = np.random.default_rng(seed)
    nb = 8 * 256                                               # 8 blocks = 4088 bits = 511 bytes
    bits = np.zeros((nb, 511), np.uint8)
    bits[:, 1:7] = rng.integers(0, 2, (nb, 6))                 # DC: a 7-bit code 0xxxxxx, no value bits
    for j in range(63):                                        # AC: 0xxxxxx and the value bit 0
        bits[:, 8 + 8 * j:14 + 8 * j] = rng.integers(0, 2, (nb, 6))
    tile = np.packbits(bits.reshape(-1)).tobytes()
    assert len(tile) == nb * 511 // 8 and 0xFF not in tile
    n = max(nbytes // 511, 1) * 511
    return head + (tile * (n // len(tile) + 1))[:n] + bytes([0xFF, 0xD9])


def densest(nbytes):
    """A valid JPEG (64x64, grey) with nbytes of all-zero data: both tables have one 1-bit code, a block is 3 bits."""
    def seg(marker, payload):
        return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload
    one = bytes([1] + [0] * 15)
    head = (bytes([0xFF, 0xD8]) + seg(0xDB, bytes([0]) + bytes([1] * 64))
            + seg(0xC0, bytes([8]) + struct.pack(">HH", 64, 64) + bytes([1, 1, 0x11, 0]))
            + seg(0xC4, bytes([0x00]) + one + bytes([0x01])) + seg(0xC4, bytes([0x10]) + one + bytes([0x00]))
            + seg(0xDA, bytes([1, 1, 0x00, 0, 63, 0])))
    return head + bytes(max(nbytes, 24)) + bytes([0xFF, 0xD9])


def _entropy_bytes(data):
    return len(data) - (data.index(bytes([0xFF, 0xDA])) + 10) - 2      # after the SOS segment, before EOI


_counting = {}


def _counted(dec, data, monkeypatch):
    """dec.decode(data) with the synchronisation kernel of a build that counts its work, every launch waited for
    before and after. Returns (image, launches, seconds); a launch is a dict: ms, rounds, blocks (CUDA blocks of
    128 chunks), first (the pass that decodes every chunk before its rounds), decodes (chunk decodes), bits."""
    dev = dec.device.id
    if dev not in _counting:
        with dec.device:
            mod = cp.RawModule(code=D._kernel_source(), options=("--std=c++14", "-DEGJ_COUNT_SYNC"))
            _counting[dev] = (mod.get_function("sync_pass"), mod.get_function("sync_count_read"))
    counted, read = _counting[dev]
    launches = []
    real = D._k
    me = threading.get_ident()

    def k(name):
        if name != "sync_pass":
            return real(name)

        def launch(grid, block, args, **kw):
            if threading.get_ident() != me:                    # another thread's decode (the background timing of
                return real(name)(grid, block, args, **kw)     # the CPU route): not this decode's work
            out = cp.zeros(2, cp.uint64)
            dec.stream.synchronize()
            t = time.perf_counter()
            counted(grid, block, args, **kw)
            dec.stream.synchronize()
            ms = (time.perf_counter() - t) * 1e3
            read((1,), (1,), (out,))
            n, bits = (int(v) for v in out.get())
            launches.append(dict(ms=ms, rounds=int(args[-1]), blocks=int(grid[0]),
                                 first=int(args[15]) == 0, decodes=n, bits=bits))
        return launch
    monkeypatch.setattr(D, "_k", k)
    t = time.perf_counter()
    img = dec.decode(data)
    dec.stream.synchronize()
    total = time.perf_counter() - t
    monkeypatch.undo()
    return img, launches, total


def test_the_rounds_of_a_launch_by_size():
    assert (D.SYNC_ROUNDS, D.SYNC_GROUP, D.MAX_SYNC_PASSES) == (129, 3, 5)
    assert [D.sync_decodes(r) for r in (0, 1, 2, 126, 127, 128, 129)] == [256, 510, 762, 16510, 16512, 16512, 16512]
    for ne in (1, 100_000, 3_000_000, 9_300_000):              # photos up to about 24 MP: every round there is
        assert D.sync_budget(ne) == (129, D.MAX_SYNC_PASSES)
    last = 129
    for ne in list(range(9_400_000, D.MAX_INPUT, 1_234_567)) + [D.MAX_INPUT]:
        r, p = D.sync_budget(ne)
        assert 1 <= r <= last and p == D.MAX_SYNC_PASSES       # never more rounds for more data
        assert ne * D.sync_decodes(r) <= D.SYNC_LAUNCH_WORK * 256                        # one launch's work: within
        assert r == 129 or ne * D.sync_decodes(r + 1) > D.SYNC_LAUNCH_WORK * 256         # and one round more is not
        last = r
    assert D.sync_budget(D.MAX_INPUT) == (1, D.MAX_SYNC_PASSES)        # the largest input there is: one round


def test_the_work_a_launch_does_is_what_the_rule_says(monkeypatch):
    """An ordinary photo and two small never-synchronising files, counted: no launch decodes a chunk more often
    than sync_decodes() allows; the photo settles in the eager group, its first launch doing a small part of what
    the rule allows; the other two get the eager group and the redo and no launch more."""
    dec = egj.Decoder()
    rng = np.random.default_rng(5)
    a = (rng.random((150, 200, 3)) * 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((1600, 1200), Image.BICUBIC).save(b, "JPEG", quality=92)
    for data, settles in ((b.getvalue(), True), (never_synchronising(300_000), False), (densest(300_000), False)):
        img, launches, total = _counted(dec, data, monkeypatch)
        if settles:
            assert img is not None and np.array_equal(img.get(), reference(data)), dec.reason
            assert len(launches) == D.SYNC_GROUP and dec.stats["passes"] == 2
        else:
            assert img is None and "synchronise" in dec.reason
            assert len(launches) == D.SYNC_GROUP + D.MAX_SYNC_PASSES
        assert launches[0]["first"] and {x["rounds"] for x in launches} == {D.SYNC_ROUNDS}
        for x in launches:
            most = x["blocks"] * D.sync_decodes(x["rounds"]) // 2          # chunk decodes: 128 x the mean per chunk
            assert x["decodes"] <= most, x
            assert x["bits"] <= 8 * _entropy_bytes(data) * D.sync_decodes(x["rounds"]) // 256 + 64 * x["decodes"], x
        if settles:
            assert launches[0]["decodes"] * 2 <= launches[0]["blocks"] * D.sync_decodes(4), launches[0]


@pytest.mark.parametrize("kind", ["8-bit symbols", "3-bit blocks"])
def test_a_file_that_never_synchronises_is_declined_within_the_work_limit(kind, monkeypatch):
    data = never_synchronising(24_000_000) if kind == "8-bit symbols" else densest(24_000_000)
    ne = _entropy_bytes(data)
    want = reference(data)
    assert want is not None and want.shape == (64, 64, 3)      # a valid file: libjpeg-turbo decodes it
    dec = egj.Decoder()
    good = never_synchronising(8 * 511)                        # the same header with just its 64 blocks of data
    img, launches, total = _counted(dec, data, monkeypatch)
    assert img is None and "synchronise" in dec.reason, dec.reason
    r, p = D.sync_budget(ne)
    assert 1 <= r < D.SYNC_ROUNDS and p == D.MAX_SYNC_PASSES
    assert {x["rounds"] for x in launches} == {r} and len(launches) == D.SYNC_GROUP + p == 8
    assert sum(x["bits"] for x in launches) <= 8 * (D.SYNC_GROUP + p) * D.SYNC_LAUNCH_WORK     # the file's whole work
    most = D.sync_decodes(r)                                   # 256 x the chunk decodes a chunk, at most
    for x in launches:
        assert x["decodes"] * 2 <= x["blocks"] * most, x       # exactly the rule's count, never more
        assert x["bits"] <= 8 * D.SYNC_LAUNCH_WORK, x          # and the work limit itself: bytes decoded
    firsts = [x for x in launches if x["first"]]
    assert len(firsts) == 2                                    # the eager group, and the redo with host checks
    for x in firsts:                                           # this file reaches the rule's count (so the count is
        assert x["decodes"] * 2 >= (x["blocks"] - 1) * most, x  # exact, not a loose estimate): every full block does
    # Time is one card's measurement (bench/sync_worst_launch.py has it); here only a loose guard, far above what
    # the work takes on the test card and far below what resets a display driver.
    assert max(x["ms"] for x in launches) < 1000, max(x["ms"] for x in launches)
    assert total < 60, total
    info = {}
    via = egj.decode(data, cpu_below=0, info=info, strict=False)   # the module-level call: the CPU decoder's
    # pixels (libjpeg-turbo warns about this crafted file: extraneous bytes before its end marker)
    assert info["path"] == "cpu" and "synchronise" in info["reason"]
    assert np.array_equal(via.get(), want)
    for _ in range(2):                                         # the decoder is fine afterwards
        img = dec.decode(good)
        assert img is not None and np.array_equal(img.get(), reference(good)), dec.reason


def test_the_cap_changes_no_pixel(corpus, monkeypatch):
    """Fewer rounds per launch means more launches to the same fixed point: with a cap of 1, of 2 and of 129 (what
    it was), every file is decoded to the same pixels or declined for not synchronising in time."""
    files = [c for c in corpus[::6] if c[2]]
    out = {}
    for r in (129, 2, 1):
        monkeypatch.setattr(D, "sync_budget", lambda ne, r=r: (r, 400))
        dec = egj.Decoder()
        for name, data, expected in files:
            img = dec.decode(data)
            assert img is not None or "synchronise" in dec.reason, (r, name, dec.reason)
            if img is not None:
                assert np.array_equal(img.get(), reference(data)), (r, name)
            out[(r, name)] = img is not None
    assert all(out[(129, name)] for name, _, _ in files)       # with the old value every one decodes
    assert sum(out[(2, name)] for name, _, _ in files) >= len(files) - 3
