"""How long can one kernel launch run on a JPEG made to stretch it? (Windows resets the display driver when the
GPU does not answer for about 2 s, which ends every process's CUDA context on the card.)

The synchronisation pass is the one decode kernel whose work per launch depends on the data: inside a CUDA block
of 128 chunks, a chunk decodes again each time the chunk before it changes its exit state, up to 129 times. Real
JPEGs synchronise after a few symbols. This file is made so that they never do: both Huffman tables hold 127
codes of 7 bits; every DC code means "no value bits" and every AC code "one coefficient, one value bit", so a
block is 7 + 63 x 8 = 511 bits whatever the data says; the data has a 0 before every code and 0 as every value
bit (no 0xFF byte, no invalid code at any bit phase). A chunk guesses that a block starts at its first bit, which
is true for one chunk in 511, and a decode from a wrong phase stays 511-periodic at that phase for ever. The image
is 64x64 grey; the data after its blocks is still part of the stream the passes cover. libjpeg-turbo decodes it.

A second file of the same kind has the most symbols a byte can hold: both tables have one code of one bit, a DC
code with one value bit and the end-of-block code, so a block is 3 bits and the data is all zero bits (16 symbols
in 3 bytes, where the first file has one a byte). Its chunks' guesses repeat with period 3 and never settle either.

The library limits the work of a synchronisation launch: the bytes of entropy-coded data it can decode
(sync_budget and sync_decodes in _decoder.py; the limit is the same on every card). This script measures what
that work takes on the card it runs on. Each synchronisation launch is waited for before it starts and after it,
and timed on its own. The run stops if a launch passes STOP_MS.
usage: python bench/sync_worst_launch.py <out.json> [MB ...]     Timings only; the files are made in memory."""
import io
import json
import os
import struct
import sys
import time

os.environ.setdefault("EGJ_SMALL_CPU_PIXELS", "200000")    # pinned: no background measurement during the run
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402

STOP_MS = 250.0
SIZES = [1, 4, 8, 12, 16, 24, 64, 128, 250]
pc = time.perf_counter


def crafted(nbytes, seed=0):
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
    data = (tile * (n // len(tile) + 1))[:n]
    return head + data + bytes([0xFF, 0xD9]), len(data)


def dense(nbytes):
    """The same idea with 3-bit blocks: one 1-bit code per table, all-zero data. (the file, its entropy-coded bytes)"""
    def seg(marker, payload):
        return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload
    one = bytes([1] + [0] * 15)
    head = (bytes([0xFF, 0xD8]) + seg(0xDB, bytes([0]) + bytes([1] * 64))
            + seg(0xC0, bytes([8]) + struct.pack(">HH", 64, 64) + bytes([1, 1, 0x11, 0]))
            + seg(0xC4, bytes([0x00]) + one + bytes([0x01])) + seg(0xC4, bytes([0x10]) + one + bytes([0x00]))
            + seg(0xDA, bytes([1, 1, 0x00, 0, 63, 0])))
    n = max(nbytes, 24)
    return head + bytes(n) + bytes([0xFF, 0xD9]), n


def timed_decode(dec, data):
    """Decode with every sync_pass launch waited for and timed. Returns (image or None, launch times in ms, the
    `rounds` argument of the launches, the whole call in ms)."""
    times, rounds = [], set()
    real = D._k

    def k(name):
        f = real(name)
        if name != "sync_pass":
            return f

        def launch(grid, block, args, **kw):
            rounds.add(int(args[-1]))
            dec.stream.synchronize()
            t = pc()
            f(grid, block, args, **kw)
            dec.stream.synchronize()
            times.append((pc() - t) * 1e3)
        return launch
    D._k = k
    try:
        t = pc()
        img = dec.decode(data)
        dec.stream.synchronize()
        total = (pc() - t) * 1e3
    finally:
        D._k = real
    return img, times, sorted(rounds), total


def main():
    out = sys.argv[1]
    sizes = [float(x) for x in sys.argv[2:]] or SIZES
    dec = egj.Decoder()
    egj.self_check()
    print("library:", egj.__file__, flush=True)
    res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], stop_ms=STOP_MS,
               timing="every synchronisation launch waited for before and after", rows=[])
    from reproduce import synthetic_photo                      # an ordinary image, for comparison
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(synthetic_photo(6000, 4000, 1)).save(b, "JPEG", quality=90)
    normal = b.getvalue()
    for _ in range(2):
        img, times, rounds, total = timed_decode(dec, normal)
    res["normal"] = dict(bytes=len(normal), ok=img is not None, launches=len(times), longest_ms=round(max(times), 3),
                         decode_ms=round(total, 2))
    print(f"ordinary 24 MP image, {len(normal) / 1e6:.1f} MB: {len(times)} launches, longest {max(times):.3f} ms", flush=True)
    print("| file | data | rounds a launch | bytes a launch can decode | result | launches | longest launch | median launch | whole call |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kind, make in (("8-bit symbols", crafted), ("3-bit blocks", dense)):
        for mb in sizes:
            data, ne = make(int(mb * 1e6))
            r, p = D.sync_budget(ne)
            work = ne * D.sync_decodes(r) // 256
            img, times, rounds, total = timed_decode(dec, data)
            row = dict(kind=kind, mb=mb, bytes=len(data), entropy_bytes=ne, rounds=r, passes=p, rounds_seen=rounds,
                       launch_work_bytes=work, launch_work_limit=D.SYNC_LAUNCH_WORK, declined=img is None,
                       reason=dec.reason, launches=len(times), longest_ms=round(max(times), 3),
                       first_ms=round(times[0], 3), median_ms=round(float(np.median(times)), 3),
                       call_ms=round(total, 1))
            res["rows"].append(row)
            print(f"| {kind} | {mb:g} MB | {r} | {work / 1e6:.0f} MB | {'declined' if img is None else 'decoded'} | {len(times)} | "
                  f"{max(times):.1f} ms | {np.median(times):.1f} ms | {total / 1e3:.2f} s |", flush=True)
            json.dump(res, open(out, "w"), indent=1)
            del data, img
            if max(times) > STOP_MS:
                print(f"stopped: a launch over {STOP_MS:g} ms", flush=True)
                res["stopped_after"] = [kind, mb]
                break
    res["normal_after_ok"] = dec.decode(normal) is not None    # the decoder is fine afterwards
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
