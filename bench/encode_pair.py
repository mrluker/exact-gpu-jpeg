"""Two exports of one photo (a typical 3840 px master, quality 95 4:4:4, and 2048 px copy, quality 90
4:4:4): two encode() calls in turn, against both scans queued with encode_async on two Encoders and then collected.

usage: python bench/encode_pair.py <out.json> <dir of photos> [--files DIR]

The photos are resized on the host (Lanczos) and held on the GPU before the timing starts. Each way is timed per
photo to JPEG bytes in host memory and, with --files, to files written in DIR (path=); rounds alternate the order.
Median ms for the pair. Counts and timings only (no file names).
"""
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402

SETTINGS = ((3840, 95, "444"), (2048, 90, "444"))


def main():
    out, d = sys.argv[1], sys.argv[2]
    fdir = sys.argv[sys.argv.index("--files") + 1] if "--files" in sys.argv else None
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())
    pairs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        pairs.append([cp.asarray(np.asarray(im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)))
                      for w, _, _ in SETTINGS])
    cp.cuda.Device().synchronize()
    e1, e2 = egj.Encoder(), egj.Encoder()
    ev = cp.cuda.Event()
    ev.record(cp.cuda.get_current_stream())
    (_, q1, s1), (_, q2, s2) = SETTINGS

    def turn(a, b, p1=None, p2=None):
        return e1.encode(a, q1, s1, path=p1), e1.encode(b, q2, s2, path=p2)

    def queued(a, b, p1=None, p2=None):
        j1 = e1.encode_async(a, q1, s1, path=p1, ready=ev)
        j2 = e2.encode_async(b, q2, s2, path=p2, ready=ev)
        return j1.result(), j2.result()
    same = all(turn(a, b) == queued(a, b) for a, b in pairs)
    modes = [("bytes", None)] + ([("files", fdir)] if fdir else [])
    res = dict(photos=len(pairs), settings=[list(x) for x in SETTINGS], identical=same, rounds={})
    for mode, dd in modes:
        p1 = os.path.join(dd, "m.jpg") if dd else None
        p2 = os.path.join(dd, "e.jpg") if dd else None
        rows = {"turn": [], "queued": []}
        for r in range(6):
            order = (("turn", turn), ("queued", queued)) if r % 2 == 0 else (("queued", queued), ("turn", turn))
            for name, fn in order:
                ts = []
                for a, b in pairs:
                    cp.cuda.Device().synchronize()
                    t = time.perf_counter()
                    fn(a, b, p1, p2)
                    ts.append((time.perf_counter() - t) * 1e3)
                rows[name].append(round(statistics.median(ts), 3))
        res["rounds"][mode] = rows
        print(mode, rows, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
