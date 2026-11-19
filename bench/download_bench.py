"""Copying a decoded image back to host memory: img.get() (into new, pageable memory) against img.get(out=...) into a
page-locked buffer allocated once (cupyx.empty_pinned), for RGB images of 1 to 30 MP; ms per copy (median of 15
after 3) and GB/s. The library itself returns GPU arrays only; this is the copy a program that wants NumPy makes.

usage: python bench/download_bench.py <out.json>"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402
import cupyx  # noqa: E402
import numpy as np  # noqa: E402


def med(f, n=15, warm=3):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 3)


def main():
    out = dict(gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), sizes={})
    for w, h in ((1224, 816), (3000, 2000), (6000, 4000), (6336, 4752)):
        img = cp.asarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8))
        pin = cupyx.empty_pinned(img.shape, img.dtype)
        r = dict(megapixels=round(w * h / 1e6, 1), mb=round(img.nbytes / 1e6, 1))
        r["get_ms"] = med(lambda: img.get())
        r["get_into_pinned_ms"] = med(lambda: img.get(out=pin))
        assert np.array_equal(pin, img.get())
        r["get_gb_per_s"] = round(img.nbytes / 1e9 / (r["get_ms"] / 1e3), 2)
        r["pinned_gb_per_s"] = round(img.nbytes / 1e9 / (r["get_into_pinned_ms"] / 1e3), 2)
        out["sizes"][f"{w}x{h}"] = r
        print(f"{w}x{h}", r, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
