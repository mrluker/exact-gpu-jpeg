"""Decode alone, like the original pipeline benchmark: GPU decode of camera JPEGs read from disk each time.

usage: python bench/decode_alone.py <out.json> <dir of JPEGs> [--label TEXT]

- one photo at a time: wall ms per photo (read + parse + upload + kernels, until the image is on the GPU) and the
  GPU's own split from CUDA events (upload, Huffman = unstuffing + sync passes + write pass + DC prediction,
  pixels = IDCT + upsampling + colour);
- photos/s for 48 decodes (the files 4 times) with 1, 2 and 3 threads (a Decoder each).
- a stream of 48 photos decoded one at a time: plain decode(path), and with a reader thread staging the next photo
  on a second Decoder (upload_ahead) while the current one decodes: median ms per decode() call, photos/s overall.
Results hold counts and timings only (no file names).
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402


def main():
    out, d = sys.argv[1], sys.argv[2]
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else ""
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))
    dec = egj.Decoder()
    assert dec.decode(files[0]) is not None
    rows = []
    for rep in range(3):
        for f in files:
            t = {}
            cp.cuda.Device().synchronize()
            a = time.perf_counter()
            img = dec.decode(f, timings=t)
            w = time.perf_counter() - a
            assert img is not None, dec.reason
            del img
            if rep:
                rows.append(dict(wall_ms=w * 1000, passes=dec.stats["passes"], **{k + "_ms": v * 1000 for k, v in t.items()}))
    mean = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]}
    thr = {}
    L = files * 4
    for n in (1, 2, 3):
        def g(p):
            assert egj.decode(p, fallback=False) is not None
        with ThreadPoolExecutor(n) as ex:
            list(ex.map(g, files[:n]))
            cp.cuda.Device().synchronize()
            for rep in range(2):
                a = time.perf_counter()
                list(ex.map(g, L))
                cp.cuda.Device().synchronize()
                thr[f"gpu_{n}_threads_run{rep}"] = round(len(L) / (time.perf_counter() - a), 1)
    # a stream of photos, one decode at a time: plain decode(path), against two decoders taking turns with a reader
    # thread that stages the next photo (Decoder.upload_ahead) while the current one decodes
    def stream(ahead):
        decs = [egj.Decoder(), egj.Decoder()]
        for x in decs:
            assert x.decode(files[0]) is not None
        cp.cuda.Device().synchronize()
        calls = []
        with ThreadPoolExecutor(1) as ex:
            a = time.perf_counter()
            fut = ex.submit(decs[0].upload_ahead, L[0]) if ahead else None
            for k, f in enumerate(L):
                if fut:
                    fut.result()
                fut = ex.submit(decs[(k + 1) % 2].upload_ahead, L[k + 1]) if ahead and k + 1 < len(L) else None
                b = time.perf_counter()
                img = decs[k % 2].decode(f)
                calls.append(time.perf_counter() - b)
                assert img is not None
                del img
            total = time.perf_counter() - a
        return dict(decode_call_ms=round(1000 * float(np.median(calls)), 2), photos_per_s=round(len(L) / total, 1))
    st = {}
    if hasattr(egj.Decoder, "upload_ahead"):
        for rep in range(2):
            for name, ah in (("plain", False), ("ahead", True)) if rep == 0 else (("ahead", True), ("plain", False)):
                st[f"{name}_run{rep}"] = stream(ah)
    res = dict(label=label, files=len(files), one_at_a_time_mean=mean, photos_per_s=thr, stream=st)
    print(json.dumps(res), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
