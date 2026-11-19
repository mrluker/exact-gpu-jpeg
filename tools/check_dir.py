"""Exactness of exact_gpu_jpeg against simplejpeg (libjpeg-turbo, accurate IDCT, fancy upsampling) on JPEG files.

usage: python tools/check_dir.py [--list FILE] [--colorspace RGB|BGR|GRAY] [--json OUT] [--batch N] [paths or dirs ...]

--batch N: decode the files N at a time with decode_batch (counts["batched"] = how many shared a set of launches).

Prints and writes aggregate counts only (no file names), so it can run on private photos. Set JSIMD_FORCENONE=1,
JSIMD_FORCESSE2=1 or JSIMD_FORCEAVX2=1 in the environment to choose libjpeg-turbo's code path for the reference.
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import simplejpeg

import exact_gpu_jpeg as egj


def files(args):
    out = []
    for p in args.paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.jpeg")))
        else:
            out.append(p)
    if args.list:
        out += [l.strip() for l in open(args.list) if l.strip()]
    seen, u = set(), []
    for f in out:                      # Windows globs are case-insensitive: drop duplicates
        k = os.path.normcase(os.path.abspath(f))
        if k not in seen:
            seen.add(k)
            u.append(f)
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--colorspace", default="RGB")
    ap.add_argument("--json")
    ap.add_argument("--batch", type=int, default=0)
    a = ap.parse_args()
    dec = egj.Decoder()
    c = collections.Counter()
    kinds, passes, reasons = collections.Counter(), collections.Counter(), collections.Counter()
    t0 = time.perf_counter()
    fl = files(a)
    step = max(a.batch, 1)
    for i0 in range(0, len(fl), step):
        datas = [open(f, "rb").read() for f in fl[i0:i0 + step]]
        if a.batch:
            outs = dec.decode_batch(datas, a.colorspace)
            why = list(dec.reasons)
            c["batched"] += dec.stats["batched"]
        for k, data in enumerate(datas):
            try:
                ref = simplejpeg.decode_jpeg(data, colorspace=a.colorspace, fastdct=False, fastupsample=False, strict=False)
            except Exception:
                ref = None
            out = outs[k] if a.batch else dec.decode(data, a.colorspace)
            c["files"] += 1
            if out is None:
                c["cpu_fallback"] += 1
                reasons[why[k] if a.batch else dec.reason] += 1
                continue
            if not a.batch:
                kinds[dec.stats.get("subsampling")] += 1
                passes[dec.stats.get("passes")] += 1
            if ref is None:
                c["gpu_but_reference_failed"] += 1
                continue
            g = out.get()
            if g.shape == ref.shape and np.array_equal(g, ref):
                c["exact"] += 1
            else:
                c["different"] += 1
    res = dict(counts=dict(c), subsampling=dict(kinds), sync_passes={str(k): v for k, v in sorted(passes.items())},
               fallback_reasons=dict(reasons), seconds=round(time.perf_counter() - t0, 1),
               simd={k: v for k, v in os.environ.items() if k.startswith("JSIMD_")}, colorspace=a.colorspace)
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0 if not c["different"] and not c["gpu_but_reference_failed"] and c["exact"] else 1   # (none decoded: not a pass)


if __name__ == "__main__":
    sys.exit(main())
