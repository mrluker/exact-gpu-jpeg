"""Encode speed: exact-gpu-jpeg vs Pillow (the bytes it matches) and simplejpeg, at three typical web export settings.

usage: python bench/encode_bench.py <out.json> <dir of source photos>

Inputs: the photos resized (Lanczos) to 3840 px q95 4:4:4, 1500 px q90 4:4:4 and 1200 px q87 4:2:0, with their
EXIF, as RGB arrays already in memory: host memory for the CPU encoders; for ours both ways, already in GPU memory
(a decode, edit, encode pipeline) and in host memory (a NumPy array, uploaded by the call: the like-for-like
comparison with the CPU encoders). Each encoder ends with the JPEG bytes in host memory. Per size: median ms one at a time; the three exports of one photo back to back; throughput
(images/s) with 1 and 2 GPU threads (an Encoder each) and 1 / 4 / 8 CPU threads. Also the bytes that cross PCIe for
a GPU-held image: raw RGB for a CPU encoder vs the compressed scan for ours. No file names in the results.
"""
import glob
import io
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simplejpeg
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402

SETTINGS = [(3840, 95, "444"), (1500, 90, "444"), (1200, 87, "420")]
MIN_S = 2.0


def pil(a, q, ss, exif):
    b = io.BytesIO()
    kw = {"exif": exif} if exif else {}
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling={"444": 0, "420": 2}[ss], **kw)
    return b.getvalue()


def sj(a, q, ss, exif):
    return simplejpeg.encode_jpeg(a, quality=q, colorsubsampling=ss, fastdct=False)


def rep(fn, items):
    ts, t0, k = [], time.perf_counter(), 0
    while k < 2 or time.perf_counter() - t0 < MIN_S:
        for x in items:
            a = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - a) * 1000)
        k += 1
    return statistics.median(ts)


def thr(fn, items, n):
    items = items * -(-16 // len(items))
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(fn, items[:n]))
        cnt, t0 = 0, time.perf_counter()
        while cnt == 0 or time.perf_counter() - t0 < MIN_S:
            list(ex.map(fn, items))
            cnt += len(items)
        return cnt / (time.perf_counter() - t0)


def main():
    out, d = sys.argv[1], sys.argv[2]
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())[:6]
    srcs = []
    for f in files:
        with Image.open(f) as im:
            srcs.append((im.convert("RGB"), im.info.get("exif")))
    res = dict(photos=len(srcs), gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), sizes={})
    enc = egj.Encoder()
    for w, q, ss in SETTINGS:
        host = [(np.asarray(im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)), ex) for im, ex in srcs]
        dev = [(cp.asarray(a), ex) for a, ex in host]
        cp.cuda.Device().synchronize()
        assert all(enc.encode(g, q, ss, None, ex) == pil(a, q, ss, ex) for (g, ex), (a, _) in zip(dev, host))
        r = dict(width=w, height=host[0][0].shape[0], quality=q, subsampling=ss)
        r["pillow_ms"] = rep(lambda x: pil(x[0], q, ss, x[1]), host)
        r["simplejpeg_ms"] = rep(lambda x: sj(x[0], q, ss, x[1]), host)
        r["gpu_ms"] = rep(lambda x: enc.encode(x[0], q, ss, None, x[1]), dev)
        r["gpu_from_host_ms"] = rep(lambda x: enc.encode(x[0], q, ss, None, x[1]), host)     # (the upload included)
        tm = {}
        for g, ex in dev:
            enc.encode(g, q, ss, None, ex, timings=tm)
        r["gpu_split_ms"] = {k: round(v * 1000 / len(dev), 2) for k, v in tm.items()}
        r["raw_rgb_mb"] = round(host[0][0].nbytes / 1e6, 2)
        r["jpeg_mb"] = round(statistics.mean(len(pil(a, q, ss, ex)) for a, ex in host) / 1e6, 2)
        for n in (1, 4, 8):
            r[f"pillow_{n}t_per_s"] = thr(lambda x: pil(x[0], q, ss, x[1]), host, n)
            r[f"simplejpeg_{n}t_per_s"] = thr(lambda x: sj(x[0], q, ss, x[1]), host, n)
        for n in (1, 2):
            r[f"gpu_{n}t_per_s"] = thr(lambda x: egj.encode(x[0], q, ss, None, x[1]), dev, n)
            r[f"gpu_from_host_{n}t_per_s"] = thr(lambda x: egj.encode(x[0], q, ss, None, x[1]), host, n)
        res["sizes"][f"{w}_q{q}_{ss}"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}
        print(json.dumps(res["sizes"][f"{w}_q{q}_{ss}"]), flush=True)
        res.setdefault("_dev", []).append(dev)
        res.setdefault("_host", []).append(host)
    # the three exports of one photo, back to back
    devs, hosts = res.pop("_dev"), res.pop("_host")
    k = len(srcs)
    three_gpu = rep(lambda i: [enc.encode(devs[s][i][0], q, ss, None, devs[s][i][1]) for s, (w, q, ss) in enumerate(SETTINGS)], list(range(k)))
    three_pil = rep(lambda i: [pil(hosts[s][i][0], q, ss, hosts[s][i][1]) for s, (w, q, ss) in enumerate(SETTINGS)], list(range(k)))
    res["three_exports_ms"] = dict(gpu=round(three_gpu, 2), pillow=round(three_pil, 2))
    print(json.dumps(res["three_exports_ms"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
