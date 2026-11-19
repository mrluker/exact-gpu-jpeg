"""What the limit on a file's synchronisation passes does: the whole call of egj.decode (the GPU attempt, then
the CPU decoder when the GPU path declines) with MAX_SYNC_PASSES at 64 (what it was), 5 (what it is) and 4, next
to the CPU decoder alone and the GPU decoder alone with 64 passes allowed. Files: the two kinds made never to
synchronise (bench/sync_worst_launch.py) at 8, 9.3 and 250 MB, and a blank page with a few lines at 64, 100 and
144 MP in 4:2:0 and 4:4:4 with default and with optimised Huffman tables (the image that decides the limit: at
100 MP and 4:2:0 it needs 5 passes and is still faster on the GPU). Median of 3 calls, ms; pixels compared. All calls
use strict=False: the never-synchronising files carry extraneous bytes, which libjpeg-turbo warns about.
usage: python bench/sync_pass_limit.py <out.json>     about 8 GB of host memory for the 144 MP images."""
import io
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image, ImageDraw
import cupy as cp
import exact_gpu_jpeg as egj
from exact_gpu_jpeg import _decoder as D
import sync_worst_launch as W

out_path = sys.argv[1]
pc = time.perf_counter
egj.self_check()
print("library:", egj.__file__, flush=True)
LIMITS = (64, 5, 4)
res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], limits=LIMITS, rows=[])
sync = cp.cuda.get_current_stream().synchronize


def flat(kind, w, h):
    if kind == "one colour":
        a = np.empty((h, w, 3), np.uint8)
        a[:] = (200, 120, 40)
        return Image.fromarray(a)
    if kind == "white":
        return Image.fromarray(np.full((h, w, 3), 255, np.uint8))
    im = Image.new("RGB", (w, h), (255, 255, 255))          # a blank page with a few lines
    d = ImageDraw.Draw(im)
    for i in range(12):
        yy = h // 10 + i * (h // 300 + 14)
        d.rectangle([w // 10, yy, w // 2, yy + 3], fill=(20, 20, 20))
    return im


def measure(label, data):
    t = pc()
    want = egj.decode_cpu(data, strict=False)       # (the never-synchronising files carry extraneous bytes:
                                                    # libjpeg-turbo warns; strict=False gives its image)
    cpu_ms = (pc() - t) * 1e3
    ts = []
    for _ in range(2):
        t = pc()
        egj.decode_cpu(data, strict=False)
        ts.append((pc() - t) * 1e3)
    cpu_ms = min(cpu_ms, min(ts))
    row = dict(label=label, file_mb=round(len(data) / 1e6, 2), cpu_ms=round(cpu_ms, 1), by_limit={})
    D.MAX_SYNC_PASSES = 64
    gdec = egj.Decoder()
    ts = []
    for _ in range(4):
        t = pc()
        img = gdec.decode(data)
        gdec.stream.synchronize()
        ts.append((pc() - t) * 1e3)
        ok, passes = img is not None, (gdec.stats.get("passes") if img is not None else None)
        del img
    row.update(gpu_alone_ms=round(statistics.median(ts[1:]), 1), gpu_alone_decoded=ok, passes=passes)
    del gdec
    for lim in LIMITS:
        D.MAX_SYNC_PASSES = lim
        ts, path = [], None
        for _ in range(3):
            info = {}
            t = pc()
            img = egj.decode(data, cpu_below=0, info=info, strict=False)
            if hasattr(img, "get"):
                sync()
            ts.append((pc() - t) * 1e3)
            path = info.get("path")
        same = bool(np.array_equal(cp.asnumpy(img), want))
        row["by_limit"][lim] = dict(ms=round(statistics.median(ts), 1), path=path, same_pixels=same)
        del img
    D.MAX_SYNC_PASSES = 64
    res["rows"].append(row)
    print(f"| {label} | {row['file_mb']} MB | {row['cpu_ms']} | {row['gpu_alone_ms']} ({'passes ' + str(row['passes']) if row['gpu_alone_decoded'] else 'declined'}) | "
          + " | ".join(f"{row['by_limit'][l]['ms']} ({row['by_limit'][l]['path']}{'' if row['by_limit'][l]['same_pixels'] else ', OTHER PIXELS'})" for l in LIMITS) + " |", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


print("| file | size | CPU decoder alone, ms | GPU decoder alone with 64 passes, ms | " + " | ".join(f"limit {l}: ms (path)" for l in LIMITS) + " |")
print("|---|---|---|---|" + "---|" * len(LIMITS))
for kind, make in (("crafted, 8-bit symbols", W.crafted), ("crafted, 3-bit blocks", W.dense)):
    for mb in (8, 9.3, 250):
        data, ne = make(int(mb * 1e6))
        measure(f"{kind}, {mb:g} MB", data)
        del data
for w, h in ((8000, 8000), (10000, 10000), (12000, 12000)):
    for kind, sub, opt in (("blank page with a few lines", 2, False), ("blank page with a few lines", 0, False),
                           ("blank page with a few lines", 2, True), ("blank page with a few lines", 0, True)):
        b = io.BytesIO()
        flat(kind, w, h).save(b, "JPEG", quality=90, subsampling=sub, optimize=opt)
        measure(f"{kind}, {w * h / 1e6:.0f} MP, {'4:4:4' if sub == 0 else '4:2:0'}, {'optimised' if opt else 'default'} tables", b.getvalue())
    cp.get_default_memory_pool().free_all_blocks()
