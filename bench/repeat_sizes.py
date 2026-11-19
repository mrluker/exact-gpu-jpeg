"""From what image size does a copy save time? For generated photo-like JPEGs of 0.1 to 24 MP: one decode from a
file and from bytes, sustained, without the cache, from a copy with the full check, and from a copy with the light
check (paths only). Modes take turns inside each of 7 rounds; medians of the rounds' medians, with the spread.
usage: python bench/repeat_sizes.py <out.json> <work dir>     timings only; generated images, written to <work dir>."""
import io
import json
import os
import statistics
import sys
import time

os.environ.setdefault("EGJ_SMALL_CPU_PIXELS", "200000")    # pinned: no background measurement during the run
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from reproduce import synthetic_photo  # noqa: E402

out, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
pc, med = time.perf_counter, statistics.median
sync = cp.cuda.get_current_stream().synchronize
egj.self_check()
print("library:", egj.__file__, flush=True)
SIZES = [(400, 250), (640, 400), (800, 600), (1200, 800), (1600, 1200), (2400, 1600), (3600, 2400), (4800, 3200), (6000, 4000)]
res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], rows=[])
print("| image | file | from a file: no cache | copy, full check | copy, light check | from bytes: no cache | copy, full check |\n|---|---|---|---|---|---|---|")
for k, (w, h) in enumerate(SIZES):
    b = io.BytesIO()
    Image.fromarray(synthetic_photo(w, h, k + 1)).save(b, "JPEG", quality=90)
    data = b.getvalue()
    p = os.path.join(work, f"s{k}.jpg")
    open(p, "wb").write(data)
    cache = egj.RepeatCache(memory_bytes=1 << 30, min_pixels=0)
    modes = {"file, no cache": lambda: egj.decode(p, cpu_below=0),
             "file, copy, full": lambda: egj.decode(p, cpu_below=0, cache=cache),
             "file, copy, light": lambda: egj.decode(p, cpu_below=0, cache=cache, check="light"),
             "bytes, no cache": lambda: egj.decode(data, cpu_below=0),
             "bytes, copy, full": lambda: egj.decode(data, cpu_below=0, cache=cache)}
    n = max(8, min(200, int(3e6 / (w * h)) * 10))
    for f in modes.values():                                   # the copies exist, everything is warm
        for _ in range(4):
            f()
    sync()
    assert cache.stats["copies"] == 1 and cache.stats["used"] >= 9
    rounds = {m: [] for m in modes}
    for r in range(7):
        order = list(modes)[r % len(modes):] + list(modes)[:r % len(modes)]
        for m in order:
            f = modes[m]
            ts = []
            for _ in range(n):
                t = pc()
                img = f()
                sync()
                ts.append((pc() - t) * 1e3)
                del img
            rounds[m].append(med(ts))
    row = dict(width=w, height=h, mp=round(w * h / 1e6, 2), file_bytes=len(data), calls_per_round=n,
               ms={m: round(med(v), 4) for m, v in rounds.items()},
               spread_ms={m: [round(min(v), 4), round(max(v), 4)] for m, v in rounds.items()},
               copy_bytes=cache.stats["bytes"])
    res["rows"].append(row)
    ms = row["ms"]
    print(f"| {w}x{h} ({row['mp']} MP) | {len(data) / 1e6:.2f} MB | {ms['file, no cache']:.3f} | {ms['file, copy, full']:.3f} | "
          f"{ms['file, copy, light']:.3f} | {ms['bytes, no cache']:.3f} | {ms['bytes, copy, full']:.3f} |", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    cache.clear()
