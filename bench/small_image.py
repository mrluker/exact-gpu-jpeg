"""egj.decode(one image): the default (small images on the CPU) against cpu_below=0 (always the GPU), paired.
Each call is timed to the point where the pixels are in GPU memory and every stream is idle.
Rounds alternate the order; 36 JPEGs per size made from the photos (quality 90), then the photos themselves as
bytes and as paths. Counts and timings only (no file names).

usage: python bench/small_image.py <out.json> <dir of photos>"""
import glob
import io
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

out, d = sys.argv[1], sys.argv[2]
files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())
photos = [Image.open(f).convert("RGB") for f in files]
null = cp.cuda.Stream.null
egj.decode(open(files[0], "rb").read())
dstream = egj._thread_decoder(None).stream


def idle():
    dstream.synchronize()
    null.synchronize()


def default(x):
    g = egj.decode(x)
    idle()
    return g


def gpu_only(x):
    g = egj.decode(x, cpu_below=0)
    idle()
    return g


def pair(data, reps, rounds=8):
    rows = {"default": [], "cpu_below=0": []}
    fns = (("default", default), ("cpu_below=0", gpu_only))
    for r in range(rounds):
        for name, fn in (fns if r % 2 == 0 else fns[::-1]):
            ts = []
            for rep in range(reps):
                for x in data:
                    idle()
                    t = time.perf_counter()
                    fn(x)
                    ts.append((time.perf_counter() - t) * 1e3)
            rows[name].append(round(statistics.median(ts), 4))
    return rows


def show(key, rows, extra=""):
    a, b = rows["default"], rows["cpu_below=0"]
    print(f"{key:22s} default {statistics.median(a):7.3f} [{min(a):.3f}-{max(a):.3f}]   cpu_below=0 "
          f"{statistics.median(b):7.3f} [{min(b):.3f}-{max(b):.3f}] ms  {extra}", flush=True)


def limit_in_use():
    """The small-image limit the default call uses now, and where it came from (the fixed default, this machine's
    saved measurement, or pinned), as device_report() gives it, without the settings file's path."""
    r = dict(egj.device_report()["cpu_below"])
    r.pop("settings_file", None)
    return r


res = dict(exact_gpu_jpeg=egj.__version__, measured=time.strftime("%Y-%m-%d"), limit_at_start=limit_in_use(), sets={})
print("limit in use at the start:", res["limit_at_start"], flush=True)
for mp, ss in [(0.02, 2), (0.05, 2), (0.1, 2), (0.1, 0), (0.15, 2), (0.2, 2), (0.2, 0), (0.25, 2), (0.5, 2), (2, 2)]:
    w = int(round((mp * 1e6 * 1.5) ** 0.5 / 8)) * 8
    h = int(round(w / 1.5 / 8)) * 8
    data = []
    for k in range(36):
        b = io.BytesIO()
        im = photos[k % len(photos)]
        im.crop((0, 0, im.width - 100 * (k // len(photos)), im.height)).resize((w, h), Image.LANCZOS).save(
            b, "JPEG", quality=90, subsampling=ss)
        data.append(b.getvalue())
    info = {}
    same = all(np.array_equal(egj.decode(x, info=info).get(), egj.decode(x, cpu_below=0).get()) for x in data[:12])
    rows = pair(data, 3)
    key = f"{mp:g}mp_{'420' if ss == 2 else '444'}"
    res["sets"][key] = dict(width=w, height=h, pixels=w * h, route=info["path"], identical=same, **rows)
    show(key, rows, f"{w}x{h} route {info['path']} identical {same}")
cam = [open(f, "rb").read() for f in files[:6]]
rows = pair(cam, 2, 6)
res["sets"]["camera_bytes"] = dict(files=len(cam), **rows)
show("camera photos, bytes", rows)
rows = pair(files[:6], 2, 6)
res["sets"]["camera_paths"] = dict(files=len(cam), **rows)
show("camera photos, paths", rows)
res["limit_at_end"] = limit_in_use()
print("limit in use at the end:", res["limit_at_end"], flush=True)
json.dump(res, open(out, "w"), indent=1)
