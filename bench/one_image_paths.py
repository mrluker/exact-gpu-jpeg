"""One image at a time, two ways: Decoder.decode(x) (the single-image path) against Decoder.decode_batch([x]) with
the batch path allowed to take a batch of one (it normally hands a lone image to decode()). Paired: rounds alternate
the order. Sizes 0.25 to 6 MP are made from the photos (quality 90, 4:2:0); the photos themselves are the last row.
Each call is timed until the pixels are in GPU memory and the decoder's stream is idle.

usage: python bench/one_image_paths.py <out.json> <dir of photos> [rounds]
Counts and timings only (no file names)."""
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
from exact_gpu_jpeg import _batch  # noqa: E402

out, d = sys.argv[1], sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())[:12]
photos = [Image.open(f).convert("RGB") for f in files[:6]]
dec = egj.Decoder()
pc = time.perf_counter
med = statistics.median


def single(x):
    return dec.decode(x)


def batch1(x):
    return dec.decode_batch([x])[0]


def pair(data, reps):
    rows = {"single": [], "batch of one": []}
    fns = (("single", single), ("batch of one", batch1))
    for r in range(ROUNDS):
        for name, fn in (fns if r % 2 == 0 else fns[::-1]):
            ts = []
            for _ in range(reps):
                for x in data:
                    t = pc()
                    g = fn(x)
                    dec.stream.synchronize()
                    ts.append((pc() - t) * 1e3)
                    del g
            rows[name].append(round(med(ts), 4))
    return rows


res = dict(lib=egj.__file__, measured=time.strftime("%Y-%m-%d %H:%M"), sets={})
print("lib", egj.__file__, flush=True)
_batch.MIN_GROUP = 1
sets = []
for mp, n in ((0.25, 24), (0.5, 24), (1, 18), (2, 12), (6, 12)):
    w = int(round((mp * 1e6 * 1.5) ** 0.5 / 8)) * 8
    h = int(round(w / 1.5 / 8)) * 8
    data = []
    for k in range(n):
        b = io.BytesIO()
        photos[k % len(photos)].crop((0, 0, 6000 - 100 * (k // len(photos)), 4000)).resize((w, h), Image.LANCZOS).save(
            b, "JPEG", quality=90, subsampling=2)
        data.append(b.getvalue())
    sets.append((f"{mp:g} MP", data, 3 if mp < 2 else 2))
sets.append(("camera photos", [open(f, "rb").read() for f in files[:6]], 2))
for name, data, reps in sets:
    same = True
    for x in data[:6]:
        a, b = single(x), batch1(x)
        same &= dec.stats.get("batched") == 1 and bool((a == b).all()) and np.array_equal(a.get(), egj.decode_cpu(x))
    rows = pair(data, reps)
    a, b = rows["single"], rows["batch of one"]
    res["sets"][name] = dict(files=len(data), identical=same, **rows)
    print(f"{name:14s} single {med(a):7.3f} [{min(a):.3f}-{max(a):.3f}]   batch of one {med(b):7.3f} "
          f"[{min(b):.3f}-{max(b):.3f}] ms   identical {same}", flush=True)
json.dump(res, open(out, "w"), indent=1)
