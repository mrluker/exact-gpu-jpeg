"""What the library's own timing features cost: off, and each of them on. Paired rounds, the modes in turn.
usage: python bench/timing_cost.py <out.json> [large.jpg]

Each cell is a decode or an encode, at 488x480 and of a large image (the file given, else a generated 24 MP one), in
eight rounds with the modes in turn: off, timings={}, Decoder.profile = True, profile = "host". Then what the marks
cost when everything is off: how many are passed per call and what one costs. Timings only."""
import io
import json
import os
import statistics
import sys
import time

os.environ["EGJ_SMALL_CPU_PIXELS"] = "200000"
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import cupy as cp  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import exact_gpu_jpeg as egj  # noqa: E402

out = sys.argv[1]
if len(sys.argv) > 2:                                  # a large JPEG of your own
    photo = open(sys.argv[2], "rb").read()
else:                                                  # else a generated 24 MP photo-like image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reproduce import synthetic_photo
    _b = io.BytesIO()
    Image.fromarray(synthetic_photo(6000, 4000, 1)).save(_b, "JPEG", quality=90)
    photo = _b.getvalue()
pc = time.perf_counter
med = statistics.median
src = Image.open(io.BytesIO(photo)).convert("RGB")
small_a = np.ascontiguousarray(np.asarray(src.resize((488, 480), Image.LANCZOS)))
b = io.BytesIO()
Image.fromarray(small_a).save(b, "JPEG", quality=90)
small = b.getvalue()
dec, enc = egj.Decoder(), egj.Encoder()
g_small = cp.asarray(small_a)
g_big = dec.decode(photo)
dec.stream.synchronize()
res = dict(env=dict(gpu=egj.device_report()["gpu"]), cells={})


def decode(data, mode):
    dec.profile = {"profile=True": True, 'profile="host"': "host"}.get(mode, False)
    t = pc()
    r = dec.decode(data, timings={} if mode == "timings={}" else None)
    dec.stream.synchronize()
    t = (pc() - t) * 1e3
    del r
    return t


def encode(g, mode):
    enc.profile = "host" if mode == 'profile="host"' else False
    t = pc()
    enc.encode(g, quality=90, subsampling="420", timings={} if mode == "timings={}" else None)
    return (pc() - t) * 1e3


CELLS = [("decode 488x480", decode, small, 150, ("off", "timings={}", "profile=True", 'profile="host"')),
         ("decode, large image", decode, photo, 10, ("off", "timings={}", "profile=True", 'profile="host"')),
         ("encode 488x480", encode, g_small, 150, ("off", "timings={}", 'profile="host"')),
         ("encode, large image", encode, g_big, 10, ("off", "timings={}", 'profile="host"'))]
ROUNDS = 8
for name, fn, arg, reps, modes in CELLS:
    for m in modes:
        for _ in range(3):
            fn(arg, m)
    rows = {m: [] for m in modes}
    for r in range(ROUNDS):
        order = modes if r % 2 == 0 else modes[::-1]
        for m in order:
            rows[m].append(med([fn(arg, m) for _ in range(reps)]))
    cell = {}
    for m in modes:
        d = [x - y for x, y in zip(rows[m], rows["off"])]
        cell[m] = dict(median_ms=round(med(rows[m]), 4), diff_vs_off_ms=round(statistics.mean(d), 4),
                       se=round(statistics.stdev(d) / len(d) ** 0.5, 4), slower_in=sum(x > 0 for x in d))
    res["cells"][name] = cell
    print(name, cell, flush=True)
dec.profile = enc.profile = False

# what a timing mark costs when everything is off: the calls made per decode / encode, and one such call
calls = {"n": 0}
real_dh, real_eh = type(dec)._host, type(enc)._host


def count_d(self, name):
    calls["n"] += 1
    return real_dh(self, name)


def count_e(self, name):
    calls["n"] += 1
    return real_eh(self, name)


type(dec)._host, type(enc)._host = count_d, count_e
marks = {}
for name, fn, arg in (("decode 488x480", decode, small), ("decode, large image", decode, photo), ("encode 488x480", encode, g_small), ("encode, large image", encode, g_big)):
    calls["n"] = 0
    fn(arg, "off")
    marks[name] = calls["n"]
type(dec)._host, type(enc)._host = real_dh, real_eh
dec.profile = True
dec.decode(small)
marks["decode 488x480, GPU-time marks (profile=True)"] = len(dec.stats.get("profile_ms", {}))
dec.profile = False
N = 2_000_000
t = pc()
for _ in range(N):
    dec._host("x")
per_call_ns = (pc() - t) / N * 1e9
t = pc()
for _ in range(N):
    pass
loop_ns = (pc() - t) / N * 1e9
res["off"] = dict(host_mark_calls_per_call=marks, ns_per_mark_call_when_off=round(per_call_ns - loop_ns, 1))
print(res["off"], flush=True)
json.dump(res, open(out, "w"), indent=1)
print("done", flush=True)
