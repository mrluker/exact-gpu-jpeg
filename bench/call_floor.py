"""The fixed cost of one call: where the time of decode() and encode() goes for an image so small that there is
almost no work in it (16x16 and 64x64), step by step, and what the pieces every step is made of cost on this
machine (a kernel launch, a small allocation, a readback, a wait).

Each step's time is the calling thread's (Decoder.profile = "host"): how long it took to queue the step's GPU work,
and, at the end, how long it waited for the GPU and read the result words back.

usage: python bench/call_floor.py <out.json> [dir of photos] [--sched auto|spin|yield|block] [--quick]
  --sched: the CUDA host-wait mode (cudaSetDeviceFlags, set before the context is made). auto is CUDA's default
           and what CuPy uses.
  dir of photos: also time whole calls at 0.25 MP (made from the photos) and on the photos themselves.
Counts and timings only (no file names)."""
import ctypes
import glob
import io
import json
import os
import statistics
import sys
import time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SCHED = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--sched"), "auto")
QUICK = "--quick" in sys.argv
out_path = args[0]
photo_dir = args[1] if len(args) > 1 and args[1] != SCHED else None
FLAGS = {"auto": 0, "spin": 1, "yield": 2, "block": 4}


def cudart():
    import importlib.util
    for mod in ("nvidia.cuda_runtime", "torch"):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.submodule_search_locations:
            continue
        for base in spec.submodule_search_locations:
            for pat in ("bin/cudart64_*.dll", "lib/cudart64_*.dll"):
                hits = glob.glob(os.path.join(base, pat))
                if hits:
                    return ctypes.WinDLL(hits[0])
    return None


rt = cudart()
set_rc = rt.cudaSetDeviceFlags(ctypes.c_uint(FLAGS[SCHED])) if rt is not None and SCHED != "auto" else None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from exact_gpu_jpeg._header import parse  # noqa: E402

pc = time.perf_counter
med = statistics.median
res = dict(lib=egj.__file__, cupy=cp.__version__, sched=SCHED, set_flags_status=set_rc,
           measured=time.strftime("%Y-%m-%d %H:%M"))
cp.zeros(1)
if rt is not None:
    f = ctypes.c_uint(99)
    rt.cudaGetDeviceFlags(ctypes.byref(f))
    res["device_flags"] = f.value          # bits 0-2: 0 auto, 1 spin, 2 yield, 4 blocking sync
print("lib", egj.__file__, "| cupy", cp.__version__, "| sched", SCHED, "| device flags", res.get("device_flags"),
      flush=True)
null = cp.cuda.Stream.null


def jpeg(w, h, ss=2, q=90, seed=0):
    r = np.random.default_rng(seed)
    a = (r.random((max(h // 8, 2), max(w // 8, 2), 3)) * 255).astype(np.uint8)
    a = np.asarray(Image.fromarray(a).resize((w, h), Image.BICUBIC))
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=ss)
    return b.getvalue(), a


def spread(v):
    return [round(med(v), 4), round(min(v), 4), round(max(v), 4)]


def steps(rows):
    """[(name, ms)] per call -> {name: median ms}, in order."""
    names = [n for n, _ in rows[0]]
    return {n: round(med([r[k][1] for r in rows]), 4) for k, n in enumerate(names)}


# ---- 1. decode ----
N = 100 if QUICK else 400
dec = egj.Decoder()
res["decode"] = {}
for w, h in ((16, 16), (64, 64)):
    data, a = jpeg(w, h)
    ref = egj.decode_cpu(data)
    assert np.array_equal(dec.decode(data).get(), ref)
    for _ in range(20):
        dec.decode(data)
    dec.profile = "host"
    rows, tot = [], []
    for _ in range(N):
        t = pc()
        dec.decode(data)
        tot.append((pc() - t) * 1e3)
        rows.append(dec.stats["host_ms"])
    dec.profile = False
    plain, wrap = [], []
    for _ in range(N):
        t = pc()
        dec.decode(data)
        plain.append((pc() - t) * 1e3)
        t = pc()
        egj.decode(data, cpu_below=0)
        wrap.append((pc() - t) * 1e3)
    dec.profile = True
    gpu = []
    for _ in range(N // 4):
        dec.decode(data)
        gpu.append(list(dec.stats["profile_ms"].items()))
    dec.profile = False
    st = steps(rows)
    res["decode"][f"{w}x{h}"] = dict(host_ms=st, sum=round(sum(st.values()), 4), call_profiled=spread(tot),
                                     call=spread(plain), egj_decode=spread(wrap), gpu_ms=steps(gpu),
                                     gpu_sum=round(sum(steps(gpu).values()), 4), bytes=len(data))
    print(f"\ndecode {w}x{h} ({len(data)} bytes): Decoder.decode {med(plain):.3f} ms [{min(plain):.3f}-{max(plain):.3f}], "
          f"egj.decode(cpu_below=0) {med(wrap):.3f} ms; host steps (sum {sum(st.values()):.3f}):", flush=True)
    for n, v in st.items():
        print(f"   {n:28s} {v:7.4f}")
    g = steps(gpu)
    print(f"   GPU time start to colour {sum(g.values()):.3f} ms: " + ", ".join(f"{n} {v:.3f}" for n, v in g.items()))

# ---- 2. encode ----
enc = egj.Encoder()
res["encode"] = {}
for w, h in ((16, 16), (64, 64)):
    _, a = jpeg(w, h)
    x = cp.asarray(a)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=90, subsampling=2)
    assert enc.encode(x, 90, "420") == b.getvalue()
    for _ in range(20):
        enc.encode(x, 90, "420")
    enc.profile = "host"
    rows = []
    for _ in range(N):
        enc.encode(x, 90, "420")
        rows.append(enc.stats["host_ms"])
    enc.profile = False
    plain = []
    for _ in range(N):
        t = pc()
        enc.encode(x, 90, "420")
        plain.append((pc() - t) * 1e3)
    st = steps(rows)
    res["encode"][f"{w}x{h}"] = dict(host_ms=st, sum=round(sum(st.values()), 4), call=spread(plain))
    print(f"\nencode {w}x{h}: Encoder.encode {med(plain):.3f} ms [{min(plain):.3f}-{max(plain):.3f}]; host steps "
          f"(sum {sum(st.values()):.3f}):", flush=True)
    for n, v in st.items():
        print(f"   {n:38s} {v:7.4f}")

# ---- 3. what the pieces cost ----
nop = cp.RawKernel(r"""
extern "C" __global__ void nop(const int* a, int n, const int* b, const int* c, const int* d, int e, int f,
                               const int* g, const int* h, int i, int j, int* o) { if (n < 0) o[0] = a[0]; }
""", "nop")
s = cp.cuda.Stream(non_blocking=True)
P = {}
with s:
    a = cp.zeros(4096, cp.int32)
    o = cp.zeros(8, cp.int32)
    i32 = np.int32
    kargs = (a, i32(1), a, a, a, i32(1), i32(1), a, a, i32(1), i32(1), o)
    nop((1,), (128,), kargs)
    s.synchronize()

    def timed(fn, reps=300, sync_each=True):
        ts = []
        for _ in range(reps):
            t = pc()
            fn()
            ts.append((pc() - t) * 1e6)
            if sync_each:
                s.synchronize()
        return round(med(ts), 2)

    for L in (1, 15, 45):
        iss, wait = [], []
        for _ in range(200):
            t = pc()
            for _ in range(L):
                nop((1,), (128,), kargs)
            t1 = pc()
            s.synchronize()
            t2 = pc()
            iss.append((t1 - t) * 1e6)
            wait.append((t2 - t1) * 1e6)
        P[f"{L} empty launches: queue them"] = round(med(iss), 1)
        P[f"{L} empty launches: wait for them"] = round(med(wait), 1)
        P[f"{L} empty launches: per launch, all in"] = round((med(iss) + med(wait)) / L, 2)
    P["one RawKernel launch (queue only, 12 arguments)"] = timed(lambda: nop((1,), (128,), kargs))
    P["wait on an idle stream"] = timed(s.synchronize, sync_each=False)
    P["cp.empty(16)"] = timed(lambda: cp.empty(16, cp.int32))
    P["cp.zeros(16) (allocation + fill launch)"] = timed(lambda: cp.zeros(16, cp.int32))
    P["x.fill(v)"] = timed(lambda: o.fill(3))
    P["cp.cumsum of 4096 ints"] = timed(lambda: cp.cumsum(a, dtype=cp.int64))
    P["x.astype"] = timed(lambda: a.astype(cp.int64))
    P["x + y"] = timed(lambda: a + a)
    P["x[y] (take)"] = timed(lambda: a[a])
    P["cp.concatenate of 4 small arrays"] = timed(lambda: cp.concatenate([o, o, o, o]))
    P["a slice (no GPU work)"] = timed(lambda: a[1:], sync_each=False)
    P["cp.cuda.Event() + record"] = timed(lambda: cp.cuda.Event().record(s))
    P["small readback, stream idle (x.get)"] = timed(lambda: o.get(stream=s), sync_each=False)

    def launch_get():
        nop((1,), (128,), kargs)
        o.get(stream=s)
    P["one launch + readback (round trip)"] = timed(launch_get, sync_each=False)
    P["cudaMemGetInfo"] = timed(lambda: cp.cuda.runtime.memGetInfo(), sync_each=False)
    P["memory pool free_bytes()"] = timed(lambda: cp.get_default_memory_pool().free_bytes(), sync_each=False)
    pin = dec._pinned(1 << 16)
    raw = cp.empty(4096, cp.uint8)
    P["pinned -> GPU copy of 4 kB (queue only)"] = timed(lambda: raw.set(pin[:4096], stream=s))
    hostarr = np.zeros(64, np.int32)
    P["cp.asarray of 64 ints (pageable)"] = timed(lambda: cp.asarray(hostarr))
d64, _ = jpeg(64, 64)
P["parse the header (64x64 file)"] = timed(lambda: parse(d64), sync_each=False)
h64 = parse(d64)
P["device tables lookup (cached)"] = timed(lambda: D._device_layout(h64, True), sync_each=False)
res["pieces_us"] = P
print("\nwhat the pieces cost (microseconds, median):")
for n, v in P.items():
    print(f"   {n:48s} {v:9.2f}")

# is the wait a spinning one? CPU time used by this process while it waits 0.3 s for the GPU
busy = cp.RawKernel(r"""
extern "C" __global__ void busy(float* o, int n) {
    float v = 0; for (int k = 0; k < n; k++) for (int j = 0; j < 1000; j++) v += sinf(v + j);
    o[threadIdx.x] = v; }
""", "busy")
with s:
    of = cp.zeros(1024, cp.float32)
    busy((64,), (256,), (of, i32(200)))
    s.synchronize()
    n = 200
    while True:
        t = pc()
        busy((64,), (256,), (of, i32(n)))
        s.synchronize()
        el = pc() - t
        if el > 0.2 or n > 1 << 24:
            break
        n *= 2
    c0, t0 = time.process_time(), pc()
    busy((64,), (256,), (of, i32(n)))
    s.synchronize()
    cpu, wall = time.process_time() - c0, pc() - t0
res["wait_cpu_share"] = round(cpu / wall, 3)
print(f"\nwaiting {wall * 1e3:.0f} ms for the GPU used {cpu * 1e3:.0f} ms of CPU time ({cpu / wall:.0%}: "
      f"{'a spinning wait' if cpu / wall > 0.7 else 'a sleeping wait'})", flush=True)

# ---- 4. whole calls on real sizes ----
if photo_dir:
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(photo_dir, "*.jpg"))}.values())[:12]
    photos = [Image.open(f).convert("RGB") for f in files[:6]]
    small = []
    for k in range(24):
        b = io.BytesIO()
        photos[k % len(photos)].crop((0, 0, 6000 - 100 * (k // len(photos)), 4000)).resize((616, 408), Image.LANCZOS).save(
            b, "JPEG", quality=90, subsampling=2)
        small.append(b.getvalue())
    cam = [open(f, "rb").read() for f in files[:6]]
    res["calls"] = {}
    for name, data, reps in (("0.25 MP", small, 3), ("camera photos (bytes)", cam, 2)):
        for x in data:
            dec.decode(x)
        rounds = []
        for _ in range(4 if QUICK else 8):
            ts = []
            for _ in range(reps):
                for x in data:
                    t = pc()
                    dec.decode(x)
                    ts.append((pc() - t) * 1e3)
            rounds.append(med(ts))
        res["calls"][name] = spread(rounds)
        print(f"Decoder.decode, {name}: {med(rounds):.3f} ms [{min(rounds):.3f}-{max(rounds):.3f}]", flush=True)
json.dump(res, open(out_path, "w"), indent=1)
