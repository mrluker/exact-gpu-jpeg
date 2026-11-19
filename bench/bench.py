"""Decode speed: exact-gpu-jpeg vs simplejpeg (libjpeg-turbo, CPU) and torchvision.io.decode_jpeg on the GPU
(nvJPEG). Needs torch + torchvision with CUDA in the same environment (for the nvJPEG rows).

usage: python bench/bench.py <out.json> <group_name>=<dir or glob> [...] [--private NAME] [--about FILE.json]

Every decoder starts from the JPEG bytes in memory (no disk reads) and ends with the image decoded:
  - simplejpeg: RGB numpy array in host memory (accurate IDCT, fancy upsampling);
  - exact-gpu-jpeg: RGB CuPy array on the GPU, stream synchronised;
  - nvJPEG: RGB torch tensor on the GPU (torchvision.io.decode_jpeg, device='cuda'), synchronised.
Latency: median over files and repeats, one image at a time. call_ms: the library's whole call (egj.decode, its
default route: the GPU, the small-image CPU route, or the CPU decoder for a file the GPU path declines), the result
on the GPU; routes: how many files took each route (the reason, for declined ones). Throughput: images/s over the whole group repeated,
with N CPU threads (simplejpeg), N GPU decoder threads (exact-gpu-jpeg), or batched calls (nvJPEG, the group
repeated to at least 8 images). Exactness against simplejpeg is recorded for both GPU decoders. Groups named with --private are
reported without file names.
"""
import glob
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simplejpeg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import exact_gpu_jpeg as egj  # noqa: E402

MIN_SECONDS = 2.0


def cpu(data):
    return simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False)


def timed_repeat(fn, items, min_s=MIN_SECONDS):
    """Per-item wall times (ms), repeating the list until min_s has passed (at least twice)."""
    ts, t0, k = [], time.perf_counter(), 0
    while k < 2 or time.perf_counter() - t0 < min_s:
        for x in items:
            a = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - a) * 1000)
        k += 1
    return ts


def throughput(fn, items, threads, min_s=MIN_SECONDS, sync=None):
    items = items * -(-16 // len(items))                  # at least 16 tasks per round, so every thread has work
    f0 = fn
    fn = lambda x: (f0(x), None)[1]                       # drop each result at once: keeping a round's decoded
                                                          # images alive (24 x 90 MB) pushed the 8 GB card into paging
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(fn, items[:threads]))                 # warm each thread
        if sync:
            sync()
        n, t0 = 0, time.perf_counter()
        while n == 0 or time.perf_counter() - t0 < min_s:
            list(ex.map(fn, items))
            n += len(items)
        if sync:
            sync()
        return n / (time.perf_counter() - t0)


def main():
    out = sys.argv[1]
    private = set()
    groups = []
    about = None
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--private":
            private.add(args[i + 1])
        elif a == "--about":                             # what the files are (bench/make_ladder.py's ladder.json)
            about = json.load(open(args[i + 1]))
        elif "=" in a and (i == 0 or args[i - 1] not in ("--private", "--about")):
            name, pat = a.split("=", 1)
            files = sorted(glob.glob(os.path.join(pat, "*.jpg"))) if os.path.isdir(pat) else sorted(glob.glob(pat))
            groups.append((name, files))
    import cupy as cp
    try:
        import torch
        import torchvision
        from torchvision.io import ImageReadMode, decode_jpeg
    except ImportError:
        torch = None
    dev = cp.cuda.runtime.getDeviceProperties(0)
    env = dict(gpu=dev["name"].decode(), cpu=platform.processor(), cpu_threads=os.cpu_count(), os=platform.platform(),
               python=platform.python_version(), cupy=cp.__version__, simplejpeg=simplejpeg.__version__,
               libjpeg_turbo="3.1.2 (bundled in simplejpeg 1.9.0)", exact_gpu_jpeg=egj.__version__,
               torch=torch.__version__ if torch else None, torchvision=torchvision.__version__ if torch else None)
    res = dict(env=env, groups={})
    if about is not None:
        res["about_the_files"] = about
    dec = egj.Decoder()
    for name, files in groups:
        datas = [open(f, "rb").read() for f in files]
        info = [egj.inspect(d) for d in datas]
        g = dict(n=len(datas), megapixels=round(statistics.mean(i.get("width", 0) * i.get("height", 0) for i in info) / 1e6, 1),
                 subsampling=sorted({str(i.get("subsampling")) for i in info}),
                 restart=sorted({i.get("restart_interval") for i in info if i.get("supported")}),
                 mb_per_file=round(statistics.mean(len(d) for d in datas) / 1e6, 2))
        if name not in private:
            g["files"] = [os.path.basename(f) for f in files]
        refs = [cpu(d) for d in datas]
        # exactness
        outs = [dec.decode(d) for d in datas]
        g["gpu_supported"] = sum(o is not None for o in outs)
        g["gpu_exact"] = sum(o is not None and np.array_equal(o.get(), r) for o, r in zip(outs, refs))
        del outs
        sup = [d for d in datas if egj.inspect(d)["supported"]]
        # latency
        g["cpu_ms"] = statistics.median(timed_repeat(cpu, datas))

        def call(d):
            o = egj.decode(d)
            cp.cuda.Device().synchronize()
            return o
        call(datas[0])
        g["call_ms"] = statistics.median(timed_repeat(call, datas))
        routes = {}
        for d in datas:
            inf = {}
            egj.decode(d, info=inf)
            k = inf["path"] if inf["path"] == "gpu" else f'{inf["path"]}: {inf["reason"]}'
            routes[k] = routes.get(k, 0) + 1
        g["routes"] = routes

        def gpu(d):
            o = dec.decode(d)
            assert o is not None, dec.reason
            return o
        if sup:
            gpu(sup[0])
            tm = {}
            g["gpu_ms"] = statistics.median(timed_repeat(gpu, sup))
            for d in sup:
                dec.decode(d, timings=tm)
            g["gpu_split_ms"] = {k: round(v * 1000 / len(sup), 2) for k, v in tm.items()}
        # throughput
        g["cpu_1t_per_s"] = throughput(cpu, datas, 1)
        g["cpu_8t_per_s"] = throughput(cpu, datas, 8)
        if sup:
            sync = cp.cuda.Device().synchronize
            for n in (1, 2):
                g[f"gpu_{n}t_per_s"] = throughput(lambda d: egj.decode(d, fallback=False), sup, n, sync=sync)
        if torch is not None:
            ts = [torch.frombuffer(bytearray(d), dtype=torch.uint8) for d in datas]
            tb = ts * -(-8 // len(ts))                            # batches of at least 8 images
            try:
                o = [decode_jpeg(t, mode=ImageReadMode.RGB, device="cuda") for t in ts]
                torch.cuda.synchronize()
                diffs = [(x.permute(1, 2, 0).cpu().numpy().astype(np.int16) - r) for x, r in zip(o, refs)]
                g["nvjpeg_exact"] = sum(not d.any() for d in diffs)
                g["nvjpeg_max_abs_diff"] = int(max(np.abs(d).max() for d in diffs))
                g["nvjpeg_mean_abs_diff"] = round(float(np.mean([np.abs(d).mean() for d in diffs])), 3)
                del o, diffs

                def nv(t):
                    x = decode_jpeg(t, mode=ImageReadMode.RGB, device="cuda")
                    torch.cuda.synchronize()
                    return x
                g["nvjpeg_ms"] = statistics.median(timed_repeat(nv, ts))
                n, t0 = 0, time.perf_counter()
                while n == 0 or time.perf_counter() - t0 < MIN_SECONDS:
                    decode_jpeg(tb, mode=ImageReadMode.RGB, device="cuda")
                    torch.cuda.synchronize()
                    n += len(tb)
                g["nvjpeg_batch_per_s"] = n / (time.perf_counter() - t0)
            except Exception as e:                               # noqa: BLE001
                g["nvjpeg_error"] = repr(e)[:200]
            torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        g = {k: round(v, 2) if isinstance(v, float) else v for k, v in g.items()}
        res["groups"][name] = g
        print(name, json.dumps({k: v for k, v in g.items() if k != "files"}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
