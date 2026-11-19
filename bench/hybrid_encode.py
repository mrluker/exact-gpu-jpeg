"""Encode throughput when a program splits a stream of images between exact-gpu-jpeg's GPU threads and CPU threads
running Pillow: exact-gpu-jpeg writes Pillow's bytes, so the two routes may share one queue and every file is still
the file Pillow writes. exact-gpu-jpeg does not do this split itself; this script is the program that does, measured.

usage: python bench/hybrid_encode.py <out.json> <dir of source photos> [--gpu 2] [--cpu 2,4,6] [--rounds 3]
       [--seconds 2] [--photos 6]

Inputs as bench/encode_bench.py: the first --photos photos of the folder resized (Lanczos) to 3840 px q95 4:4:4,
1500 px q90 4:4:4 and 1200 px q87 4:2:0, with their EXIF, as RGB arrays in host memory (a NumPy array: the GPU
threads upload it, like for like with the CPU encoders). Each encoder ends with the JPEG bytes in host memory.
For each size:
  - exactness first: one untimed pass of each combination below over the images, every file compared with Pillow's,
    byte for byte (the combinations with simplejpeg are compared too, and reported: simplejpeg does not write
    Pillow's bytes);
  - images per second: exact-gpu-jpeg on the --gpu threads alone (egj.encode: an Encoder per thread), Pillow on 8 CPU
    threads alone, simplejpeg on 8 CPU threads alone, and the --gpu GPU threads with N CPU threads pulling from one
    shared queue, N from --cpu, the CPU threads running Pillow, and again running simplejpeg. Every worker encodes one
    image before the clock starts; then all take the next image until --seconds have passed; the count covers every
    image finished. Each measurement is repeated --rounds times, the configurations taking turns in each round; the
    median is reported with each round's value.
No file names in the results. Run it alone (timing).
"""
import glob
import io
import itertools
import json
import os
import platform
import statistics
import sys
import threading
import time

import numpy as np
import simplejpeg
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402

SETTINGS = [(3840, 95, "444"), (1500, 90, "444"), (1200, 87, "420")]
CPU_ALONE = 8


def pil(a, q, ss, exif):
    b = io.BytesIO()
    kw = {"exif": exif} if exif else {}
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling={"444": 0, "420": 2}[ss], **kw)
    return b.getvalue()


def sj(a, q, ss, exif):
    return simplejpeg.encode_jpeg(a, quality=q, colorsubsampling=ss, fastdct=False)


def gpu(a, q, ss, exif):
    return bytes(egj.encode(a, q, ss, None, exif))


def run(items, q, ss, n_gpu, cpu_fn, n_cpu, seconds, keep=False):
    """n_gpu GPU threads and n_cpu CPU threads (cpu_fn) take images from one shared counter until `seconds` have
    passed (or, with keep, for one pass, returning every file). Returns (images/s, images per route, files)."""
    nxt, lock = itertools.count(), threading.Lock()
    done = {"gpu": 0, "cpu": 0}
    outs = [None] * len(items)
    start, go = threading.Barrier(n_gpu + n_cpu + 1), threading.Barrier(n_gpu + n_cpu + 1)
    stop_at = [None]

    def worker(route):
        fn = gpu if route == "gpu" else cpu_fn
        if not keep:
            fn(items[0][0], q, ss, items[0][1])            # warm this thread (its Encoder and stream)
        start.wait()
        go.wait()
        k = 0
        while True:
            with lock:
                i = next(nxt)
            if keep:
                if i >= len(items):
                    break
                outs[i] = fn(items[i][0], q, ss, items[i][1])
            else:
                if time.perf_counter() >= stop_at[0]:
                    break
                a, ex = items[i % len(items)]
                fn(a, q, ss, ex)
            k += 1
        with lock:
            done[route] += k

    ts = [threading.Thread(target=worker, args=("gpu",)) for _ in range(n_gpu)]
    ts += [threading.Thread(target=worker, args=("cpu",)) for _ in range(n_cpu)]
    for t in ts:
        t.start()
    start.wait()
    t0 = time.perf_counter()
    stop_at[0] = t0 + seconds
    go.wait()
    for t in ts:
        t.join()
    cp.cuda.Device().synchronize()
    el = time.perf_counter() - t0
    return (done["gpu"] + done["cpu"]) / el, dict(done), outs


def main():
    out, d, args = sys.argv[1], sys.argv[2], sys.argv[3:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default
    n_gpu = int(opt("--gpu", "2"))
    cpu_ns = [int(x) for x in opt("--cpu", "2,4,6").split(",")]
    rounds, seconds, n_photos = int(opt("--rounds", "3")), float(opt("--seconds", "2")), int(opt("--photos", "6"))
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())[:n_photos]
    srcs = []
    for f in files:
        with Image.open(f) as im:
            srcs.append((im.convert("RGB"), im.info.get("exif")))
    egj.self_check()
    res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], cpu=platform.processor(),
               logical_cpus=os.cpu_count(), exact_gpu_jpeg=egj.__version__, pillow=Image.__version__,
               simplejpeg=simplejpeg.__version__, photos=len(srcs), gpu_threads=n_gpu, cpu_alone_threads=CPU_ALONE,
               cpu_threads_tried=cpu_ns, rounds=rounds, seconds_a_measurement=seconds,
               source="host memory (a NumPy array; the GPU threads upload it)", sizes={})
    for w, q, ss in SETTINGS:
        items = [(np.asarray(im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)), ex) for im, ex in srcs]
        want = [pil(a, q, ss, ex) for a, ex in items]
        configs = [("gpu_alone", n_gpu, pil, 0), ("pillow_alone", 0, pil, CPU_ALONE),
                   ("simplejpeg_alone", 0, sj, CPU_ALONE)]
        configs += [(f"gpu+pillow{n}", n_gpu, pil, n) for n in cpu_ns]
        configs += [(f"gpu+simplejpeg{n}", n_gpu, sj, n) for n in cpu_ns]
        r = dict(width=w, height=items[0][0].shape[0], quality=q, subsampling=ss)
        exact = {}
        for key, ng, fn, nc in configs:                  # one untimed pass of each combination, every file checked
            _, routes, outs = run(items, q, ss, ng, fn, nc, 0, keep=True)
            exact[key] = dict(identical_to_pillow=sum(o == w_ for o, w_ in zip(outs, want)), files=len(items),
                              routes=routes)
            if fn is pil and exact[key]["identical_to_pillow"] != len(items):
                raise SystemExit(f"{w}: {key} wrote a file that differs from Pillow's")
        r["exactness_pass"] = exact
        per = {k: [] for k, *_ in configs}
        shares = {k: [] for k, *_ in configs}
        for rd in range(rounds):
            for key, ng, fn, nc in (configs if rd % 2 == 0 else configs[::-1]):
                ips, routes, _ = run(items, q, ss, ng, fn, nc, seconds)
                per[key].append(round(ips, 2))
                shares[key].append(round(routes["gpu"] / max(1, routes["gpu"] + routes["cpu"]), 3))
                cp.get_default_memory_pool().free_all_blocks()
        r["images_per_s"] = {k: round(statistics.median(v), 2) for k, v in per.items()}
        r["rounds"] = per
        r["gpu_share_of_images"] = {k: statistics.median(v) for k, v in shares.items() if k.startswith("gpu+")}
        for lib in ("pillow", "simplejpeg"):
            best = max(cpu_ns, key=lambda n: r["images_per_s"][f"gpu+{lib}{n}"])
            r[f"best_{lib}_threads"] = best
            r[f"best_gpu+{lib}_per_s"] = r["images_per_s"][f"gpu+{lib}{best}"]
        res["sizes"][f"{w}_q{q}_{ss}"] = r
        print(f"{w}_q{q}_{ss}", json.dumps(r["images_per_s"]), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
