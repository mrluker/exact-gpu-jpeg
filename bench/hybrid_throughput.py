"""Decode throughput when a program splits a stream of JPEGs between exact-gpu-jpeg's GPU threads and libjpeg-turbo
CPU threads: both routes give the same bytes, so a program may let both pull from one queue.

exact-gpu-jpeg does not do this split itself: egj.decode() routes each call (a small image to the CPU decoder, a
declined file to the CPU fallback) and egj.decode_batch() decodes a list on the GPU, but nothing in the library hands
a stream of files to GPU and CPU threads at once. This script is the program that does, measured.

usage: python bench/hybrid_throughput.py <out.json> <group>=<file, dir or glob> [...] [--private NAME]
       [--gpu 2] [--cpu 2,4,6] [--rounds 3] [--seconds 2]

For each group (its JPEG bytes in memory, no disk reads):
  - exactness first: every file through egj.decode (the GPU route, fallback=False) and through simplejpeg (accurate
    IDCT, fancy upsampling) must be byte-identical; then one untimed pass of every combination below over the group,
    each output compared with libjpeg-turbo's;
  - images per second, as bench/bench.py counts them, from JPEG bytes in host memory to the decoded RGB image (GPU
    threads: on the GPU, the device synchronised before the clock stops; CPU threads: in host memory):
      exact-gpu-jpeg on the --gpu threads alone (each thread has its own Decoder and CUDA stream: egj.decode),
      libjpeg-turbo (simplejpeg) on 8 CPU threads alone,
      both at once: the --gpu GPU threads and N CPU threads pulling from one shared queue, for each N of --cpu;
    every worker decodes one file before the clock starts; then all pull the next file until --seconds have passed,
    and the count covers every file finished. Each measurement is repeated --rounds times, the configurations taking
    turns in each round; the median is reported, with each round's value.
Results drop each decoded image at once (as bench/bench.py). Groups named with --private are reported without file
names. Run it alone (timing).
"""
import glob
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402

CPU_ALONE = 8


def cpu(data):
    return simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False)


def gpu(data):
    return egj.decode(data, fallback=False)


def run(datas, n_gpu, n_cpu, seconds, keep=False):
    """n_gpu GPU threads and n_cpu CPU threads pull files from one shared counter until `seconds` have passed (or,
    with keep, for one pass over the files, returning every output). Returns (images/s, images per route, outputs)."""
    nxt, lock = itertools.count(), threading.Lock()
    done = {"gpu": 0, "cpu": 0}
    outs = [None] * len(datas)
    start = threading.Barrier(n_gpu + n_cpu + 1)        # every worker warmed
    go = threading.Barrier(n_gpu + n_cpu + 1)           # the clock and the deadline set
    stop_at = [None]

    def worker(route):
        fn = gpu if route == "gpu" else cpu
        if not keep:
            fn(datas[0])                                   # warm this thread (its Decoder and stream, its cache)
            if route == "gpu":
                cp.cuda.get_current_stream().synchronize()
        start.wait()
        go.wait()
        k = 0
        while True:
            with lock:
                i = next(nxt)
            if keep:
                if i >= len(datas):
                    break
                img = fn(datas[i])
                outs[i] = (route, cp.asnumpy(img) if route == "gpu" else img)
            else:
                if time.perf_counter() >= stop_at[0]:
                    break
                fn(datas[i % len(datas)])
            k += 1
        if route == "gpu":
            cp.cuda.get_current_stream().synchronize()
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


def files_of(spec):
    if os.path.isdir(spec):
        fs = sorted(glob.glob(os.path.join(spec, "*.jpg")) + glob.glob(os.path.join(spec, "*.jpeg")))
    else:
        fs = sorted(glob.glob(spec)) or [spec]
    return [f for f in fs if os.path.isfile(f)]


def main():
    out, args = sys.argv[1], sys.argv[2:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default
    private = {args[i + 1] for i, a in enumerate(args) if a == "--private"}
    n_gpu = int(opt("--gpu", "2"))
    cpu_ns = [int(x) for x in opt("--cpu", "2,4,6").split(",")]
    rounds, seconds = int(opt("--rounds", "3")), float(opt("--seconds", "2"))
    skip = {i + 1 for i, a in enumerate(args) if a.startswith("--")}
    groups = [a.split("=", 1) for i, a in enumerate(args) if "=" in a and i not in skip and not a.startswith("--")]
    egj.self_check()
    res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"],
               cpu=platform.processor(), logical_cpus=os.cpu_count(), exact_gpu_jpeg=egj.__version__,
               simplejpeg=simplejpeg.__version__, gpu_threads=n_gpu, cpu_alone_threads=CPU_ALONE,
               cpu_threads_tried=cpu_ns, rounds=rounds, seconds_a_measurement=seconds, groups={})
    for name, spec in groups:
        fs = files_of(spec)
        datas = [open(f, "rb").read() for f in fs]
        refs = [cpu(d) for d in datas]
        mp = statistics.median(r.shape[0] * r.shape[1] / 1e6 for r in refs)
        g = dict(files=len(datas), megapixels=round(mp, 2))
        if name not in private:
            g["names"] = [os.path.basename(f) for f in fs]
        g["gpu_exact"] = sum(bool(np.array_equal(cp.asnumpy(gpu(d)), r)) for d, r in zip(datas, refs))
        if g["gpu_exact"] != len(datas):
            raise SystemExit(f"{name}: the GPU route is not byte-identical on every file")
        configs = [("gpu_alone", n_gpu, 0), ("cpu_alone", 0, CPU_ALONE)] + [(f"gpu+cpu{n}", n_gpu, n) for n in cpu_ns]
        exact = {}
        for key, ng, nc in configs:                      # one untimed pass of each combination, every output checked
            _, routes, outs = run(datas, ng, nc, 0, keep=True)
            exact[key] = dict(identical=sum(bool(np.array_equal(o[1], r)) for o, r in zip(outs, refs)), routes=routes)
            if exact[key]["identical"] != len(datas):
                raise SystemExit(f"{name}: {key} gave an output that differs from libjpeg-turbo's")
        g["exactness_pass"] = exact
        per = {k: [] for k, _, _ in configs}
        shares = {k: [] for k, _, _ in configs}
        for r in range(rounds):
            order = configs if r % 2 == 0 else configs[::-1]
            for key, ng, nc in order:
                ips, routes, _ = run(datas, ng, nc, seconds)
                per[key].append(round(ips, 2))
                shares[key].append(round(routes["gpu"] / max(1, routes["gpu"] + routes["cpu"]), 3))
                cp.get_default_memory_pool().free_all_blocks()
        g["images_per_s"] = {k: round(statistics.median(v), 2) for k, v in per.items()}
        g["rounds"] = per
        g["gpu_share_of_images"] = {k: statistics.median(v) for k, v in shares.items() if k.startswith("gpu+")}
        best = max(cpu_ns, key=lambda n: g["images_per_s"][f"gpu+cpu{n}"])
        g["best_cpu_threads"] = best
        g["best_hybrid_per_s"] = g["images_per_s"][f"gpu+cpu{best}"]
        g["best_hybrid_vs_gpu_alone"] = round(g["best_hybrid_per_s"] / g["images_per_s"]["gpu_alone"], 3)
        g["best_hybrid_vs_cpu_alone"] = round(g["best_hybrid_per_s"] / g["images_per_s"]["cpu_alone"], 3)
        res["groups"][name] = g
        print(name, json.dumps({k: g[k] for k in ("images_per_s", "best_cpu_threads", "best_hybrid_vs_gpu_alone",
                                                 "best_hybrid_vs_cpu_alone")}), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
