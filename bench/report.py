"""Markdown tables from bench/bench.py output.

usage: python bench/report.py <bench.json> [conditions.json] > table.md
"""
import json
import sys


def f(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    r = json.load(open(sys.argv[1]))
    env = r["env"]
    cond = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    print(f"GPU {env['gpu']}; CPU {env['cpu']} ({env['cpu_threads']} threads); {env['os']}; Python {env['python']}, "
          f"CuPy {env['cupy']}, simplejpeg {env['simplejpeg']} (libjpeg-turbo 3.1.2), torch {env['torch']}, "
          f"torchvision {env['torchvision']}.")
    if cond:
        idle = cond.get("idle_before") or cond.get("idle") or {}
        if idle:
            print(f"Machine otherwise idle before the run (GPU ≤ {idle.get('gpu_util_max')}%, "
                  f"CPU ≤ {idle.get('cpu_load_max')}%), shared-PC lock held.")
    print()
    print("Latency, one image at a time (median ms, JPEG bytes in memory → decoded image):")
    print()
    print("| image | MP | MB | libjpeg-turbo 1 thread (CPU) | **exact-gpu-jpeg** | nvJPEG (torchvision) | GPU split: upload / Huffman / pixels |")
    print("|---|---|---|---|---|---|---|")
    for name, g in r["groups"].items():
        sp = g.get("gpu_split_ms", {})
        split = " / ".join(f(sp.get(k)) for k in ("upload", "huffman", "pixels")) if sp else "–"
        print(f"| {name} | {g['megapixels']} | {g['mb_per_file']} | {f(g.get('cpu_ms'))} | **{f(g.get('gpu_ms'))}** | "
              f"{f(g.get('nvjpeg_ms'))} | {split} |")
    print()
    print("Throughput (images/s):")
    print()
    print("| image | CPU 1 thread | CPU 8 threads | **GPU 1 thread** | **GPU 2 threads** | nvJPEG batched | exact (ours) | nvJPEG max / mean abs diff |")
    print("|---|---|---|---|---|---|---|---|")
    for name, g in r["groups"].items():
        ex = f"{g.get('gpu_exact')}/{g['n']}"
        nv = f"{g.get('nvjpeg_max_abs_diff', '–')} / {g.get('nvjpeg_mean_abs_diff', '–')}"
        print(f"| {name} | {f(g.get('cpu_1t_per_s'))} | {f(g.get('cpu_8t_per_s'))} | **{f(g.get('gpu_1t_per_s'))}** | "
              f"**{f(g.get('gpu_2t_per_s'))}** | {f(g.get('nvjpeg_batch_per_s'))} | {ex} | {nv} |")


if __name__ == "__main__":
    main()
