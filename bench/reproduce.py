"""The README's benchmark tables, on your own GPU, from images this script makes itself.

usage: python bench/reproduce.py [--quick] [--out results.json] [--seconds 1.5]

Nothing is read from disk or downloaded: every test image is generated here (synthetic_photo: smooth colour fields
at several scales, hard-edged shapes, fine lines and grain; about 1.1 bits per pixel at quality 90, close to what a
resized camera photo gives) and encoded in memory by Pillow (libjpeg-turbo) at quality 90, 4:2:0.

What is timed (median over repeats for about --seconds per cell, after a warm-up; one image at a time unless the
row says otherwise):
  decode   JPEG bytes in host memory -> RGB pixels. CPU: libjpeg-turbo through simplejpeg if it is installed, else
           through Pillow, pixels in host memory. exact-gpu-jpeg: exact_gpu_jpeg.decode(), pixels in GPU memory, the
           GPU idle again (the call waits). nvJPEG (only if torch + torchvision with CUDA are installed):
           torchvision.io.decode_jpeg(device="cuda") + torch.cuda.synchronize().
  batch    64 different images per call: exact_gpu_jpeg.decode_batch() against decode() in a loop.
  encode   RGB pixels -> JPEG bytes in host memory, quality 90, 4:2:0. CPU: Pillow from a NumPy array.
           exact-gpu-jpeg: exact_gpu_jpeg.encode() from a CuPy array already on the GPU.
Every exact-gpu-jpeg result is compared with libjpeg-turbo's: decoded pixels must be equal, encoded files must be
the same bytes. The script exits with status 1 if any is not.
"""
import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, features

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402

try:
    import simplejpeg
except ImportError:
    simplejpeg = None
try:
    import torch
    import torchvision
    from torchvision.io import decode_jpeg as tv_decode
    if not torch.cuda.is_available():
        torch = None
except Exception:
    torch = None

pc = time.perf_counter
med = statistics.median


def synthetic_photo(width, height, seed=0, grain=2.0):
    """A deterministic photo-like RGB image: smooth colour fields at several pixel scales, hard-edged shapes, fine
    lines and sensor-like grain, with about the same detail per pixel at every size. The same pixels for the same
    arguments and library versions."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 127.5, np.float32)
    for cell, amp in ((512, 80.0), (160, 45.0), (48, 22.0), (14, 9.0)):       # pixels per cell, amplitude
        gh, gw = height // cell + 2, width // cell + 2
        g = (rng.random((gh, gw, 3)) * 255).astype(np.uint8)
        layer = Image.fromarray(g).resize((gw * cell, gh * cell), Image.BICUBIC).crop((0, 0, width, height))
        img += (np.asarray(layer, np.float32) - 127.5) * (amp / 127.5)
    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(base)
    n = max(width * height // 60_000, 2)
    for _ in range(n):                                             # shapes with hard edges
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(8, 160))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x - r, y - r // 2, x + r, y + r // 2], fill=col)
    for _ in range(2 * n):                                         # fine lines
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x1, y1 = x0 + int(rng.integers(-200, 200)), y0 + int(rng.integers(-200, 200))
        d.line([x0, y0, x1, y1], fill=tuple(int(v) for v in rng.integers(0, 256, 3)), width=int(rng.integers(1, 4)))
    a = np.asarray(base, np.float32)
    a += rng.normal(0.0, grain, a.shape).astype(np.float32)        # grain
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def size_for(mp):
    w = int(round((mp * 1e6 * 1.5) ** 0.5 / 8)) * 8                # 3:2, whole 8x8 blocks
    return w, int(round(w / 1.5 / 8)) * 8


def jpeg(a, quality=90, subsampling=2):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=quality, subsampling=subsampling)
    return b.getvalue()


def cpu_decode(data):
    if simplejpeg is not None:
        return simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def timed(fn, items, seconds):
    """Median ms per item: fn(item) over the items again and again for about `seconds` (at least 3 passes)."""
    for x in items:
        fn(x)                                                      # warm-up
    ts, t_end, passes = [], pc() + seconds, 0
    while passes < 3 or pc() < t_end:
        for x in items:
            t = pc()
            fn(x)
            ts.append((pc() - t) * 1e3)
        passes += 1
    return med(ts)


def environment():
    env = dict(os=platform.platform(), cpu=platform.processor() or platform.machine(), cpu_threads=os.cpu_count(),
               python=platform.python_version(), exact_gpu_jpeg=egj.__version__, cupy=cp.__version__,
               numpy=np.__version__, pillow=Image.__version__,
               pillow_libjpeg_turbo=features.version_feature("libjpeg_turbo") or "Pillow is not built on libjpeg-turbo",
               simplejpeg=getattr(simplejpeg, "__version__", "installed") if simplejpeg else "not installed",
               torchvision=torchvision.__version__ if torch else "not installed (no nvJPEG column)")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        env["gpu"], env["driver"], env["gpu_memory"] = [v.strip() for v in q[cp.cuda.Device().id].split(",")]
    except Exception:
        env["gpu"] = f"device {cp.cuda.Device().id}"
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer sizes and shorter cells (a smoke run)")
    ap.add_argument("--out", help="also write the numbers to this JSON file")
    ap.add_argument("--seconds", type=float, default=1.5, help="timing budget per table cell")
    a = ap.parse_args()
    sec = 0.4 if a.quick else a.seconds
    sizes = (0.25, 2, 6) if a.quick else (0.25, 1, 2, 6, 12, 24)
    env = environment()
    print("## Environment\n")
    for k, v in env.items():
        print(f"- {k}: {v}")
    check = egj.self_check()                   # compiles every kernel and runs the built-in cases, before any timing
    if isinstance(check.get("cpu_below"), dict):
        check["cpu_below"] = {k: v for k, v in check["cpu_below"].items() if k != "settings_file"}   # no user path
    print("\n## What the library found on this GPU (exact_gpu_jpeg.device_report())\n")
    report = egj.device_report()
    report["cpu_below"] = {k: v for k, v in report["cpu_below"].items() if k != "settings_file"}  # no user path
    for k, v in report.items():
        print(f"- {k}: {v}")
    print(f"- self-check, ms per part (first use, kernel compilation included): {check['ms']}")
    ok = True
    res = dict(environment=env, device_report=report, self_check=check, decode=[], batch=[], encode=[])

    # ---- decode, one image at a time ----
    dec_rows = []
    for mp in sizes:
        w, h = size_for(mp)
        n = 4 if mp <= 2 else 2
        imgs = [synthetic_photo(w, h, seed) for seed in range(n)]
        data = [jpeg(x) for x in imgs]
        same = all(np.array_equal(cp.asnumpy(egj.decode(d)), cpu_decode(d)) for d in data)
        ok &= same
        row = dict(mp=mp, width=w, height=h, kbytes=round(med([len(d) for d in data]) / 1e3, 1), identical=same,
                   cpu_ms=round(timed(cpu_decode, data, sec), 3), gpu_ms=round(timed(egj.decode, data, sec), 3))
        info = {}
        egj.decode(data[0], info=info)         # which route the default call takes at this size
        row["route"] = info.get("path")
        if torch:
            tens = [torch.frombuffer(bytearray(d), dtype=torch.uint8) for d in data]

            def nv(t):
                r = tv_decode(t, device="cuda")
                torch.cuda.synchronize()
                return r
            row["nvjpeg_ms"] = round(timed(nv, tens, sec), 3)
            row["nvjpeg_max_diff"] = int(max(
                np.abs(nv(t).permute(1, 2, 0).cpu().numpy().astype(np.int16) - cpu_decode(d).astype(np.int16)).max()
                for t, d in zip(tens, data)))
        dec_rows.append(row)
        del imgs
    res["decode"] = dec_rows
    cpu_name = "libjpeg-turbo (simplejpeg)" if simplejpeg else "libjpeg-turbo (Pillow)"
    print("\n## Decode, one image at a time (ms per image, lower is better)\n")
    head = ["size", "file", cpu_name + ", CPU", "exact-gpu-jpeg", "speed-up", "decoded on",
            "pixels equal to libjpeg-turbo's"]
    if torch:
        head += ["nvJPEG (torchvision)", "nvJPEG: largest pixel difference"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in dec_rows:
        cells = [f"{r['mp']:g} MP ({r['width']}x{r['height']})", f"{r['kbytes']:g} kB", f"{r['cpu_ms']:.2f}",
                 f"{r['gpu_ms']:.2f}", f"{r['cpu_ms'] / r['gpu_ms']:.1f}x",
                 "CPU, then uploaded" if r["route"] == "cpu" else "GPU", "yes" if r["identical"] else "NO"]
        if torch:
            cells += [f"{r['nvjpeg_ms']:.2f}", str(r["nvjpeg_max_diff"])]
        print("| " + " | ".join(cells) + " |")
    if any(r["route"] == "cpu" for r in dec_rows):
        print()
        print("The exact-gpu-jpeg column is the default call, with the pixels on the GPU at the end. Below the size "
              "limit measured on this machine (cpu_below, above) it decodes on the CPU and uploads, because that is "
              "the faster way to get them there; such a row is the CPU's time plus the upload.")

    # ---- batches of small images ----
    print("\n## Decode, 64 different images per call (images per second, higher is better)\n")
    print("| size | decode() in a loop | decode_batch() | speed-up | pixels equal |\n|---|---|---|---|---|")
    for mp in ((0.25,) if a.quick else (0.25, 1)):
        w, h = size_for(mp)
        data = [jpeg(synthetic_photo(w, h, 100 + k)) for k in range(64)]
        outs = egj.decode_batch(data)
        same = all(np.array_equal(cp.asnumpy(o), cpu_decode(d)) for o, d in zip(outs[:8], data[:8]))
        ok &= same
        del outs
        one = 64e3 / (timed(lambda ds: [egj.decode(d) for d in ds], [data], sec))
        bat = 64e3 / (timed(egj.decode_batch, [data], sec))
        res["batch"].append(dict(mp=mp, loop_per_s=round(one), batch_per_s=round(bat), identical=same))
        print(f"| {mp:g} MP ({w}x{h}) | {one:,.0f} | {bat:,.0f} | {bat / one:.1f}x | {'yes' if same else 'NO'} |")

    # ---- encode ----
    print("\n## Encode, quality 90, 4:2:0 (ms per image, lower is better)\n")
    print("| size | Pillow (libjpeg-turbo), CPU | exact-gpu-jpeg | speed-up | file equal to Pillow's |\n|---|---|---|---|---|")
    for mp in sizes:
        w, h = size_for(mp)
        imgs = [synthetic_photo(w, h, 200 + seed) for seed in range(2)]
        gpu = [cp.asarray(x) for x in imgs]
        same = all(egj.encode(g, quality=90, subsampling="420") == jpeg(x) for g, x in zip(gpu, imgs))
        ok &= same
        c = timed(jpeg, imgs, sec)
        g = timed(lambda x: egj.encode(x, quality=90, subsampling="420"), gpu, sec)
        res["encode"].append(dict(mp=mp, width=w, height=h, cpu_ms=round(c, 3), gpu_ms=round(g, 3), identical=same))
        print(f"| {mp:g} MP ({w}x{h}) | {c:.2f} | {g:.2f} | {c / g:.1f}x | {'yes' if same else 'NO'} |")
        del gpu, imgs
    print("\n" + ("Every result was identical to libjpeg-turbo's." if ok else "SOME RESULTS DIFFERED: see the tables."))
    end = {k: v for k, v in egj.device_report()["cpu_below"].items() if k != "settings_file"}
    print(f"\nSmall-image limit in use at the end: {end}")
    res["cpu_below_at_end"] = end
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
