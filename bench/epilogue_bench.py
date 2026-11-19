"""Decode straight to a model input: a normalised float16 CHW array (ImageNet mean / std), written by the decoder's
colour kernel (dtype / layout / scale / bias), against decoding to uint8 and converting with CuPy afterwards.

usage: python bench/epilogue_bench.py <dir of JPEGs> [out.json]

From JPEG bytes in memory, one photo at a time, 3 rounds; ms per photo, synchronised. Also checks that both give
the same float16 bits. Prints counts and timings only (no file names), and writes them to out.json if given.
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402

files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(sys.argv[1], "*.jpg"))}.values())
data = [open(f, "rb").read() for f in files]
dec = egj.Decoder()
mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sc, bi = cp.asarray(1 / (255 * std)), cp.asarray(-mean / std)


def two_step(d):
    img = dec.decode(d)
    return cp.ascontiguousarray((img.astype(cp.float32) * sc + bi).astype(cp.float16).transpose(2, 0, 1))


def fused(d):
    return dec.decode(d, dtype="float16", layout="CHW", scale=1 / (255 * std), bias=-mean / std)


for f in (two_step, fused):
    f(data[0])
print(f"{len(data)} photos")
rounds = {"decode + CuPy convert": [], "fused epilogue": []}
for r in range(3):
    for name, f in (("decode + CuPy convert", two_step), ("fused epilogue", fused)):
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        for d in data:
            f(d)
        cp.cuda.Device().synchronize()
        ms = (time.perf_counter() - t) * 1e3 / len(data)
        rounds[name].append(round(ms, 3))
        print(f"r{r} {name:22}: {ms:.2f} ms per photo")
same = all(np.array_equal(two_step(d).get().view(np.uint16), fused(d).get().view(np.uint16)) for d in data)
print("identical (all photos):", same)
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        json.dump(dict(what="decode to a normalised float16 CHW array: fused into the colour kernel, against decode "
                            "+ CuPy convert; ms per photo, 3 rounds", photos=len(data),
                       decode_then_cupy_ms=rounds["decode + CuPy convert"], fused_ms=rounds["fused epilogue"],
                       identical=same, gpu=egj.device_report()["gpu"],
                       measured=time.strftime("%Y-%m-%d %H:%M")), f, indent=1)
