"""Compare the GPU path with libjpeg-turbo's three x86-64 code paths (AVX2, SSE2, plain C), each decoded in
its own process (JSIMD_FORCEAVX2 / JSIMD_FORCESSE2 / JSIMD_FORCENONE).

usage: python tools/check_simd.py <work_dir> [--extreme N]

Files: the synthetic test corpus (tests/corpus.py) plus N crafted files with random, often extreme,
coefficients (tests/jpegwriter.py). For each file: whether the three references agree, and whether the GPU
output (if any) equals all of them. A GPU image that differs from any of the three is a failure.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--extreme", type=int, default=400)
    a = ap.parse_args()
    import corpus
    import exact_gpu_jpeg as egj
    from jpegwriter import grid, write_jpeg
    from refdecode import digest, refs
    d = os.path.join(a.work_dir, "files")
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, data, _ in corpus.generate():
        p = os.path.join(d, name)
        open(p, "wb").write(data)
        paths[p] = "corpus"
    rng = np.random.default_rng(11)
    for k in range(a.extreme):
        samp = [[(1, 1)], [(1, 1)] * 3, [(2, 2), (1, 1), (1, 1)], [(2, 1), (1, 1), (1, 1)]][k % 4]
        W, H = int(rng.integers(8, 48)), int(rng.integers(8, 48))
        mag = [64, 256, 1023, 2047, 4096, 16383, 32767][k % 7]
        qmax = [1, 4, 16, 64, 255][(k // 7) % 5]
        blocks = []
        for ci in range(len(samp)):
            g = grid(samp, W, H, ci)
            b = np.zeros(g + (64,), np.int32)
            m = rng.random(b.shape) < rng.uniform(0.02, 1.0)
            b[m] = rng.integers(-mag, mag + 1, m.sum())
            b[..., 0] = rng.integers(-min(mag, 2047), min(mag, 2047) + 1, g)
            blocks.append(b)
        p = os.path.join(d, f"extreme_{k:04d}.jpg")
        open(p, "wb").write(write_jpeg(blocks, [rng.integers(1, qmax + 1, 64) for _ in samp], samp, W, H,
                                       dri=int(rng.integers(0, 3))))
        paths[p] = "extreme"
    R = refs(list(paths), a.work_dir)
    dec = egj.Decoder()
    c = collections.Counter()
    fails = []
    for p, group in paths.items():
        r = {m: R[m][p] for m in R}
        agree = len(set(r.values())) == 1 and "error" not in r.values()
        c[f"{group}: references agree" if agree else f"{group}: references differ"] += 1
        out = dec.decode(p)
        if out is None:
            c[f"{group}: GPU declined" + ("" if not agree else " (references agree)")] += 1
            continue
        dg = digest(out.get())
        if all(dg == v for v in r.values()):
            c[f"{group}: GPU equals all three"] += 1
        else:
            c[f"{group}: GPU DIFFERS"] += 1
            fails.append(os.path.basename(p))
    res = dict(counts=dict(sorted(c.items())), failures=fails, modes=list(R))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.work_dir, "check_simd.json"), "w"), indent=1)
    decoded = sum(v for k, v in c.items() if k.endswith("GPU equals all three"))
    return 1 if fails or not decoded else 0                    # (nothing decoded on the GPU: not a pass)


if __name__ == "__main__":
    sys.exit(main())
