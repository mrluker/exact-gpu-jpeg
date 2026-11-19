"""Fast-DCT mode (Decoder(fast_dct=True)) against libjpeg-turbo's fast integer IDCT (JDCT_IFAST,
simplejpeg fastdct=True), decoded under each x86-64 code path in its own process: DEFAULT (what libjpeg-turbo
picks on this CPU), JSIMD_FORCEAVX2, JSIMD_FORCESSE2, JSIMD_FORCENONE.

usage: python tools/check_fastdct.py <work_dir> [--extreme N] [--dir D ...] [--no-gpu]

x86-64 libjpeg-turbo has one SIMD ifast (jidctfst-sse2.asm), used when the CPU has SSE2; JSIMD_FORCEAVX2 keeps
only the AVX2 flag, so ifast then runs the C version (jidctfst.c), as JSIMD_FORCENONE does. The two differ only
where a 16-bit intermediate wraps (see idct_ifast in _kernels.cu); the GPU declines those blocks.

Files: the synthetic corpus (tests/corpus.py), N crafted files with random, often extreme coefficients, and the
JPEGs in each --dir. Per group: whether the references agree, and whether the GPU output (if any) equals all of
them. The output holds counts only (and the names of failing files from the corpus and crafted groups).
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, HERE)

MODES = ("DEFAULT", "AVX2", "SSE2", "NONE")


def crafted(d, n):
    from jpegwriter import grid, write_jpeg
    rng = np.random.default_rng(11)
    out = []
    for k in range(n):
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
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--extreme", type=int, default=400)
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    import corpus
    from refdecode import digest, refs
    d = os.path.join(a.work_dir, "files")
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, data, _ in corpus.generate():
        p = os.path.join(d, name)
        open(p, "wb").write(data)
        paths[p] = "corpus"
    for p in crafted(d, a.extreme):
        paths[p] = "extreme"
    for k, dd in enumerate(a.dir):
        for p in sorted(glob.glob(os.path.join(dd, "*.jpg"))):
            paths[p] = f"dir{k}"
    R = refs(list(paths), a.work_dir, modes=MODES, fastdct=True)
    dec = None
    if not a.no_gpu:
        import exact_gpu_jpeg as egj
        dec = egj.Decoder(fast_dct=True)
    c = collections.Counter()
    fails, reasons = [], collections.Counter()
    for p, group in paths.items():
        r = {m: R[m][p] for m in MODES}
        if "error" in r.values():
            c[f"{group}: reference error"] += 1
            continue
        same = {m: r[m] == r["DEFAULT"] for m in MODES}
        agree = all(same.values())
        c[f"{group}: files"] += 1
        c[f"{group}: references agree" if agree else
          f"{group}: references differ (" + "/".join(m for m in MODES if not same[m]) + " != DEFAULT)"] += 1
        if dec is None:
            continue
        out = dec.decode(p)
        if out is None:
            c[f"{group}: GPU declined" + (" (references agree)" if agree else " (references differ)")] += 1
            reasons[dec.reason] += 1
            continue
        dg = digest(out.get())
        if all(dg == v for v in r.values()):
            c[f"{group}: GPU equals all four"] += 1
        else:
            c[f"{group}: GPU DIFFERS" + ("" if dg == r["DEFAULT"] else " from DEFAULT")] += 1
            if not group.startswith("dir"):
                fails.append(os.path.basename(p))
            else:
                fails.append(group)
    res = dict(counts=dict(sorted(c.items())), failures=fails, decline_reasons=dict(reasons), modes=list(MODES))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.work_dir, "check_fastdct.json"), "w"), indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
