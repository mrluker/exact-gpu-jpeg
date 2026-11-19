"""Parameter fuzzer for exact_gpu_jpeg.encode. Everything is generated locally.

usage: python tools/fuzz_encode.py <out.json> [--iters N] [--seed S]

Valid cases (random size 1..600 x 1..600, and one in 12 from 1,100 to 2,600 on a side so that the routes for more
than 16,384 and more than 65,536 blocks are taken; quality 1..100, 4:4:4 / 4:2:2 / 4:2:0, content: noise, gradients,
edges, flat, saturated; random ICC profile up to 200 kB and EXIF up to 64 kB; through encode() returning bytes,
encode(path=) writing a file, or Encoder.encode_async().result()): the bytes must equal Pillow's.
Invalid cases (wrong dtype / shape / layout, quality and subsampling out of range, oversized metadata, wrong
types): encode must raise ValueError or TypeError and leave the GPU usable (the next valid case still matches).
"""
import io
import json
import random
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image

import cupy as cp
import exact_gpu_jpeg as egj

SS = {"444": 0, "422": 1, "420": 2}


def pil(a, q, ss, icc, exif):
    kw = {}
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=SS[ss], **kw)
    return b.getvalue()


def image(r, rng):
    h, w = r.randint(1, 600), r.randint(1, 600)
    if r.random() < 1 / 12:                                      # the general routes (above 16,384 / 65,536 blocks)
        h, w = r.randint(1100, 2200), r.randint(1100, 2600)
    elif r.random() < 0.3:
        h, w = r.choice([(1, r.randint(1, 600)), (r.randint(1, 600), 1), (r.randint(1, 20), r.randint(1, 20))])
    k = r.randrange(5)
    y, x = np.mgrid[0:h, 0:w]
    if k == 0:
        a = rng.integers(0, 256, (h, w, 3))
    elif k == 1:
        a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) % 256], -1) + rng.normal(0, 10, (h, w, 3))
    elif k == 2:
        a = ((x // r.randint(1, 9) + y // r.randint(1, 9)) % 2 * 255)[..., None].repeat(3, -1)
    elif k == 3:
        a = np.full((h, w, 3), r.randrange(256))
    else:
        a = (rng.random((h, w, 3)) > 0.5) * 255
    return np.clip(a, 0, 255).astype(np.uint8)


def invalid(r):
    bad = r.randrange(9)
    a = np.zeros((8, 8, 3), np.uint8)
    kw = dict(quality=90, subsampling="444")
    if bad == 0:
        return np.zeros((8, 8), np.uint8), kw
    if bad == 1:
        return np.zeros((8, 8, r.choice([1, 2, 4])), np.uint8), kw
    if bad == 2:
        return np.zeros((8, 8, 3), r.choice([np.float32, np.uint16, np.int8])), kw
    if bad == 3:
        return cp.zeros((8, 16, 3), cp.uint8)[:, ::2], kw
    if bad == 4:
        return a, dict(kw, quality=r.choice([0, -5, 101, 1000, 50.5, None, "90"]))
    if bad == 5:
        return a, dict(kw, subsampling=r.choice(["411", "440", 3, -1, None, "444 "]))
    if bad == 6:
        return a, dict(kw, exif=bytes(r.randint(65534, 70000)))
    if bad == 7:
        return a, dict(kw, icc_profile=r.choice(["text", 5, 1.5]))
    return r.choice([None, [1, 2, 3], "image"]), kw


def main():
    out = sys.argv[1]
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 2000
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    r = random.Random(seed)
    rng = np.random.default_rng(seed)
    c = dict(valid=0, identical=0, different=0, invalid=0, raised_cleanly=0, not_raised=0, other_exception=0,
             by_route=dict(bytes=0, path=0, encode_async=0), over_16384_blocks=0, over_65536_blocks=0)
    work = tempfile.mkdtemp(prefix="egj-fuzz-encode-")
    enc = egj.Encoder()
    t0 = time.time()
    for it in range(iters):
        if r.random() < 0.8:
            a = image(r, rng)
            q, ss = r.randint(1, 100), r.choice(list(SS))
            icc = bytes(rng.integers(0, 256, r.randint(1, 200000), dtype=np.uint8)) if r.random() < 0.2 else None
            ex = b"Exif\x00\x00" + bytes(rng.integers(0, 256, r.randint(0, 65000), dtype=np.uint8)) if r.random() < 0.2 else None
            c["valid"] += 1
            sh, sv = {"444": (1, 1), "422": (2, 1), "420": (2, 2)}[ss]
            blocks = -(-a.shape[0] // (8 * sv)) * -(-a.shape[1] // (8 * sh)) * (sh * sv + 2)
            c["over_16384_blocks"] += blocks > 16384
            c["over_65536_blocks"] += blocks > 65536
            src = cp.asarray(a) if r.random() < 0.5 else a
            route = r.choices(["bytes", "path", "encode_async"], [0.7, 0.15, 0.15])[0]
            c["by_route"][route] += 1
            try:
                if route == "bytes":
                    got = egj.encode(src, q, ss, icc, ex)
                elif route == "path":
                    f = os.path.join(work, "out.jpg")
                    n = egj.encode(src, q, ss, icc, ex, path=f)
                    got = open(f, "rb").read()
                    assert n == len(got), (n, len(got))
                else:
                    got = enc.encode_async(src, q, ss, icc, ex).result()
                same = got == pil(a, q, ss, icc, ex)
            except Exception as e:                               # noqa: BLE001
                c["other_exception"] += 1
                print("exception on valid input", repr(e), a.shape, q, ss)
                continue
            c["identical" if same else "different"] += 1
            if not same:
                print("DIFFERENT", a.shape, q, ss, bool(icc), bool(ex))
        else:
            img, kw = invalid(r)
            c["invalid"] += 1
            try:
                egj.encode(img, **kw)
                c["not_raised"] += 1
                print("did not raise", type(img), getattr(img, "shape", None), kw.get("quality"), kw.get("subsampling"))
            except (ValueError, TypeError):
                c["raised_cleanly"] += 1
            except Exception as e:                               # noqa: BLE001
                c["other_exception"] += 1
                print("other exception", repr(e))
    cp.cuda.runtime.deviceSynchronize()
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    res = dict(iterations=iters, seed=seed, counts=c, seconds=round(time.time() - t0, 1))
    print(json.dumps(res))
    json.dump(res, open(out, "w"), indent=1)
    bad = c.get("different", 0) + c.get("not_raised", 0) + c.get("other_exception", 0)
    return 1 if bad or not c.get("identical") else 0


if __name__ == "__main__":
    sys.exit(main())
