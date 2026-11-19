"""Byte-exactness of exact_gpu_jpeg.encode against Pillow on your own photos (counts only, no file names).

usage: python tools/check_encode.py <dir of JPEGs> [--json OUT]

Each photo is decoded, resized like three typical web exports (3840x2560 q95 4:4:4, 1500x1000 q90 4:4:4,
1200x800 q87 4:2:0, keeping the aspect ratio), and encoded with Pillow and with the GPU, with the photo's own ICC
profile and EXIF. Also the full-size photo at q90 in 4:4:4, 4:2:2 and 4:2:0.
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import exact_gpu_jpeg as egj  # noqa: E402

SETTINGS = [(3840, 95, "444"), (1500, 90, "444"), (1200, 87, "420")]


def pil(a, q, ss, icc, exif):
    kw = {}
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling={"444": 0, "422": 1, "420": 2}[ss], **kw)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(a.dir, "*.jpg"))}.values())
    res = {}
    for f in files:
        with Image.open(f) as im:
            icc, exif = im.info.get("icc_profile"), im.info.get("exif")
            rgb = im.convert("RGB")
        for w, q, ss in SETTINGS + [(None, 90, "444"), (None, 90, "422"), (None, 90, "420")]:
            im2 = rgb if w is None else rgb.resize((w, round(rgb.height * w / rgb.width)), Image.LANCZOS)
            arr = np.asarray(im2)
            key = f"{'full' if w is None else w}_q{q}_{ss}"
            ok = egj.encode(arr, q, ss, icc, exif) == pil(arr, q, ss, icc, exif)
            r = res.setdefault(key, dict(n=0, identical=0))
            r["n"] += 1
            r["identical"] += ok
        print(os.path.basename(os.path.dirname(f)) and "", end="", flush=True)
    meta = dict(with_icc=sum(1 for f in files if Image.open(f).info.get("icc_profile")),
                with_exif=sum(1 for f in files if Image.open(f).info.get("exif")))
    out = dict(files=len(files), metadata=meta, results=res)
    print(json.dumps(out))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    differ = sum(r["n"] - r["identical"] for r in res.values())
    return 1 if differ or not files else 0                     # (no file found: not a pass)


if __name__ == "__main__":
    sys.exit(main())
