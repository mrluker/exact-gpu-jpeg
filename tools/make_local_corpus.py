"""Realistic test and benchmark JPEGs made locally from your own photos: nothing is downloaded.

usage: python tools/make_local_corpus.py <dir of source JPEGs> <out_dir> [--max-sources N]

Each source photo is decoded once (libjpeg-turbo) and re-encoded with libjpeg-turbo (Pillow and simplejpeg) as:
  - 4:4:4, 4:2:2, 4:2:0 at qualities 50 / 75 / 90 / 95 (full size), plus 4:4:0 and 4:1:1 (simplejpeg),
  - optimised Huffman tables, restart intervals (every MCU, every 7 MCUs, every MCU row), progressive,
  - grayscale,
  - odd sizes: crops of 1×1, 7×13, 333×501, 1001×1499 pixels and a 1/3-scale resize,
and sized/ holds one source at about 2, 6, 12 and 24 MP in 4:4:4, 4:2:2, 4:2:0 and 4:2:0 with restart markers
(for benchmarks). Output files are named by index, not by source name. Keep <out_dir> out of git: it shows your
photos. index.json lists each file with its settings and whether the GPU path is expected to decode it.
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import simplejpeg
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--max-sources", type=int, default=12)
    a = ap.parse_args()
    srcs = sorted(glob.glob(os.path.join(a.src, "*.jpg")) + glob.glob(os.path.join(a.src, "*.JPG")))
    seen, u = set(), []
    for f in srcs:
        k = os.path.normcase(os.path.abspath(f))
        if k not in seen:
            seen.add(k)
            u.append(f)
    srcs = u[:a.max_sources]
    os.makedirs(os.path.join(a.out, "files"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "sized"), exist_ok=True)
    index = []

    def put(data, sub, **info):
        name = f"{len(index):05d}.jpg"
        with open(os.path.join(a.out, "files", name), "wb") as f:
            f.write(data)
        index.append(dict(file=name, **info, gpu_expected=not info.get("progressive", False)))

    def pil(img, **kw):
        b = io.BytesIO()
        img.save(b, "JPEG", **kw)
        return b.getvalue()

    for si, f in enumerate(srcs):
        arr = simplejpeg.decode_jpeg(open(f, "rb").read(), colorspace="RGB")
        im = Image.fromarray(arr)
        for sub, code in (("4:4:4", 0), ("4:2:2", 1), ("4:2:0", 2)):
            for q in (50, 75, 90, 95)[si % 2::2]:          # two qualities per source, alternating
                put(pil(im, quality=q, subsampling=code), sub, source=si, quality=q, subsampling=sub)
        sub_k = ("4:4:4", "4:2:2", "4:2:0")[si % 3]
        code = si % 3
        put(pil(im, quality=85, subsampling=code, optimize=True), sub_k, source=si, optimize=True, subsampling=sub_k)
        put(pil(im, quality=85, subsampling=code, restart_marker_blocks=1), sub_k, source=si, restart="1 MCU", subsampling=sub_k)
        put(pil(im, quality=85, subsampling=code, restart_marker_blocks=7), sub_k, source=si, restart="7 MCUs", subsampling=sub_k)
        put(pil(im, quality=85, subsampling=code, restart_marker_rows=1, optimize=True), sub_k, source=si,
            restart="1 row", optimize=True, subsampling=sub_k)
        put(pil(im, quality=85, subsampling=code, progressive=True), sub_k, source=si, progressive=True, subsampling=sub_k)
        put(pil(im.convert("L"), quality=85), "gray", source=si, subsampling="gray")
        for sj_sub in ("440", "411"):
            put(simplejpeg.encode_jpeg(arr, quality=85, colorsubsampling=sj_sub), sj_sub, source=si,
                subsampling={"440": "4:4:0", "411": "4:1:1"}[sj_sub])
        h, w = arr.shape[:2]
        for ch, cw in ((1, 1), (7, 13), (333, 501), (1001, 1499)):
            y0, x0 = (h - ch) // 3, (w - cw) // 3
            crop = np.ascontiguousarray(arr[y0:y0 + ch, x0:x0 + cw])
            put(pil(Image.fromarray(crop), quality=90, subsampling=si % 3), "crop", source=si, size=[ch, cw],
                subsampling=("4:4:4", "4:2:2", "4:2:0")[si % 3])
        small = im.resize((w // 3, h // 3), Image.LANCZOS)
        put(pil(small, quality=92, subsampling=2), "4:2:0", source=si, size=[h // 3, w // 3], subsampling="4:2:0")
        if si == 0:
            for mp in (2, 6, 12, 24):
                s = (mp * 1e6 / (w * h)) ** 0.5
                sw, sh = int(round(w * s / 16)) * 16, int(round(h * s / 16)) * 16
                b = np.asarray(im.resize((sw, sh), Image.LANCZOS))
                for sub in ("444", "422", "420"):
                    with open(os.path.join(a.out, "sized", f"{mp:02d}mp_{sub}.jpg"), "wb") as fo:
                        fo.write(simplejpeg.encode_jpeg(b, quality=90, colorsubsampling=sub))
                with open(os.path.join(a.out, "sized", f"{mp:02d}mp_420rst.jpg"), "wb") as fo:
                    fo.write(pil(Image.fromarray(b), quality=90, subsampling=2, restart_marker_rows=1))
        print(f"source {si + 1}/{len(srcs)}: {len(index)} files", flush=True)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
