"""Deterministic synthetic test corpus: JPEGs written by libjpeg-turbo (through simplejpeg and Pillow) and by
tests/jpegwriter.py, across subsamplings, qualities, sizes, optimised tables, restart intervals, progressive
mode, colour-space markers and table layouts. No photos: everything is generated from seeded noise and shapes,
so the corpus can be regenerated anywhere and has no licence questions.

    python tests/corpus.py <out_dir>        # writes the files and index.json
    corpus.generate() -> list of (name, bytes, gpu_expected)

gpu_expected: True if the GPU path must decode the file (and exactly), False if it must decline (return None).
"""
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegwriter import grid, write_jpeg  # noqa: E402

SIZES = [(1, 1), (1, 17), (17, 1), (2, 3), (7, 9), (8, 8), (9, 2), (16, 16), (15, 33), (31, 64), (97, 203),
         (480, 640), (1080, 1920)]
QUALITIES = [1, 5, 25, 50, 75, 90, 95, 100]


def content(rng, h, w, kind):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "smooth":          # gradients plus low-pass noise: photo-like
        a = np.stack([x * 255 / max(w - 1, 1), y * 255 / max(h - 1, 1), 128 + 100 * np.sin((x + y) / 23)], -1)
        n = rng.normal(0, 40, (h // 4 + 2, w // 4 + 2, 3))
        a += np.kron(n, np.ones((4, 4, 1)))[:h, :w]
    elif kind == "edges":         # hard edges and thin lines: ringing, overshoot
        a = np.zeros((h, w, 3), np.float32)
        a[(x.astype(int) // 5 + y.astype(int) // 7) % 2 == 1] = (255, 255, 255)
        a[:, ::9] = (255, 0, 0)
        a[::11, :] = (0, 0, 255)
    else:                         # white noise: worst case for Huffman
        a = rng.integers(0, 256, (h, w, 3)).astype(np.float32)
    return np.clip(a, 0, 255).astype(np.uint8)


def generate():
    import simplejpeg
    from PIL import Image
    rng = np.random.default_rng(20261113)
    out = []

    def pil(name, a, expected=True, **kw):
        b = io.BytesIO()
        Image.fromarray(a).save(b, "JPEG", **kw)
        out.append((name, b.getvalue(), expected))

    # libjpeg-turbo (simplejpeg 1.9 bundles 3.1.2) across subsamplings, qualities and sizes
    subs = ["444", "422", "420", "440", "411", "Gray"]
    for si, (h, w) in enumerate(SIZES):
        kinds = ["smooth", "edges", "noise"] if h * w < 300000 else ["smooth"]
        for kind in kinds:
            a = content(rng, h, w, kind)
            for k, sub in enumerate(subs):
                q = QUALITIES[(si + k + len(kind)) % len(QUALITIES)]
                src = a[..., :1].copy() if sub == "Gray" else a
                cs = "GRAY" if sub == "Gray" else "RGB"
                out.append((f"sj_{h}x{w}_{kind}_{sub}_q{q}.jpg",
                            simplejpeg.encode_jpeg(src, quality=q, colorspace=cs, colorsubsampling=sub), True))
    # Pillow (libjpeg-turbo 3.2.0 in Pillow 13): optimised tables, restart intervals, progressive
    for (h, w) in [(1, 1), (9, 2), (16, 16), (15, 33), (97, 203), (480, 640)]:
        a = content(rng, h, w, "smooth")
        for sub in (0, 1, 2):
            pil(f"pil_{h}x{w}_s{sub}_opt.jpg", a, quality=85, subsampling=sub, optimize=True)
            pil(f"pil_{h}x{w}_s{sub}_rst1.jpg", a, quality=85, subsampling=sub, restart_marker_blocks=1)
            pil(f"pil_{h}x{w}_s{sub}_rst7.jpg", a, quality=60, subsampling=sub, restart_marker_blocks=7)
            pil(f"pil_{h}x{w}_s{sub}_rstrow.jpg", a, quality=95, subsampling=sub, restart_marker_rows=1, optimize=True)
            pil(f"pil_{h}x{w}_s{sub}_prog.jpg", a, False, quality=85, subsampling=sub, progressive=True)
        pil(f"pil_{h}x{w}_gray_rst3.jpg", a[..., 0].copy(), quality=80, restart_marker_blocks=3)
        pil(f"pil_{h}x{w}_gray_prog.jpg", a[..., 0].copy(), False, quality=80, progressive=True)

    # crafted with jpegwriter: unusual (valid) tables and markers
    for k in range(24):
        samp = [[(1, 1)], [(1, 1)] * 3, [(2, 2), (1, 1), (1, 1)], [(2, 1), (1, 1), (1, 1)], [(1, 2), (1, 1), (1, 1)],
                [(4, 1), (1, 1), (1, 1)], [(2, 2), (2, 1), (2, 1)], [(1, 1), (1, 1), (1, 1)]][k % 8]
        W, H = int(rng.integers(1, 70)), int(rng.integers(1, 70))
        blocks = []
        for ci in range(len(samp)):
            g = grid(samp, W, H, ci)
            b = (rng.integers(-6, 7, g + (64,)) * (rng.random(g + (64,)) < 0.15)).astype(np.int32)
            b[..., 0] = rng.integers(-60, 61, g)
            blocks.append(b)
        quant = [rng.integers(1, 17, 64) for _ in samp]   # pixels stay near [0, 255], as with a real encoder
        dri = [0, 1, 2, 5][k % 4]
        out.append((f"craft_{k}_dri{dri}.jpg", write_jpeg(blocks, quant, samp, W, H, dri=dri), True))
    # colour-space markers: no JFIF with ids 1,2,3 (YCbCr), Adobe transform 1 (YCbCr) -> GPU; RGB-coded -> CPU
    a = content(rng, 40, 56, "smooth")
    base = simplejpeg.encode_jpeg(a, quality=80, colorsubsampling="420")
    nojfif = _drop_app0(base)
    out.append(("mk_nojfif_ids123.jpg", nojfif, True))
    out.append(("mk_adobe1.jpg", _insert_after_soi(nojfif, _adobe(1)), True))
    out.append(("mk_adobe0_rgb.jpg", _insert_after_soi(nojfif, _adobe(0)), False))
    out.append(("mk_jfif_and_adobe0.jpg", _insert_after_soi(base, _adobe(0)), True))     # JFIF wins: YCbCr
    out.append(("mk_rgb_ids.jpg", _set_ids(nojfif, (82, 71, 66)), False))
    out.append(("mk_no_dht_std_tables.jpg", _drop_dht(simplejpeg.encode_jpeg(a, quality=75, colorsubsampling="422")),
                True))
    out.append(("mk_comment_after_scan.jpg", base[:-2] + b"\xff\xfe\x00\x05abc" + b"\xff\xd9", True))
    out.append(("mk_no_eoi.jpg", base[:-2], False))     # libjpeg-turbo warns (no end marker): declined by default
    # one colour with optimised tables: every block is the same 2 bits, and such a stream does not synchronise by
    # itself. At these sizes the quick passes do not settle, so the decode is done a second time (with host
    # checks) in every route, with no setting changed. The names begin with "redo_": tests that need this kind
    # pick them by it, and assert that the second decode really happened. In a batch every image starts from a
    # known state and is followed by one, which can save a pass: the first settles there; the third (more than
    # three CUDA blocks of chunks) cannot, and its batch goes one by one.
    for h, w, sub in ((2000, 3000, 2), (1536, 2304, 0), (1800, 2800, 0)):
        a = np.empty((h, w, 3), np.uint8)
        a[:] = (200, 120, 40)
        pil(f"redo_flat_{h}x{w}_s{sub}_opt.jpg", a, quality=90, subsampling=sub, optimize=True)
    return out


def _segments(d):
    i, segs = 2, []
    while d[i] == 0xFF and d[i + 1] != 0xDA:
        L = int.from_bytes(d[i + 2:i + 4], "big")
        segs.append((i, d[i + 1], L))
        i += 2 + L
    return segs


def _drop_app0(d):
    for i, m, L in _segments(d):
        if m == 0xE0:
            return d[:i] + d[i + 2 + L:]
    return d


def _drop_dht(d):
    out = d
    for i, m, L in reversed(_segments(d)):
        if m == 0xC4:
            out = out[:i] + out[i + 2 + L:]
    return out


def _adobe(transform):
    payload = b"Adobe" + b"\x00\x64" + b"\x00\x00" + b"\x00\x00" + bytes([transform])
    return b"\xff\xee" + (len(payload) + 2).to_bytes(2, "big") + payload


def _insert_after_soi(d, seg):
    return d[:2] + seg + d[2:]


def _set_ids(d, ids):
    d = bytearray(d)
    for i, m, L in _segments(bytes(d)):
        if m in (0xC0, 0xC1):
            for k in range(3):
                d[i + 4 + 6 + 3 * k] = ids[k]
    j = bytes(d).find(b"\xff\xda")
    for k in range(3):
        d[j + 5 + 2 * k] = ids[k]
    return bytes(d)


if __name__ == "__main__":
    od = sys.argv[1]
    os.makedirs(od, exist_ok=True)
    idx = []
    for name, data, exp in generate():
        with open(os.path.join(od, name), "wb") as f:
            f.write(data)
        idx.append(dict(name=name, bytes=len(data), gpu_expected=exp))
    json.dump(idx, open(os.path.join(od, "index.json"), "w"), indent=0)
    print(len(idx), "files")
