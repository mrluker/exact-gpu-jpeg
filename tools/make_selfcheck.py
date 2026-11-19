"""Write the two data modules of the package: src/exact_gpu_jpeg/_selfcheck_data.py, the built-in cases of the
start-up self-check and their hashes, and src/exact_gpu_jpeg/_route_data.py, three small JPEGs for timing the CPU
route against the GPU route. Every image in both is generated here from formulas and seeded random numbers; no
photograph is embedded in the package.

usage: python tools/make_selfcheck.py [--check]

No GPU is used. The JPEGs are encoded by Pillow (libjpeg-turbo) from the self-check's synthetic image
(exact_gpu_jpeg._selfcheck.synthetic) at quality 90 with the standard Huffman tables; each one's hash is the
SHA-256 of the pixels libjpeg-turbo decodes from it (simplejpeg, accurate IDCT, fancy upsampling), and Pillow must
decode the same pixels or the script stops. The encode cases store the SHA-256 of the file Pillow writes. The output
is deterministic for given Pillow and simplejpeg versions (they are written into the file); --check compares with
the committed files instead of writing them (tests/test_selfcheck.py does the same).

The timing JPEGs are photo_like() images (the generator of bench/reproduce.py: smooth colour fields, shapes, lines
and grain, about 1.2 bits per pixel at quality 90, as a resized camera photo has), 4:2:0, at about 0.06, 0.12 and
0.25 MP, seeds 1 to 3.
"""
import base64
import hashlib
import io
import os
import sys

import numpy as np
import PIL
import simplejpeg
from PIL import Image, ImageDraw, features

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from exact_gpu_jpeg._selfcheck import synthetic  # noqa: E402

OUT = os.path.join(ROOT, "src", "exact_gpu_jpeg", "_selfcheck_data.py")
OUT_ROUTE = os.path.join(ROOT, "src", "exact_gpu_jpeg", "_route_data.py")
ROUTE = [("0.06 MP", 304, 200, 1), ("0.12 MP", 424, 280, 2), ("0.25 MP", 616, 408, 3)]      # name, width, height, seed
# name, width, height, Pillow subsampling (None: gray), restart interval in MCU rows, colorspace of the decode
DECODE = [
    ("decode 4:2:0 with restart markers", 97, 75, 2, 1, "RGB"),
    ("decode 4:2:0 without restarts", 160, 120, 2, 0, "RGB"),
    ("decode 4:2:2", 80, 56, 1, 0, "RGB"),
    ("decode 4:4:4", 64, 48, 0, 0, "RGB"),
    ("decode gray", 64, 48, None, 0, "GRAY"),
]
BATCH = ("decode batch of two", [(96, 72, 2), (80, 64, 2)])       # two files that share tables and sampling
# The checks of the decode modes, each run when the mode is first asked for: (part, case, width, height, Pillow
# subsampling, decode options). Each case is decoded once alone and once in a batch of two with the same options.
# The expected pixels are libjpeg-turbo's in that mode (simplejpeg), and for float output NumPy's float32 steps on
# libjpeg-turbo's pixels (scale and bias rounded to float32, a multiply and an add, float16 rounding to nearest even).
MODES = [
    ("fast_dct", "fast IDCT, 4:2:0", 160, 120, 2, dict(fast_dct=True)),
    ("fast_dct", "fast IDCT, 4:4:4", 64, 48, 0, dict(fast_dct=True)),
    ("fast_upsample", "fast upsampling, 4:2:0", 160, 120, 2, dict(fast_upsample=True)),
    ("fast_upsample", "fast upsampling, 4:2:2", 80, 56, 1, dict(fast_upsample=True)),
    ("output", "float32 CHW, scale and bias", 97, 75, 2,
     dict(dtype="float32", layout="CHW", scale=[0.0172, 0.0175, 0.0174], bias=[-2.12, -2.04, -1.80])),
    ("output", "float16 HWC, BGR", 80, 56, 1, dict(colorspace="BGR", dtype="float16", scale=1 / 255)),
    ("output", "uint8 BGR", 64, 48, 0, dict(colorspace="BGR")),
    ("output", "float32 GRAY", 64, 48, 2, dict(colorspace="GRAY", dtype="float32", scale=1 / 255)),
    ("output", "float32 RGB of a grayscale file", 64, 48, None, dict(dtype="float32", scale=1 / 255, bias=-0.5)),
    ("output", "float32 4:4:4", 72, 40, 0, dict(dtype="float32")),
]
# name, width, height, quality, subsampling: the few-launch route (small images) and the general route
ENCODE = [("encode few-launch", 64, 48, 90, "420"), ("encode general", 600, 592, 75, "444")]


def photo_like(width, height, seed=0, grain=2.0):
    """bench/reproduce.py's synthetic_photo, line for line (tests/test_selfcheck.py checks they agree)."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 127.5, np.float32)
    for cell, amp in ((512, 80.0), (160, 45.0), (48, 22.0), (14, 9.0)):
        gh, gw = height // cell + 2, width // cell + 2
        g = (rng.random((gh, gw, 3)) * 255).astype(np.uint8)
        layer = Image.fromarray(g).resize((gw * cell, gh * cell), Image.BICUBIC).crop((0, 0, width, height))
        img += (np.asarray(layer, np.float32) - 127.5) * (amp / 127.5)
    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(base)
    n = max(width * height // 60_000, 2)
    for _ in range(n):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(8, 160))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x - r, y - r // 2, x + r, y + r // 2], fill=col)
    for _ in range(2 * n):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x1, y1 = x0 + int(rng.integers(-200, 200)), y0 + int(rng.integers(-200, 200))
        d.line([x0, y0, x1, y1], fill=tuple(int(v) for v in rng.integers(0, 256, 3)), width=int(rng.integers(1, 4)))
    a = np.asarray(base, np.float32)
    a += rng.normal(0.0, grain, a.shape).astype(np.float32)
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def build_route():
    lines = ['"""Three small JPEGs for timing the CPU route against the GPU route (_route.py). Written by',
             "tools/make_selfcheck.py: do not edit. Generated images (formulas and seeded random numbers, see photo_like",
             f"there), not photographs; encoded by Pillow {PIL.__version__} at quality 90, 4:2:0.\"\"\"",
             "", "# (name, the JPEG in base64)", "FILES = ["]
    for name, w, h, seed in ROUTE:
        b = io.BytesIO()
        Image.fromarray(photo_like(w, h, seed)).save(b, "JPEG", quality=90, subsampling=2)
        lines.append(f'    ("{name}",\n{b64(b.getvalue())}),')
    lines += ["]", ""]
    return "\n".join(lines)


def jpeg(w, h, ss, rows=0):
    a = synthetic(w, h, gray=ss is None)
    b = io.BytesIO()
    kw = dict(restart_marker_rows=rows) if rows else {}
    if ss is None:
        Image.fromarray(a, "L").save(b, "JPEG", quality=90, **kw)
    else:
        Image.fromarray(a).save(b, "JPEG", quality=90, subsampling=ss, **kw)
    return b.getvalue()


def pixels(data, cs):
    a = simplejpeg.decode_jpeg(data, colorspace=cs, fastdct=False, fastupsample=False)
    p = np.asarray(Image.open(io.BytesIO(data)).convert("L" if cs == "GRAY" else "RGB"))
    if cs == "GRAY":
        p = p[..., None]
    if not np.array_equal(a, p):
        sys.exit(f"simplejpeg and Pillow decode different pixels ({cs}, {len(data)} bytes): not a usable case")
    return a


def mode_pixels(data, opts):
    """libjpeg-turbo's pixels in the mode, with the output options applied as the library defines them."""
    cs = opts.get("colorspace", "RGB")
    a = simplejpeg.decode_jpeg(data, colorspace=cs, fastdct=opts.get("fast_dct", False),
                               fastupsample=opts.get("fast_upsample", False))
    dt = opts.get("dtype", "uint8")
    if dt == "uint8":
        return a
    s = np.float32(1.0) if opts.get("scale") is None else np.asarray(opts["scale"], np.float32)
    b = np.float32(0.0) if opts.get("bias") is None else np.asarray(opts["bias"], np.float32)
    v = (a.astype(np.float32) * s).astype(np.float32)
    v = (v + b).astype(np.float32)
    if dt == "float16":
        v = v.astype(np.float16)
    if opts.get("layout") == "CHW":
        v = np.ascontiguousarray(v.transpose(2, 0, 1))
    return v


def sha(b):
    return hashlib.sha256(b).hexdigest()


def b64(data):
    s = base64.b64encode(data).decode()
    return "\n".join('     "' + s[i:i + 100] + '"' for i in range(0, len(s), 100))


def build():
    lines = ['"""The start-up self-check\'s cases. Written by tools/make_selfcheck.py: do not edit.',
             "",
             f"JPEGs encoded by Pillow {PIL.__version__} (libjpeg-turbo {features.version_feature('libjpeg_turbo')}), "
             f"quality 90, from _selfcheck.synthetic;",
             f"pixel hashes from simplejpeg {simplejpeg.__version__}'s decode (accurate IDCT, fancy upsampling), "
             "equal to Pillow's.",
             'Encode hashes: the file Pillow writes for _selfcheck.synthetic(width, height)."""',
             "", "# (case, colorspace, SHA-256 of the decoded pixels, the JPEG in base64)", "DECODE = ["]
    for name, w, h, ss, rows, cs in DECODE:
        d = jpeg(w, h, ss, rows)
        lines.append(f'    ("{name}", "{cs}", "{sha(pixels(d, cs).tobytes())}",\n{b64(d)}),')
    lines += ["]", "", "# (case, [(SHA-256 of the decoded pixels, the JPEG in base64), ...]): decoded by one decode_batch call",
              f'BATCH = ("{BATCH[0]}", [']
    for w, h, ss in BATCH[1]:
        d = jpeg(w, h, ss)
        lines.append(f'    ("{sha(pixels(d, "RGB").tobytes())}",\n{b64(d)}),')
    lines += ["])", "", "# (part, case, decode options, SHA-256 of the output, the JPEG in base64): the decode modes' checks",
              "MODES = ["]
    for part, name, w, h, ss, opts in MODES:
        d = jpeg(w, h, ss)
        lines.append(f'    ("{part}", "{name}", {opts!r}, "{sha(mode_pixels(d, opts).tobytes())}",\n{b64(d)}),')
    lines += ["]", "", "# (case, width, height, quality, subsampling, SHA-256 of the encoded file)", "ENCODE = ["]
    for name, w, h, q, ss in ENCODE:
        b = io.BytesIO()
        Image.fromarray(synthetic(w, h)).save(b, "JPEG", quality=q, subsampling={"444": 0, "422": 1, "420": 2}[ss])
        lines.append(f'    ("{name}", {w}, {h}, {q}, "{ss}", "{sha(b.getvalue())}"),')
    lines += ["]", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    todo = [(OUT, build()), (OUT_ROUTE, build_route())]
    if "--check" in sys.argv:
        same = all(os.path.exists(f) and open(f, encoding="utf-8").read() == text for f, text in todo)
        print("up to date" if same else "STALE: run tools/make_selfcheck.py")
        sys.exit(0 if same else 1)
    for f, text in todo:
        open(f, "w", encoding="utf-8", newline="\n").write(text)
        print(f"wrote {f} ({len(text)} characters)")
