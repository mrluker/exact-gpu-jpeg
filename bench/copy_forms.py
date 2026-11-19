"""What each form of a kept copy costs and saves, at the handle level. For generated photo-like JPEGs of 0.1 to
24 MP, from bytes, one Decoder, sustained: a normal decode; a decode that also keeps a packed or a raw copy; a decode
from a packed handle and from a raw one. Also the GPU memory each form holds.
Modes take turns inside each of 7 rounds; medians of the rounds' medians, with the spread.
usage: python bench/copy_forms.py <out.json>     timings only; generated images, nothing is written but the result."""
import io
import json
import os
import statistics
import sys
import time

os.environ.setdefault("EGJ_SMALL_CPU_PIXELS", "200000")    # pinned: no background measurement during the run
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from reproduce import synthetic_photo  # noqa: E402

out = sys.argv[1]
pc, med = time.perf_counter, statistics.median
egj.self_check()
print("library:", egj.__file__, flush=True)
SIZES = [(400, 250), (1200, 800), (2400, 1600), (3600, 2400), (6000, 4000)]
res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], rows=[])
dec = egj.Decoder()
sync = dec.stream.synchronize


def keeping(form, data):
    def f():
        dec.keep_coefs = form
        img = dec.decode(data)
        h = dec.take_coefs()
        dec.keep_coefs = None
        return img, h
    return f


print("| image | JPEG | packed copy | raw copy | decode | + keep packed | + keep raw | from packed | from raw |\n"
      "|---|---|---|---|---|---|---|---|---|")
for k, (w, h) in enumerate(SIZES):
    b = io.BytesIO()
    Image.fromarray(synthetic_photo(w, h, k + 1)).save(b, "JPEG", quality=90)
    data = b.getvalue()
    img, packed = keeping("packed", data)()
    img, raw = keeping("raw", data)()
    assert (packed.form, raw.form) == ("packed", "raw")
    sizes = dict(jpeg=len(data), packed=packed.nbytes, raw=raw.nbytes)
    del img

    modes = {"decode": lambda: (dec.decode(data), None),
             "decode, keep packed": keeping("packed", data),
             "decode, keep raw": keeping("raw", data),
             "from a packed handle": lambda: (dec.decode_from_handle(packed), None),
             "from a raw handle": lambda: (dec.decode_from_handle(raw), None)}
    n = max(8, min(200, int(3e6 / (w * h)) * 10))
    for f in modes.values():                                   # everything warm
        for _ in range(4):
            img, hd = f()
            sync()
            if hd is not None:
                hd.free()
    rounds = {m: [] for m in modes}
    for r in range(7):
        order = list(modes)[r % len(modes):] + list(modes)[:r % len(modes)]
        for m in order:
            f = modes[m]
            ts = []
            for _ in range(n):
                t = pc()
                img, hd = f()
                sync()
                ts.append((pc() - t) * 1e3)
                if hd is not None:
                    hd.free()
                del img
            rounds[m].append(med(ts))
    row = dict(width=w, height=h, mp=round(w * h / 1e6, 2), bytes=sizes, calls_per_round=n,
               ms={m: round(med(v), 4) for m, v in rounds.items()},
               spread_ms={m: [round(min(v), 4), round(max(v), 4)] for m, v in rounds.items()})
    res["rows"].append(row)
    ms = row["ms"]
    print(f"| {w}x{h} ({row['mp']} MP) | {len(data) / 1e6:.2f} MB | {sizes['packed'] / 1e6:.2f} MB | {sizes['raw'] / 1e6:.2f} MB | "
          + " | ".join(f"{ms[m]:.3f}" for m in modes) + " |", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    packed.free()
    raw.free()
