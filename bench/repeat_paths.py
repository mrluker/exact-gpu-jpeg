"""A repeat decode end to end, through the public API, against a normal decode of the same file: is the repeat
path faster, and by how much? For every JPEG (from a path, RGB, uint8, the GPU's result waited for):

  normal            egj.decode(path)
  cache, light      egj.decode(path, cache=c, check="light")    (c holds a packed copy of the file)
  cache, full       egj.decode(path, cache=c, check="full")

Also, for each JPEG, the host cost of its identity alone (_repeat.check_value of its bytes): what the full check
pays besides reading the file.

Each path's pixels are compared with the normal decode's, byte for byte, before anything is timed. Modes take
turns inside each of 7 rounds of 5 calls; the result is the median of the rounds' medians, with the lowest and
highest round median. Files are read through the operating system's file cache (warm), as a program decoding the
same file again would find them.

usage: python bench/repeat_paths.py <out.json> [--photos DIR] [--public DIR]
  --photos DIR: up to 12 *.jpg files of DIR, reported as photo01, photo02, ... (no file names are recorded).
  --public DIR: the imagecompression.info images (the folder of their *.ppm files, as bench/public_set.py
    unpacks them), each made into a JPEG with libjpeg-turbo through Pillow at quality 95, 4:2:0, in a temporary
    folder outside the repository.
Run it alone (timing)."""
import glob
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

os.environ.setdefault("EGJ_SMALL_CPU_PIXELS", "200000")    # pinned: no background measurement during the run
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _repeat  # noqa: E402

ROUNDS, CALLS = 7, 5
pc, med = time.perf_counter, statistics.median


def arg(name):
    a = sys.argv[2:]
    return a[a.index(name) + 1] if name in a else None


def inputs(work):
    items = []
    d = arg("--photos")
    if d:
        files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())[:12]
        items += [(f"photo{k + 1:02d}", "photos", f) for k, f in enumerate(files)]
    d = arg("--public")
    if d:
        for p in sorted(glob.glob(os.path.join(d, "*.ppm"))):
            name = os.path.splitext(os.path.basename(p))[0]
            j = os.path.join(work, name + "_q95_420.jpg")
            Image.open(p).convert("RGB").save(j, "JPEG", quality=95, subsampling=2)
            items.append((name, "public", j))
    if not items:
        raise SystemExit("give --photos DIR and/or --public DIR")
    return items


def main():
    out_path = sys.argv[1]
    egj.self_check()
    dev = cp.cuda.Device()
    work = tempfile.mkdtemp(prefix="egj-repeat-paths-")
    res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], version=egj.__version__,
               rounds=ROUNDS, calls_a_round=CALLS, unit="ms a call", rows=[])
    try:
        dec = egj.Decoder()
        for label, kind, path in inputs(work):
            data = open(path, "rb").read()
            hd = egj.parse(np.frombuffer(data, np.uint8))
            ref = egj.decode(path)
            dev.synchronize()
            ref = ref.get()
            caches = {"light": egj.RepeatCache(), "full": egj.RepeatCache()}
            for chk, c in caches.items():
                egj.decode(path, cache=c, check=chk)                 # the first decode keeps the copy
                dev.synchronize()

            def normal():
                return egj.decode(path)

            def cached(chk):
                c = caches[chk]
                return lambda: egj.decode(path, cache=c, check=chk)

            modes = {"normal": normal, "cache, light": cached("light"), "cache, full": cached("full")}
            for name, f in modes.items():                            # the same pixels first
                used = {k: c.stats["used"] for k, c in caches.items()}
                got = f()
                dev.synchronize()
                assert got is not None, (label, name, dec.reason)
                assert np.array_equal(got.get(), ref), (label, name)
                if name.startswith("cache"):
                    k = name.split(", ")[1]
                    assert caches[k].stats["used"] == used[k] + 1, (label, name, "the copy was not used")
            rounds = {n: [] for n in modes}
            for r in range(ROUNDS):
                for name, f in modes.items():
                    t = []
                    for _ in range(CALLS):
                        dev.synchronize()
                        t0 = pc()
                        f()
                        dev.synchronize()
                        t.append((pc() - t0) * 1e3)
                    rounds[name].append(med(t))
            ids = {}
            t = []
            for _ in range(ROUNDS * CALLS):
                t0 = pc()
                _repeat.check_value(data)
                t.append((pc() - t0) * 1e3)
            ids["jpeg identity"] = round(med(t), 2)
            row = dict(image=label, set=kind, mp=round(hd.width * hd.height / 1e6, 2), jpeg_mb=round(len(data) / 1e6, 2),
                       ms={n: dict(median=round(med(v), 2), low=round(min(v), 2), high=round(max(v), 2))
                           for n, v in rounds.items()},
                       identities_ms=ids)
            n0 = row["ms"]["normal"]["median"]
            row["faster_than_normal"] = {n: row["ms"][n]["median"] < n0 for n in modes if n != "normal"}
            res["rows"].append(row)
            print(label, row["mp"], "MP |", " | ".join(f"{n} {row['ms'][n]['median']}" for n in modes),
                  "| ids", ids, flush=True)
            for c in caches.values():
                c.clear()
            with open(out_path, "w") as f:
                json.dump(res, f, indent=1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
