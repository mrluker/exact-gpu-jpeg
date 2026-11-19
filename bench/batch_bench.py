"""Small-image decode throughput: decode_batch against decode() one image at a time, libjpeg-turbo on CPU threads
and nvJPEG's batched call (torchvision, if torch is installed).

usage: python bench/batch_bench.py <out.json> <dir of photos> [--sizes 0.25,0.5,1,2,6] [--n 64] [--vary]

--vary: every image of a set has its own size (widths 8 px apart around the nominal one), as in a dataset of
all-different files.

The photos are only read: each is resized in memory (Lanczos) to every size (megapixels, 3:2) and encoded by Pillow
(libjpeg-turbo) at quality 90, 4:2:0, so every set holds n different JPEGs of one size (the photos repeated if
there are fewer than n). Every decoder starts from the JPEG bytes in memory and ends with the decoded images
(GPU decoders: in GPU memory, synchronised). Images per second over repeats of the set for at least 2 s.
Results hold counts and timings only (no file names).
"""
import glob
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simplejpeg
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402

MIN_S = 2.0


def rate(fn, n):
    """images/s: fn() decodes n images; repeated for at least MIN_S (and twice)."""
    fn()
    t0, k = time.perf_counter(), 0
    while k < 2 or time.perf_counter() - t0 < MIN_S:
        fn()
        k += 1
    return k * n / (time.perf_counter() - t0)


def main():
    out, d = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    sizes = [float(x) for x in args[args.index("--sizes") + 1].split(",")] if "--sizes" in args else [0.25, 0.5, 1, 2, 6]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 64
    vary = "--vary" in args
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())
    photos = [Image.open(f).convert("RGB") for f in files]
    try:
        import torch
        from torchvision.io import decode_jpeg
    except ImportError:
        torch = None
    dec = egj.Decoder()
    res = dict(env=dict(gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                        cpu=platform.processor(), cpu_threads=os.cpu_count(), python=platform.python_version(),
                        cupy=cp.__version__, simplejpeg=simplejpeg.__version__,
                        torch=torch.__version__ if torch else None, exact_gpu_jpeg=egj.__version__),
               photos=len(photos), images_per_set=n, varied_sizes=vary, sets={})
    for mp in sizes:
        w = int(round((mp * 1e6 * 1.5) ** 0.5 / 8)) * 8
        h = int(round(w / 1.5 / 8)) * 8
        data = []
        for k in range(n):
            im = photos[k % len(photos)]
            if k >= len(photos):                          # a different crop for each repeat of a photo
                s = 1 - 0.04 * (k // len(photos))
                im = im.crop((0, 0, int(im.width * s), int(im.height * s)))
            b = io.BytesIO()
            wk = w + 8 * (k - n // 2) if vary else w
            im.resize((wk, h), Image.LANCZOS).save(b, "JPEG", quality=90, subsampling=2)
            data.append(b.getvalue())
        ref = [simplejpeg.decode_jpeg(x, fastdct=False, fastupsample=False) for x in data]
        got = dec.decode_batch(data)
        r = dict(width=w, height=h, megapixels=round(w * h / 1e6, 3), kb_per_file=round(sum(map(len, data)) / n / 1024, 1),
                 batched=dec.stats["batched"],
                 exact=sum(g is not None and np.array_equal(g.get(), x) for g, x in zip(got, ref)))
        del got

        def sync():
            dec.stream.synchronize()

        def one_by_one():
            for x in data:
                dec.decode(x)
            sync()
        r["gpu_single_per_s"] = rate(one_by_one, n)
        for bs in (8, 16, 32, n):
            def batched(bs=bs):
                for a in range(0, n, bs):
                    dec.decode_batch(data[a:a + bs])
                sync()
            r[f"gpu_batch{bs}_per_s"] = rate(batched, n)

        def cpu1():
            for x in data:
                simplejpeg.decode_jpeg(x, fastdct=False, fastupsample=False)
        r["cpu_1t_per_s"] = rate(cpu1, n)
        for nt in (4, 8):
            with ThreadPoolExecutor(nt) as ex:
                r[f"cpu_{nt}t_per_s"] = rate(lambda: list(ex.map(
                    lambda x: simplejpeg.decode_jpeg(x, fastdct=False, fastupsample=False), data)), n)
        if torch:
            tens = [torch.frombuffer(bytearray(x), dtype=torch.uint8) for x in data]

            def nv():
                decode_jpeg(tens, device="cuda")
                torch.cuda.synchronize()
            r["nvjpeg_batch_per_s"] = rate(nv, n)
        res["sets"][f"{mp:g}mp"] = r
        print(f"{mp:g} MP", json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
