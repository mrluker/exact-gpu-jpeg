"""Keep the GPU busy from a second process while the tests or gates run in another.

usage: python tools/gpu_load.py <seconds>
   e.g.  python tools/gpu_load.py 120 &  pytest

Timing-dependent faults (work on one CUDA stream read before another stream has finished writing it) hardly ever
show on an idle GPU, where queued work runs at once; with another process taking GPU time they do. Every gate of
this library is run once this way. The load is this library decoding and encoding synthetic images in a loop.
"""
import io
import sys
import time

import numpy as np
from PIL import Image

import exact_gpu_jpeg as egj


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(2)
    data = []
    for _ in range(24):
        a = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
        b = io.BytesIO()
        Image.fromarray(a).resize((1600, 1200), Image.BICUBIC).save(b, "JPEG", quality=90, subsampling=2)
        data.append(b.getvalue())
    dec, enc = egj.Decoder(), egj.Encoder()
    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        for d in data:
            img = dec.decode(d)
            enc.encode(img, 85, "420")
            n += 1
    print(f"gpu_load: {n} decode + encode rounds in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
