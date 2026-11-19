"""Two small exports to JPEG bytes (488x480, quality 90, 4:2:0, two different images already on the GPU): two
Encoder.encode() calls in turn, against both queued with encode_async on two Encoders behind one event and then
collected. The figure in docs/pipelines.md. Every file is checked against Pillow's first.

usage: python bench/encode_two_small.py <out.json>

The images are generated (bench/reproduce.py's synthetic_photo); no photos, no downloads. 9 rounds of 200 pairs,
alternating which way goes first; ms per pair, the median of each round, and the median of the rounds."""
import io
import json
import os
import statistics
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from reproduce import synthetic_photo  # noqa: E402

W, H, Q, SUB = 488, 480, 90, "420"
ROUNDS, PAIRS, WARM = 9, 200, 50


def main():
    out = sys.argv[1]
    host = [synthetic_photo(W, H, s) for s in (1, 2)]
    a, b = (cp.asarray(x) for x in host)
    ref = []
    for x in host:
        buf = io.BytesIO()
        Image.fromarray(x).save(buf, "JPEG", quality=Q, subsampling=2)
        ref.append(buf.getvalue())
    e1, e2 = egj.Encoder(), egj.Encoder()
    ev = cp.cuda.Event()
    ev.record(cp.cuda.get_current_stream())

    def turn():
        return e1.encode(a, Q, SUB), e1.encode(b, Q, SUB)

    def queued():
        j1 = e1.encode_async(a, Q, SUB, ready=ev)
        j2 = e2.encode_async(b, Q, SUB, ready=ev)
        return j1.result(), j2.result()

    identical = list(turn()) == ref and list(queued()) == ref
    assert identical, "a file differs from Pillow's"
    for fn in (turn, queued):
        for _ in range(WARM):
            fn()
    rows = {"two encode() calls": [], "queued on two encoders": []}
    ways = list(zip(rows, (turn, queued)))
    for r in range(ROUNDS):
        for name, fn in (ways if r % 2 == 0 else ways[::-1]):
            ts = []
            for _ in range(PAIRS):
                cp.cuda.Device().synchronize()
                t = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t) * 1e3)
            rows[name].append(round(statistics.median(ts), 4))
    res = dict(exact_gpu_jpeg=egj.__version__, measured=time.strftime("%Y-%m-%d %H:%M"),
               gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               image=f"{W}x{H}", quality=Q, subsampling=SUB, identical_to_pillow=identical,
               rounds_ms_per_pair=rows, median_ms_per_pair={k: round(statistics.median(v), 3) for k, v in rows.items()})
    print(json.dumps(res["median_ms_per_pair"]), "identical", identical, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
