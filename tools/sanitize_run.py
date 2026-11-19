"""Decode corpus files and fuzzed inputs with CuPy's memory pools disabled, for running under NVIDIA
compute-sanitizer (memcheck / racecheck / initcheck):

    compute-sanitizer --tool memcheck python tools/sanitize_run.py [N_MUTATIONS]

Without the pools every buffer is its own cudaMalloc, so an out-of-bounds access lands outside an allocation
and memcheck reports it.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, HERE)

import cupy as cp  # noqa: E402

cp.cuda.set_allocator(None)
cp.cuda.set_pinned_memory_allocator(None)

import corpus  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from fuzz import mutate  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    dec = egj.Decoder()
    files = [d for name, d, e in corpus.generate() if len(d) < 400000]
    ok = none = 0
    for d in files[::3]:
        r = dec.decode(d)
        ok += r is not None
        none += r is None
    rng = random.Random(99)
    for _ in range(n):
        r = dec.decode(mutate(rng.choice(files), rng))
        ok += r is not None
        none += r is None
    print(f"decoded {ok}, declined {none}")


if __name__ == "__main__":
    main()
