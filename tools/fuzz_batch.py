"""Fuzzer for decode_batch: batches of mutated JPEGs mixed with untouched ones, guard bands on.

usage: python tools/fuzz_batch.py <out.json> [--batches N] [--size K] [--seed S]

For every batch (K inputs: the fuzzer's seeds, each untouched with probability 0.3, else mutated by tools/fuzz.py's
mutator), decode_batch must give, for every input, exactly what decode() gives for it alone: the same image, or
None for both. An image the GPU returns must also equal libjpeg-turbo's. Guard bands (canary) surround every device
buffer in both decoders, so a write outside a buffer raises. Counts only in the output; a finding's batch is saved
next to it (<out>.finding_<n>.bin, the inputs length-prefixed) for a regression test.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz import make_seeds, mutate, reference, strict_refuses  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seeds = make_seeds(random.Random(a.seed))
    batch, one = egj.Decoder(canary=True), egj.Decoder(canary=True)
    c = dict(batches=0, inputs=0, untouched=0, gpu_exact=0, declined=0, batched=0, findings=0)
    t0 = time.time()
    for b in range(a.batches):
        files = []
        for _ in range(rng.choice([2, 3, a.size // 2, a.size])):
            d = rng.choice(seeds)
            if rng.random() < 0.3:
                c["untouched"] += 1
            else:
                d = mutate(d, rng)
            files.append(d)
        problem = None
        try:
            got = batch.decode_batch(files)
            c["batched"] += batch.stats["batched"]
            for k, d in enumerate(files):
                want = one.decode(d)
                if want is None and got[k] is not None and one.reason == "Huffman chunks did not synchronise":
                    # documented: a batch may decode a file a single decode declines for synchronisation (never
                    # the reverse); its image must still be libjpeg-turbo's
                    ref = reference(d)
                    g = got[k].get()
                    if ref is None or ref.shape != g.shape or not np.array_equal(g, ref):
                        problem = f"input {k}: decoded only by the batch, and differs from libjpeg-turbo"
                        break
                    c["batch_only_sync"] = c.get("batch_only_sync", 0) + 1
                    continue
                if (want is None) != (got[k] is None):
                    problem = f"input {k}: batch {'declined' if got[k] is None else 'decoded'}, decode() did not agree"
                    break
                if want is None:
                    c["declined"] += 1
                    continue
                g = got[k].get()
                if g.shape != want.shape or not np.array_equal(g, want.get()):
                    problem = f"input {k}: batch image differs from decode()"
                    break
                ref = reference(d)
                if ref is None or ref.shape != g.shape or not np.array_equal(g, ref):
                    problem = f"input {k}: differs from libjpeg-turbo"
                    break
                if strict_refuses(d):                         # the decoders are strict (the default)
                    problem = f"input {k}: decoded under strict=True, and libjpeg-turbo warns about it"
                    break
                c["gpu_exact"] += 1
        except Exception as e:                                # includes guard-band violations and CUDA errors
            problem = f"{type(e).__name__}: {e}"
        c["batches"] += 1
        c["inputs"] += len(files)
        if problem:
            c["findings"] += 1
            with open(f"{a.out}.finding_{c['findings']}.bin", "wb") as f:
                for d in files:
                    f.write(len(d).to_bytes(4, "little") + bytes(d))
            print("FINDING", b, problem, flush=True)
            batch, one = egj.Decoder(canary=True), egj.Decoder(canary=True)
        if b % 50 == 49:
            print(b + 1, json.dumps(c), flush=True)
    c.update(seed=a.seed, seconds=round(time.time() - t0, 1))
    json.dump(c, open(a.out, "w"), indent=1)
    print("done", json.dumps(c))
    return 1 if c["findings"] or not c["batched"] else 0       # (no image went through the batch kernels: no test)


if __name__ == "__main__":
    sys.exit(main())
