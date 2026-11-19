"""The check of the copy kernels, run once per device at the first use of the packed form there
(keep_coefs="packed", a RepeatCache; never at start-up, and never in a process that keeps no packed copies): a
packed copy of a built-in test vector, made and unpacked on this GPU, against the vector's stored hashes.

A failure does not switch the GPU path off, and no decode is affected: copies are then kept in the raw form (the
decoder's own arrays). One warning says so; device_report() says why.
"""
import hashlib
import os
import threading
import warnings

import numpy as np

PARTS = ("packed",)
WHAT = {"packed": "packed copies are kept in the raw form"}
# the test vector: 64 blocks of formula-made coefficients
BLOCKS = 64
ARRAYS = "41c119404c6401a4a512901b14839de86b507cd7f9a81ea90c35802f18a77f86"
PACKED_BODY = "3fc65fcfba9b23f77e70aa124ab8ee5058c24c5d9ecd7b78828da61968571b4d"
_state = {}                # device id -> {part: None (passed) or why it failed}
_lock = threading.Lock()


def vector_arrays(seed=1, B=BLOCKS):
    """(coef (B, 64) int16, nq (B,) uint8, dc (B,) int32) of the packed form's formula vectors (xorshift64 from the
    seed; values within 11 bits)."""
    M = (1 << 64) - 1
    s = seed

    def nxt():
        nonlocal s
        s ^= (s << 13) & M
        s ^= s >> 7
        s ^= (s << 17) & M
        return s
    coef = np.zeros((B, 64), np.int16)
    nq = np.zeros(B, np.uint8)
    dc = np.zeros(B, np.int32)
    for b in range(B):
        r = nxt()
        dc[b] = (r % 2047) - 1023
        k = (r >> 16) % 8
        count = 0 if k == 0 else 63 if k == 1 else 1 + ((r >> 24) % 24)
        limit = 1023 if k == 2 else 127 if k == 3 else 7
        top = 0
        for j in range(count):
            r2 = nxt()
            pos = 1 + j if k == 1 else 1 + (r2 % 63)
            coef[b, pos] = -(1 + ((r2 >> 8) % limit)) if (r2 >> 40) & 1 else 1 + ((r2 >> 8) % limit)
            top = max(top, pos)
        nq[b] = 0 if count == 0 else top // 16 + 1
    return coef, nq, dc


def _arrays_sha(coef, nq, dc):
    c = np.array(coef, "<i2").reshape(len(nq), 64)
    c[np.arange(64)[None, :] >= 16 * np.asarray(nq, np.int64)[:, None]] = 0        # past a block's quads: not defined
    return hashlib.sha256(c.tobytes() + np.asarray(nq, np.uint8).tobytes() + np.asarray(dc, "<i4").tobytes()).hexdigest()


def _run(dev):
    import cupy as cp
    from . import _packed
    from ._decoder import Decoder
    res = dict.fromkeys(PARTS)
    forced = [p.strip() for p in os.environ.get("EGJ_COPYCHECK_FAIL", "").split(",")]
    dec = Decoder(dev)
    coef, nq, dc = vector_arrays()
    if _arrays_sha(coef, nq, dc) != ARRAYS:                    # (the generator itself)
        return dict.fromkeys(PARTS, "the test vector's arrays differ")

    def fresh():
        return (dec._alloc(BLOCKS * 64, cp.int16, zero=True), dec._alloc(BLOCKS, cp.uint8, zero=True),
                dec._alloc(BLOCKS, cp.int32, zero=True), dec._alloc(1, cp.int32, zero=True))

    def back(c, n, d, e):
        return int(e.get()[0]) == 0 and _arrays_sha(cp.asnumpy(c), cp.asnumpy(n), cp.asnumpy(d)) == ARRAYS

    def part(name, f):
        if name in forced:
            res[name] = "forced by EGJ_COPYCHECK_FAIL"
            return
        try:
            with dec.device, dec.stream:
                res[name] = f()
        except cp.cuda.memory.OutOfMemoryError:                # says nothing about the kernels: checked again later
            dec.stream.synchronize()
            res[name] = NO_MEMORY
        except Exception as e:                                 # a kernel that does not compile, a CUDA error
            line = str(e).strip().splitlines()
            res[name] = f"{type(e).__name__}: {line[0][:200]}" if line else type(e).__name__

    def packed():
        g = (cp.asarray(coef.reshape(-1)), cp.asarray(nq), cp.asarray(dc))
        made = _packed.pack(dec._alloc, *g, BLOCKS, dec.stream)
        if made is None or hashlib.sha256(_packed.body(*made)).hexdigest() != PACKED_BODY:
            return "the packed copy of the test vector differs"
        c, n, d, e = fresh()
        _packed.unpack(made[0], made[1], made[2], BLOCKS, c, d, n, e)
        return None if back(c, n, d, e) else "the unpacked test vector differs"

    part("packed", packed)
    return res


NO_MEMORY = "out of GPU memory (not checked; it runs again at the next use)"


def state(dev):
    """{part: None or why it failed} for the device; the check runs at the first call."""
    r = _state.get(dev)
    if r is None:
        from . import _selfcheck
        if _selfcheck.ensure(dev, "decode") is not None:       # the GPU path is off: nothing to check, no copies
            return dict.fromkeys(PARTS, "the GPU path is off")
        with _lock:
            r = _state.get(dev)
            if r is None:
                r = _run(dev)
                if NO_MEMORY in r.values():                    # not checked: no copy of that form this time, and
                    return r                                   # the check runs again at the next use
                _state[dev] = r                                # (stored before the warning, which may raise)
                bad = [f"{p}: {r[p]} ({WHAT[p]})" for p in PARTS if r[p] is not None]
                if bad:
                    from ._selfcheck import gpu_name
                    warnings.warn(f"exact_gpu_jpeg: the check of the copy kernels failed on {gpu_name(dev)}: "
                                  + "; ".join(bad) + ". Decodes are not affected.", RuntimeWarning, stacklevel=4)
    return r


def report(dev):
    """For device_report(): each part "passed", why it failed, or "not run yet" (it runs at the first use of a
    copy form on the device)."""
    r = _state.get(dev)
    return {p: ("not run yet" if r is None else "passed" if r[p] is None else "failed: " + r[p]) for p in PARTS}
