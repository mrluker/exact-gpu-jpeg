"""The start-up self-check, and what happens when the GPU path cannot be trusted.

Once per process and per GPU, before the first decode, the first batch and the first encode return anything, a few
small built-in cases run on the GPU and their results are compared with SHA-256 hashes stored in the package
(_selfcheck_data.py, written by tools/make_selfcheck.py): JPEGs encoded by libjpeg-turbo whose decoded pixels were
hashed from libjpeg-turbo's own decode, and synthetic images whose encoded files were hashed from Pillow's.
A decode-only program never compiles the encoder: each part runs when it is first needed
(exact_gpu_jpeg.self_check() runs them all, e.g. at warm-up).

If a case fails, if a kernel does not compile, or if CUDA reports an error at any later time, the GPU path is
switched off for this process on that device, with one warning that names the GPU, the driver and the reason.
From then on Decoder.decode() declines every file with that reason (so exact_gpu_jpeg.decode() uses the CPU
decoder) and encode() raises it. Out-of-memory is not such an error: it declines one file and nothing else.

EGJ_SELFCHECK_FAIL=1 forces the check to fail (for testing the fallback). There is no switch to skip it.

Before any of that, and before any kernel is compiled, the device check reads what the card and the CUDA runtime
report (compute capability, memory, the kernel compiler) and turns the GPU path off in the same way when a
requirement is not met. Two settings are derived from the reported values, the same every time for the same card:
the GPU memory budget of a decode (90% of the free memory) and the amount of data one batch may hold (in
proportion to the card's memory, up to the 64 MB that was tested). Nothing is timed and nothing is stored.
"""
from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import threading
import time
import warnings

import numpy as np

FAILED = {}                # device id -> why the GPU path is off for this process
_done = set()              # (device id, part) that passed
_lock = threading.RLock()
_tl = threading.local()
_names = {}
PARTS = ("decode", "batch", "encode")
# Decode modes the start-up check does not run: each is checked on the device when it is first asked for, with the
# same rule (a failure switches the GPU path off for the process). "output": float output, CHW, BGR.
MODE_PARTS = ("fast_dct", "fast_upsample", "output")
MIN_CAPABILITY = (5, 0)    # the oldest GPUs CUDA 12 supports; the kernels need nothing newer
BUDGET = 0.9               # share of the free GPU memory one decode (or one batch) may plan to use
BATCH_RAW_MAX = 64 << 20   # entropy-coded data per batch on an 8 GB card and larger (the tested value); smaller
BATCH_RAW_MIN = 8 << 20    # cards get total / 128, not less than this
_props = {}                # device id -> what detect() returned
_checked = set()           # device ids whose device check has run
times_ms = {}              # (device id, part) -> how long the part took (first use, compilation included)


class GuardBandError(RuntimeError):
    """A kernel wrote outside a device buffer (found by the guard-band test mode). Never turned into a decline."""


def synthetic(width, height, gray=False):
    """The self-check's image, the same rule in exact-gpu-jpeg and its sister library for Apple GPUs:
    v = (x*7 + y*13 + c*61 + ((x*y) >> 3)) & 255, then +40 where ((x >> 4) + (y >> 4)) is even and -40 where it
    is odd, clamped to 0..255; uint8 (height, width, 3) RGB, or (height, width) with c = 0 for gray."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    c = np.arange(1 if gray else 3, dtype=np.int64).reshape(1, 1, -1)
    v = (x[..., None] * 7 + y[..., None] * 13 + c * 61 + ((x * y) >> 3)[..., None]) & 255
    v = v + np.where((((x >> 4) + (y >> 4)) & 1) == 0, 40, -40)[..., None]
    v = np.clip(v, 0, 255).astype(np.uint8)
    return v[..., 0] if gray else v


def gpu_name(dev):
    """'<GPU name>, driver <version>' for the messages (nvidia-smi; 'CUDA device N' if it cannot be asked)."""
    r = _names.get(dev)
    if r is None:
        r = f"CUDA device {dev}, driver unknown"
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", f"--id={dev}"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            name, drv = [v.strip() for v in out[0].split(",")]
            if name and drv:
                r = f"{name}, driver {drv}"
        except Exception:
            pass
        _names[dev] = r
    return r


def detect(dev):
    """What the device and the CUDA runtime report. Tests replace this function to pretend to be another card."""
    import cupy as cp
    d = cp.cuda.Device(dev)
    cc = str(d.compute_capability)
    free, total = d.mem_info
    rt = cp.cuda.runtime.runtimeGetVersion()
    try:
        nvrtc = ".".join(str(v) for v in cp.cuda.nvrtc.getVersion())
    except Exception as e:                     # the kernel compiler is not installed or does not load
        nvrtc = None
        _props.setdefault(("nvrtc error", dev), _first_line(e))
    return dict(compute_capability=(int(cc[:-1]), int(cc[-1])), total_memory=int(total), free_memory=int(free),
                cuda_runtime=f"{rt // 1000}.{rt % 1000 // 10}", nvrtc=nvrtc)


def free_memory(dev):
    """Free GPU memory now, as the driver reports it (tests replace this together with detect)."""
    import cupy as cp
    with cp.cuda.Device(dev):
        return int(cp.cuda.runtime.memGetInfo()[0])


def properties(dev):
    r = _props.get(dev)
    if r is None:
        r = _props[dev] = detect(dev)
    return r


def settings(dev):
    """The settings derived from what the device reports: the same card always gets the same values."""
    total = properties(dev)["total_memory"]
    return dict(memory_budget_share=BUDGET,
                batch_data_bytes=int(min(BATCH_RAW_MAX, max(BATCH_RAW_MIN, total // 128))))


def budget(dev):
    """Bytes of GPU memory a decode may plan to use right now: BUDGET of what is free (plus what CuPy's pool holds)."""
    import cupy as cp
    return int(BUDGET * (free_memory(dev) + cp.get_default_memory_pool().free_bytes()))


def _device_check(dev):
    """None, or which requirement the device does not meet."""
    try:
        p = properties(dev)
    except Exception as e:
        return f"the device could not be queried ({_first_line(e)})"
    cc = p["compute_capability"]
    if cc < MIN_CAPABILITY:
        return (f"compute capability {cc[0]}.{cc[1]} is below {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]}, the oldest "
                "this library's kernels are built for")
    if p["nvrtc"] is None:
        why = _props.get(("nvrtc error", dev))
        return "the CUDA kernel compiler (NVRTC) could not be loaded" + (f" ({why})" if why else "")
    return None


def _first_line(e):
    s = str(e).strip().splitlines()
    return f"{type(e).__name__}: {s[0][:200]}" if s else type(e).__name__


def _off(dev, text):
    with _lock:
        r = FAILED.get(dev)
        if r is None:
            r = FAILED[dev] = text
            warnings.warn(f"exact_gpu_jpeg: {r}. The GPU path is off for this process: decode() uses the CPU "
                          "decoder, encode() raises.", RuntimeWarning, stacklevel=4)
        return r


def failed(dev):
    """Why the GPU path is off on this device, or None."""
    return FAILED.get(dev)


def gpu_error(dev, exc):
    """A CUDA error or a kernel that did not compile: the GPU path goes off. Returns the reason. Inside the
    self-check the error is only noted: the check reports it under its own wording, with the case it happened in
    ("self-check failed on <GPU>: <case> (<error>)")."""
    if getattr(_tl, "active", False):
        _tl.error = _first_line(exc)
        return f"GPU error: {_tl.error}"
    return _off(dev, f"GPU error on {gpu_name(dev)}: {_first_line(exc)}")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class _NoMemory(Exception):
    """A case of the check could not run for lack of GPU memory: that says nothing about the kernels."""


def _memory(reason):
    """A decline for lack of GPU memory, or of page-locked host memory for the file: not a fault of the kernels."""
    return reason is not None and (reason == "out of GPU memory" or "of GPU memory, more than allowed" in reason
                                   or reason.startswith("could not page-lock"))


def _run(dev, part):
    """None if every case of the part gives the stored hash, else the name of the first case that does not.
    Raises _NoMemory when a case could not run for lack of GPU memory."""
    from . import _selfcheck_data as D
    if os.environ.get("EGJ_SELFCHECK_FAIL") == "1":
        first = {"decode": D.DECODE[0][0], "batch": D.BATCH[0], "encode": D.ENCODE[0][0]}
        first.update({p: next(m[1] for m in D.MODES if m[0] == p) for p in MODE_PARTS})
        return first[part] + " (forced by EGJ_SELFCHECK_FAIL)"
    if part in MODE_PARTS:
        return _run_mode(dev, part, D)
    if part == "encode":
        from ._encoder import Encoder
        enc = Encoder(dev)
        for name, w, h, q, ss, sha in D.ENCODE:
            _tl.case = name
            if _sha(enc.encode(synthetic(w, h), quality=q, subsampling=ss)) != sha:
                return name
        return None
    from ._decoder import Decoder
    dec = Decoder(dev)
    if part == "decode":
        for name, cs, sha, b64 in D.DECODE:
            _tl.case = name
            img = dec.decode(base64.b64decode(b64), cs)
            if img is None and not getattr(_tl, "error", None) and _memory(dec.reason):
                raise _NoMemory(dec.reason)
            if img is None:
                return f"{name} ({getattr(_tl, 'error', None) or 'declined: ' + str(dec.reason)})"
            if _sha(img.get().tobytes()) != sha:
                return name
        return None
    name, items = D.BATCH
    _tl.case = name
    imgs = dec.decode_batch([base64.b64decode(b64) for _, b64 in items])
    if getattr(_tl, "error", None):            # the batch kernels failed (the files were then decoded one by one)
        return f"{name} ({_tl.error})"
    if any(_memory(why) for why in dec.reasons) or dec.stats.get("batched") != len(items):
        # the batch kernels did not run on both files (short of memory, most likely): the part is not checked
        raise _NoMemory("the batch could not be decoded as one batch")
    for (sha, _), img, why in zip(items, imgs, dec.reasons):
        if img is None:
            return f"{name} ({getattr(_tl, 'error', None) or 'declined: ' + str(why)})"
        if _sha(img.get().tobytes()) != sha:
            return name
    return None


def _run_mode(dev, part, D):
    """A decode mode's cases: each decoded alone and in a batch of two (the batch kernels have their own pixel
    stage), compared with the stored hash."""
    from ._decoder import Decoder
    for p, name, opts, want, b64 in D.MODES:
        if p != part:
            continue
        _tl.case = name
        opts = dict(opts)
        cs = opts.pop("colorspace", "RGB")
        fast_dct, fast_up = opts.pop("fast_dct", False), opts.pop("fast_upsample", False)
        dec = Decoder(dev, fast_dct=fast_dct, fast_upsample=fast_up)
        data = base64.b64decode(b64)
        img = dec.decode(data, cs, **opts)
        r = _verdict(name, img, dec.reason, want)              # (the single decode's reason, read before the batch)
        if r is not None:
            return r
        imgs = list(dec.decode_batch([data, data], cs, **opts))
        if getattr(_tl, "error", None):        # the batch kernels failed (the files were then decoded one by one)
            return f"{name} (batch: {_tl.error})"
        if not fast_dct and (any(_memory(w) for w in dec.reasons) or dec.stats.get("batched") != 2):
            # the batch kernels did not run in this mode (short of memory, most likely): the part is not checked
            # (the fast IDCT has no batch kernel: its batch half is two single decodes)
            raise _NoMemory("the mode's batch could not be decoded as one batch")
        for img, why in zip(imgs, dec.reasons):
            r = _verdict(name + " (batch)", img, why, want)
            if r is not None:
                return r
    return None


def _verdict(name, img, why, want):
    """None if img is there and has the stored hash; raises _NoMemory for a memory decline; else the case's name."""
    if img is None:
        why = getattr(_tl, "error", None) or why
        if not getattr(_tl, "error", None) and _memory(why):
            raise _NoMemory(why)
        return f"{name} ({why or 'declined'})"
    if _sha(img.get().tobytes()) != want:
        return name
    return None


def mode_parts(fast_dct, fast_upsample, epi, colorspace):
    """The mode checks a decode with these settings needs."""
    return tuple(p for p, on in (("fast_dct", fast_dct), ("fast_upsample", fast_upsample),
                                 ("output", epi is not None or colorspace == "BGR")) if on)


def ensure_modes(dev, parts):
    """ensure() for each of the parts; the first reason, or None."""
    for p in parts:
        r = ensure(dev, p)
        if r is not None:
            return r
    return None


def _cp_oom():
    try:
        import cupy as cp
        return cp.cuda.memory.OutOfMemoryError
    except ImportError:                        # (no CuPy: the check cannot run anyway)
        return _NoMemory


def ensure(dev, part):
    """Run the part's cases if they have not run in this process on this device. Returns None when the GPU path
    can be used, else the reason it is off."""
    r = FAILED.get(dev)
    if r is not None or (dev, part) in _done or getattr(_tl, "active", False):
        return r
    with _lock:                                # concurrent first calls: one runs the check, the others wait
        if FAILED.get(dev) is not None or (dev, part) in _done:
            return FAILED.get(dev)
        if dev not in _checked:                # the device check, once, before any kernel is compiled
            _checked.add(dev)
            why = _device_check(dev)
            if why is not None:
                return _off(dev, f"device check failed on {gpu_name(dev)}: {why}")
        _tl.active = True                      # the check's own calls must not start it again
        _tl.error = _tl.case = None
        t = time.perf_counter()
        try:
            case = _run(dev, part)
        except GuardBandError:
            raise
        except (_NoMemory, _cp_oom()) as e:
            # out of GPU memory says nothing about the kernels: neither failed nor done, so the next call runs
            # the check again; this call is declined (decode() then uses the CPU decoder)
            _tl.active = False
            times_ms[(dev, part)] = round((time.perf_counter() - t) * 1e3, 1)
            return f"out of GPU memory (the first-use check of the {part} kernels could not run: {_first_line(e)})"
        except Exception as e:                 # a kernel that does not compile, a CUDA error, a missing NVRTC
            case = f"{_tl.case or part} ({_tl.error or _first_line(e)})"
        finally:
            _tl.active = False
        times_ms[(dev, part)] = round((time.perf_counter() - t) * 1e3, 1)
        if case is None:
            _done.add((dev, part))
            return None
        return _off(dev, f"self-check failed on {gpu_name(dev)}: {case}")


def run_all(dev):
    """Every part, now. A report: device, gpu, ok, reason, and the time each part took when it ran."""
    reason = None
    for part in PARTS + MODE_PARTS:
        reason = ensure(dev, part)
        if reason is not None:
            break
    return dict(device=dev, gpu=gpu_name(dev), ok=reason is None, reason=reason,
                ms={p: times_ms.get((dev, p)) for p in PARTS + MODE_PARTS})


def report(dev):
    """What was detected, what was chosen and what has been checked so far, for a user or a bug report. Runs no
    kernel: the self-check parts that have not run yet are reported as "not run yet"."""
    out = dict(device=dev, gpu=gpu_name(dev))
    try:
        p = properties(dev)
        out.update(compute_capability=f"{p['compute_capability'][0]}.{p['compute_capability'][1]}",
                   total_memory_mb=p["total_memory"] >> 20, free_memory_mb=free_memory(dev) >> 20,
                   cuda_runtime=p["cuda_runtime"], nvrtc=p["nvrtc"] or "not available")
        out["derived_settings"] = settings(dev)
    except Exception as e:
        out["detect_error"] = _first_line(e)
    out["minimum_compute_capability"] = f"{MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]}"
    out["self_check"] = {p: ("passed" if (dev, p) in _done else "not run yet") for p in PARTS + MODE_PARTS}
    out["gpu_path"] = "off: " + FAILED[dev] if dev in FAILED else "on"
    return out
