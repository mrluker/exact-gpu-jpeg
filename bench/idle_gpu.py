"""What a decode costs after the GPU has been idle, and whether a small GPU call during the pause changes it.

usage: python bench/idle_gpu.py <out.json> [--file large.jpg]

Nothing on the machine is changed (no power setting is touched). A large JPEG is decoded on the GPU (the file
given, else a generated 24 MP photo-like image; never the CPU route) back to back to get the sustained time; then,
after pauses of 0.25, 1, 2, 5 and 10 s in which the process sleeps, decodes run back to back for 1.5 s and each
is timed: the first three, and how long it takes until a decode is back within 1.2x the sustained time. When the
first decode after a pause is slow (more than 1.5x), the driver's clocks and performance state are read through
NVML right after it (nvml.dll / libnvidia-ml, if it loads). The same for a 488x480 decode after 0.25 and 1 s.
Then the 5 s pause again with one 64x64 GPU decode every 0.25, 0.5 and 1 s during it. About six minutes.
Timings only (no file names).
"""
import argparse
import ctypes as C
import io
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("EGJ_SMALL_CPU_PIXELS", "200000")     # pinned: no background measurement during the run
import cupy as cp  # noqa: E402
from PIL import Image  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402
from reproduce import synthetic_photo  # noqa: E402

pc = time.perf_counter
med = statistics.median
sync = cp.cuda.get_current_stream().synchronize


def jpeg(a, q=90):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q)
    return b.getvalue()


def nvml():
    """A function that reads the driver's clocks, performance state and power, or None if NVML does not load."""
    for name in ("nvml.dll", "libnvidia-ml.so.1"):
        try:
            nv = C.CDLL(name)
            break
        except OSError:
            nv = None
    if nv is None or nv.nvmlInit_v2() != 0:
        return None
    h = C.c_void_p()
    if nv.nvmlDeviceGetHandleByIndex_v2(C.c_uint(cp.cuda.Device().id), C.byref(h)) != 0:
        return None

    def read():
        o = {}
        for label, t in (("graphics_mhz", 0), ("memory_mhz", 2)):
            v = C.c_uint()
            o[label] = v.value if nv.nvmlDeviceGetClockInfo(h, C.c_int(t), C.byref(v)) == 0 else None
        v = C.c_int()
        o["pstate"] = v.value if nv.nvmlDeviceGetPerformanceState(h, C.byref(v)) == 0 else None
        v = C.c_uint()
        o["power_w"] = round(v.value / 1e3, 1) if nv.nvmlDeviceGetPowerUsage(h, C.byref(v)) == 0 else None
        return o
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--file")
    a = ap.parse_args()
    large = open(a.file, "rb").read() if a.file else jpeg(synthetic_photo(6000, 4000, 1))
    small, tiny = jpeg(synthetic_photo(488, 480, 2)), jpeg(synthetic_photo(64, 64, 3))
    read = nvml()

    def call(d):
        t = pc()
        r = egj.decode(d, cpu_below=0)
        sync()
        t = (pc() - t) * 1e3
        del r
        return t

    for d in (large, small, tiny):
        for _ in range(4):
            call(d)
    res = dict(measured=time.strftime("%Y-%m-%d %H:%M"), gpu=egj.device_report()["gpu"], large_bytes=len(large),
               nvml_warm=read() if read else None, rows=[])
    warm = {"large": med([call(large) for _ in range(30)]), "small": med([call(small) for _ in range(200)])}
    res["sustained_ms"] = {k: round(v, 3) for k, v in warm.items()}

    def after(gap, keep, what, data, seconds):
        if keep is None:
            time.sleep(gap)
        else:
            t_end = pc() + gap
            while pc() + keep < t_end:
                time.sleep(keep)
                call(tiny)
            time.sleep(max(0.0, t_end - pc()))
        t0 = pc()
        first = call(data)
        state = read() if (read and first > 1.5 * warm[what]) else None
        ts, normal_from = [first], (0.0 if first < 1.2 * warm[what] else None)
        while pc() - t0 < seconds:
            at = (pc() - t0) * 1e3
            ms = call(data)
            ts.append(ms)
            if normal_from is None and ms < 1.2 * warm[what]:
                normal_from = at
        return dict(first3=[round(x, 2) for x in ts[:3]], normal_from_ms=None if normal_from is None else round(normal_from),
                    slow=first > 1.5 * warm[what], nvml_after_first=state)

    plan = ([("large", large, g, None, n, 1.5) for g, n in ((0.25, 15), (1, 8), (2, 8), (5, 8), (10, 6))]
            + [("small", small, g, None, 15, 0.05) for g in (0.25, 1)]
            + [("large", large, 5, k, 8, 1.5) for k in (0.25, 0.5, 1)])
    print("| pause | call | first decode after it, ms (median; range) | slow (more than 1.5x) | normal again after, ms | driver during a slow one |\n|---|---|---|---|---|---|")
    for what, data, gap, keep, tries, seconds in plan:
        rs = [after(gap, keep, what, data, seconds) for _ in range(tries)]
        firsts = [r["first3"][0] for r in rs]
        slow = [r for r in rs if r["slow"]]
        states = sorted({f"P{r['nvml_after_first']['pstate']} {r['nvml_after_first']['graphics_mhz']} MHz" for r in slow if r["nvml_after_first"]})
        back = [r["normal_from_ms"] for r in slow if r["normal_from_ms"] is not None]
        res["rows"].append(dict(call=what, pause_s=gap, keep_alive_s=keep, tries=tries, first_ms_median=round(med(firsts), 2),
                                first_ms_range=[min(firsts), max(firsts)], slow=len(slow), normal_again_ms=back, driver=states, runs=rs))
        label = f"{gap:g} s" + ("" if keep is None else f", a 64x64 GPU decode every {keep:g} s")
        print(f"| {label} | {what} (sustained {warm[what]:.2f}) | {med(firsts):.2f}; {min(firsts):.2f}-{max(firsts):.2f} | {len(slow)} of {tries} | "
              f"{'-' if not back else f'{min(back)}-{max(back)}'} | {', '.join(states) or '-'} |", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
