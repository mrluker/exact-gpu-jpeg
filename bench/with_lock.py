"""Run a timing benchmark alone on the machine: under a lock file, and only once the machine is idle.

The lock file is the environment variable EGJ_BENCH_LOCK, or exact-gpu-jpeg-bench.lock in the temporary folder.

usage: python bench/with_lock.py "<what is running>" <conditions.json> -- <command> [args...]

1. While the lock exists: print it, wait (re-check every 2 min). A lock older than 12 h is
   reported and NOT deleted; this script exits (remove the file yourself if nothing is running).
2. Wait until the machine is otherwise idle: GPU utilisation <= 10% and CPU load <= 25% (Win32 LoadPercentage; the sampling itself and desktop noise read 10-20%)
   (sampled 5 times over ~10 s; other processes' GPU memory is recorded).
3. Create the lock: "exact-gpu-jpeg | <what> | <start time>".
4. Run the command; remove the lock afterwards, even on failure.
The idle check results go into <conditions.json>, so every results table can say whether the
machine was otherwise idle.
"""
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

LOCK = os.environ.get("EGJ_BENCH_LOCK") or os.path.join(tempfile.gettempdir(), "exact-gpu-jpeg-bench.lock")


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    u, m, t, c = [float(x) for x in out.split(",")]
    return dict(gpu_util=u, gpu_mem_mib=m, gpu_temp_c=t, gpu_sm_mhz=c)


def cpu_load():
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).LoadPercentage"], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


def idle_sample(n=5):
    s = []
    for _ in range(n):
        g = gpu_state()
        g["cpu_load"] = cpu_load()
        s.append(g)
        time.sleep(1)
    return dict(gpu_util_max=max(x["gpu_util"] for x in s), cpu_load_max=max(x["cpu_load"] for x in s),
                gpu_mem_mib=s[-1]["gpu_mem_mib"], gpu_temp_c=s[-1]["gpu_temp_c"], samples=s)


def main():
    what, cond_path = sys.argv[1], sys.argv[2]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    while os.path.exists(LOCK):
        age_h = (time.time() - os.path.getmtime(LOCK)) / 3600
        txt = open(LOCK).read().strip()
        if age_h > 12:
            print(f"LOCK OLDER THAN 12 h ({age_h:.1f} h): {txt}\nNot deleting it; remove it yourself if no benchmark is running.", flush=True)
            sys.exit(3)
        print(f"{datetime.datetime.now():%H:%M:%S} waiting for lock: {txt}", flush=True)
        time.sleep(120)
    tries = 0
    while True:
        st = idle_sample()
        if st["gpu_util_max"] <= 10 and st["cpu_load_max"] <= 25:
            break
        tries += 1
        print(f"{datetime.datetime.now():%H:%M:%S} machine busy (GPU {st['gpu_util_max']:.0f}%, CPU {st['cpu_load_max']:.0f}%), waiting", flush=True)
        if os.path.exists(LOCK):
            return main()
        time.sleep(30 if tries < 20 else 120)
    start = datetime.datetime.now()
    try:
        with open(LOCK, "x") as f:                    # atomic: fails if the other project just took it
            f.write(f"exact-gpu-jpeg | {what} | {start:%Y-%m-%d %H:%M:%S}\n")
    except FileExistsError:
        return main()
    rc = 1
    try:
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        wall = time.perf_counter() - t0
    finally:
        try:
            os.remove(LOCK)
        except FileNotFoundError:
            pass
    st_after = gpu_state()
    with open(cond_path, "w") as f:
        json.dump(dict(what=what, start=f"{start:%Y-%m-%d %H:%M:%S}", wall_s=wall, returncode=rc,
                       idle_before=st, gpu_after=st_after,
                       otherwise_idle=bool(st["gpu_util_max"] <= 10 and st["cpu_load_max"] <= 25)), f, indent=1)
    print(f"done in {wall:.1f}s (rc {rc}); lock released; conditions -> {cond_path}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
