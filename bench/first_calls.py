"""What the first calls of a process cost, with nothing called before and after egj.self_check().

usage: python bench/first_calls.py <out.json> [--rounds 5]

Each measurement is a fresh Python process (this script starts them): the process either calls nothing first or
calls egj.self_check(), then, in a new thread with a new Decoder and Encoder as a worker would have, decodes one
file from its path four times and encodes one image to a file four times. The first of each is "the first call",
the median of the other three "in a loop". Cases: a 488x480 file and a 64x64 encode; a JPEG of about 4.8 MB (the
route of files with many Huffman chunks); a 3840x2560 encode at 4:4:4 (more than 65,536 blocks). The images are
generated (noise-free photo-like content, the benchmark's own generator), written to a temporary folder, and
removed afterwards. EGJ_SMALL_CPU_PIXELS is pinned so that no background measurement runs. Timings only.

Also the first run on a machine, when the kernels are compiled: a fresh process with CUPY_CACHE_DIR set to a new,
empty folder (CuPy's own kernel cache is not touched) decodes one 488x480 file, or encodes one 64x64 image, or
decodes a list of two 488x480 files; the time of that first call, which compiles what it needs. Two rounds.
"""
import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
pc = time.perf_counter


def child(a):
    import cupy as cp
    import numpy as np
    from PIL import Image
    import exact_gpu_jpeg as egj
    res = {}
    if a.compile:                                      # the first call of a process whose kernel cache is empty
        t = pc()
        if a.compile == "decode":
            egj.Decoder().decode(a.file)
        elif a.compile == "batch":
            egj.Decoder().decode_batch([a.file, a.file])
        else:
            egj.Encoder().encode(cp.zeros((64, 64, 3), cp.uint8))
        cp.cuda.Device().synchronize()
        print(json.dumps(dict(first_ms=round((pc() - t) * 1e3, 1))))
        return
    if a.warm == "self_check":
        t = pc()
        r = egj.self_check()
        res["self_check_ms"] = round((pc() - t) * 1e3, 1)
        assert r["ok"], r

    def work():
        dec = egj.Decoder()
        ts = []
        for _ in range(4):
            t = pc()
            g = dec.decode(a.file)
            dec.stream.synchronize()
            ts.append(round((pc() - t) * 1e3, 3))
        res["decode_ms"] = ts
        w, h = (int(v) for v in a.encode.split(":")[0].split("x"))
        ss = a.encode.split(":")[1]
        img = cp.asarray(np.asarray(Image.open(a.source).convert("RGB").resize((w, h), Image.BICUBIC)))
        cp.cuda.get_current_stream().synchronize()
        enc = egj.Encoder()
        ts = []
        for k in range(4):
            p = os.path.join(a.tmp, f"enc_{os.getpid()}_{k}.jpg")
            t = pc()
            enc.encode(img, quality=90, subsampling=ss, path=p)
            ts.append(round((pc() - t) * 1e3, 3))
            os.remove(p)
        res["encode_ms"] = ts
    th = threading.Thread(target=work)
    th.start()
    th.join()
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--rounds", type=int, default=5)
    for name in ("--child", "--warm", "--file", "--source", "--encode", "--tmp", "--compile"):
        ap.add_argument(name, default=None)
    a = ap.parse_args()
    if a.child:
        return child(a)
    import numpy as np
    from PIL import Image
    from reproduce import synthetic_photo
    tmp = tempfile.mkdtemp(prefix="egj_first_")
    small = os.path.join(tmp, "small_488.jpg")
    Image.fromarray(synthetic_photo(488, 480, 1)).save(small, "JPEG", quality=90)
    rng = np.random.default_rng(3)
    big = np.asarray(Image.fromarray(synthetic_photo(500, 375, 2)).resize((4000, 3000), Image.BICUBIC)).astype(np.int16)
    big = np.clip(big + rng.integers(-10, 11, big.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    mid, mid_bytes = os.path.join(tmp, "mid.jpg"), 0
    for q in range(50, 100, 2):                        # a file of 4.4 to 5.8 MB: more than 16,384 chunks of 2048 bits
        b = io.BytesIO()
        Image.fromarray(big).save(b, "JPEG", quality=q)
        if 4.4e6 <= b.tell() <= 5.8e6:
            open(mid, "wb").write(b.getvalue())
            mid_bytes = b.tell()
            break
    assert mid_bytes, "no quality gave a file of 4.4 to 5.8 MB"
    env = dict(os.environ, EGJ_SMALL_CPU_PIXELS="200000")
    cases = [("488x480 decode, 64x64 encode", small, "64x64:420"), (f"{mid_bytes / 1e6:.1f} MB decode, 3840x2560 4:4:4 encode", mid, "3840x2560:444")]
    rows = {}
    for r in range(a.rounds):
        for name, f, enc in cases:
            for warm in (("nothing", "self_check") if r % 2 == 0 else ("self_check", "nothing")):
                p = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "--child", "1", "--warm", warm, "--file", f,
                                    "--source", small, "--encode", enc, "--tmp", tmp], capture_output=True, text=True, env=env)
                assert p.returncode == 0, p.stderr[-2000:]
                rows.setdefault((name, warm), []).append(json.loads(p.stdout.strip().splitlines()[-1]))
    compiled = {}
    for r in range(2):
        for what in ("decode", "encode", "batch"):
            cache_dir = tempfile.mkdtemp(prefix="egj_kcache_")
            p = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "--child", "1", "--compile",
                                what, "--file", small, "--tmp", tmp], capture_output=True, text=True,
                               env=dict(env, CUPY_CACHE_DIR=cache_dir))
            assert p.returncode == 0, p.stderr[-2000:]
            compiled.setdefault(what, []).append(json.loads(p.stdout.strip().splitlines()[-1])["first_ms"])
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    med = statistics.median
    out = dict(measured=time.strftime("%Y-%m-%d %H:%M"), rounds=a.rounds, cases={},
               first_run_on_a_machine_ms={k: dict(each=v, median=round(med(v))) for k, v in compiled.items()})
    for (name, warm), rs in rows.items():
        c = out["cases"].setdefault(name, {})
        c[warm] = dict(first_decode_ms=round(med([x["decode_ms"][0] for x in rs]), 1), first_encode_ms=round(med([x["encode_ms"][0] for x in rs]), 1),
                       loop_decode_ms=round(med([med(x["decode_ms"][1:]) for x in rs]), 2), loop_encode_ms=round(med([med(x["encode_ms"][1:]) for x in rs]), 2))
        if warm == "self_check":
            c[warm]["self_check_ms"] = round(med([x["self_check_ms"] for x in rs]))
    print("| | nothing called before | after `egj.self_check()` | the same call in a loop |\n|---|---|---|---|")
    sc = [c["self_check"]["self_check_ms"] for c in out["cases"].values()]
    print(f"| `egj.self_check()` itself | - | {round(med(sc))} | - |")
    for name, c in out["cases"].items():
        d, e = name.split(", ")
        print(f"| first {d.replace(' decode', '')} decode from a file | {c['nothing']['first_decode_ms']:.0f} | {c['self_check']['first_decode_ms']:.0f} | {c['self_check']['loop_decode_ms']:.1f} |")
        print(f"| first {e.replace(' encode', '')} encode to a file | {c['nothing']['first_encode_ms']:.0f} | {c['self_check']['first_encode_ms']:.0f} | {c['self_check']['loop_encode_ms']:.1f} |")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
