"""Mutation fuzzer for exact_gpu_jpeg. Everything is generated locally; nothing is downloaded.

usage: python tools/fuzz.py <out_dir> [--iters N] [--seed S] [--threads T] [--timeout SEC] [--seeds-dir DIR]

A parent process runs a worker and watches it. The worker mutates seed JPEGs and decodes each mutant on T threads
at once (each thread its own Decoder and CUDA stream; default 2). Seeds: synthetic JPEGs generated here (libjpeg-
turbo through simplejpeg and Pillow, and tests/jpegwriter.py), plus small files from --seeds-dir if given (e.g.
crops of your own photos made by tools/make_local_corpus.py; findings from those stay in <out_dir>, not in git).

Mutations: bit flips, byte changes, truncation anywhere and at every structural point (marker boundaries, inside
segments, inside the scan), inserted/deleted bytes, inserted, removed or renumbered restart markers, stray 0xFF
and broken 0xFF00 stuffing, fill bytes, corrupted DQT/DHT/SOF/SOS/DRI fields, absurd or zero dimensions, wrong
component counts, zero-length and oversized segment lengths, duplicated, dropped and reordered segments, random
scan data, spliced scans.

Findings (each input saved to <out_dir>/findings):
  exception        decode raised (it must return an array or None)
  mismatch         an image differing from libjpeg-turbo's (simplejpeg strict=False, else Pillow), or an image
                   for a file libjpeg-turbo cannot decode
  nondeterministic two decodes of the same input differ
  leak             after a declined (bad) input, a known-good file no longer decodes byte-exactly
  cuda_error       the CUDA context reports an error after a decode
  canary           (with EGJ_CANARY=1) a kernel wrote outside a device buffer
  crash / hang     the worker died, or made no progress for --timeout seconds
Summary with counts, seed and settings: <out_dir>/summary.json.
"""
import argparse
import collections
import io
import json
import os
import random
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))


# ---------------- seeds ----------------

def make_seeds(rng):
    import numpy as np
    import simplejpeg
    from PIL import Image
    from jpegwriter import grid, write_jpeg
    seeds = []
    nrng = np.random.default_rng(rng.randrange(1 << 30))

    def content(h, w):
        y, x = np.mgrid[0:h, 0:w]
        a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x * y) % 256], -1).astype(np.float32)
        a += nrng.normal(0, 30, a.shape)
        return np.clip(a, 0, 255).astype(np.uint8)
    for h, w in [(8, 8), (16, 24), (33, 17), (1, 1), (5, 70), (64, 64), (120, 90)]:
        a = content(h, w)
        for sub in ("444", "422", "420", "440", "411", "Gray"):
            src = a[..., :1].copy() if sub == "Gray" else a
            seeds.append(simplejpeg.encode_jpeg(src, quality=80, colorspace="GRAY" if sub == "Gray" else "RGB",
                                                colorsubsampling=sub))
        for sub in (0, 1, 2):
            for kw in (dict(restart_marker_blocks=1), dict(restart_marker_rows=1), dict(optimize=True),
                       dict(progressive=True)):
                b = io.BytesIO()
                Image.fromarray(a).save(b, "JPEG", quality=int(nrng.integers(5, 100)), subsampling=sub, **kw)
                seeds.append(b.getvalue())
    for k in range(10):                         # crafted streams with restart intervals
        samp = [[(1, 1)], [(2, 2), (1, 1), (1, 1)], [(2, 1), (1, 1), (1, 1)]][k % 3]
        W, H = int(nrng.integers(8, 40)), int(nrng.integers(8, 40))
        blocks = []
        for ci in range(len(samp)):
            g = grid(samp, W, H, ci)
            blocks.append(nrng.integers(-6, 7, g + (64,)) * (nrng.random(g + (64,)) < 0.2))
        seeds.append(write_jpeg(blocks, [nrng.integers(1, 17, 64) for _ in samp], samp, W, H, dri=k % 3))
    return seeds


def dir_seeds(d, max_bytes=300_000):
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if name.lower().endswith(".jpg") and os.path.getsize(p) <= max_bytes:
            out.append(open(p, "rb").read())
    return out


# ---------------- mutations ----------------

def segments(d):
    """(offset, marker, length) of the marker segments before the scan, and the scan's first byte."""
    out, i = [], 2
    while i + 4 <= len(d) and d[i] == 0xFF:
        m = d[i + 1]
        L = int.from_bytes(d[i + 2:i + 4], "big")
        out.append((i, m, L))
        if m == 0xDA:
            return out, min(len(d), i + 2 + L)
        i += 2 + L
    return out, len(d)


def mutate(d, rng):
    """A mutated copy of d. Never raises (a mutation that fails on an odd input returns a bit-flipped copy)."""
    try:
        return _mutate(d, rng)
    except Exception:                                            # noqa: BLE001
        d = bytearray(d)
        if d:
            d[rng.randrange(len(d))] ^= 1 << rng.randrange(8)
        return bytes(d)


def _mutate(d, rng):
    if len(d) < 8:                                     # too short to mutate meaningfully: grow it
        return bytes(d) + rng.randbytes(rng.randint(1, 8))
    d = bytearray(d)
    segs, s0 = segments(d)
    n = len(d)
    k = rng.randrange(20)
    if k == 0:                                         # flip bits in the scan
        for _ in range(rng.randint(1, 8)):
            if n > s0:
                d[rng.randrange(s0, n)] ^= 1 << rng.randrange(8)
    elif k == 1:                                       # random bytes anywhere
        for _ in range(rng.randint(1, 4)):
            d[rng.randrange(n)] = rng.randrange(256)
    elif k == 2:                                       # truncate anywhere
        d = d[:rng.randrange(2, n)]
    elif k == 3:                                       # truncate at a structural point (+- a few bytes)
        pts = [i for i, _, _ in segs] + [i + 4 for i, _, _ in segs] + [s0, n - 2, (s0 + n) // 2]
        cut = max(2, min(n - 1, rng.choice(pts) + rng.randint(-2, 2)))
        d = d[:cut]
    elif k == 4:                                       # insert bytes into the scan
        i = rng.randrange(s0, n + 1)
        d[i:i] = rng.choice([b"\xff", b"\xff\x00", bytes([0xFF, rng.randrange(256)]), b"\xff\xff\xff",
                             rng.randbytes(rng.randint(1, 16))])
    elif k == 5:                                       # delete a range
        i = rng.randrange(n)
        del d[i:i + rng.randint(1, 64)]
    elif k == 6:                                       # insert, remove or renumber restart markers
        idx = [i for i in range(s0, n - 1) if d[i] == 0xFF and 0xD0 <= d[i + 1] <= 0xD7]
        r = rng.random()
        if idx and r < 0.33:
            d[rng.choice(idx) + 1] = 0xD0 + rng.randrange(8)
        elif idx and r < 0.66:
            i = rng.choice(idx)
            del d[i:i + 2]
        else:
            i = rng.randrange(s0, n + 1)
            d[i:i] = bytes([0xFF, 0xD0 + rng.randrange(8)])
    elif k == 7:                                       # change a header field
        if segs:
            i, m, L = rng.choice(segs)
            j = i + 2 + rng.randrange(max(L, 1))
            if j < n:
                d[j] = rng.choice([0, 1, 2, 3, 4, 15, 16, 17, 0x11, 0x22, 0x41, 0x7F, 0x80, 0xFF, rng.randrange(256)])
    elif k == 8:                                       # random scan data
        d = d[:s0] + bytearray(rng.randbytes(rng.randint(1, 4096))) + b"\xff\xd9"
    elif k == 9:                                       # absurd or zero dimensions
        sofs = [s for s in segs if 0xC0 <= s[1] <= 0xCF and s[1] not in (0xC4, 0xC8, 0xCC)]
        if sofs:
            i = sofs[0][0]
            d[i + 5:i + 9] = rng.choice([b"\xff\xff\xff\xff", b"\x00\x00\x00\x10", b"\x00\x10\x00\x00",
                                         b"\x7f\xff\x7f\xff", b"\xff\xdc\xff\xdc", b"\x00\x01\xff\xdc",
                                         rng.randbytes(4)])
    elif k == 10:                                      # duplicate, drop or reorder segments
        if len(segs) >= 2:
            r = rng.random()
            i, m, L = rng.choice(segs[:-1] if r < 0.66 else segs)
            seg = d[i:i + 2 + L]
            if r < 0.33:
                d[i:i] = seg
            elif r < 0.66:
                del d[i:i + 2 + L]
            else:                                      # swap two neighbouring segments
                j = rng.randrange(len(segs) - 1)
                (a, _, La), (b, _, Lb) = segs[j], segs[j + 1]
                d[a:b + 2 + Lb] = d[b:b + 2 + Lb] + d[a:a + 2 + La]
    elif k == 11:                                      # drop the 0x00 after a 0xFF in the scan
        idx = [i for i in range(s0, n - 1) if d[i] == 0xFF and d[i + 1] == 0]
        if idx:
            del d[rng.choice(idx) + 1]
    elif k == 12:                                      # splice the scan
        if n - s0 > 8:
            a, b = sorted(rng.sample(range(s0, n), 2))
            d = d[:a] + d[b:]
    elif k == 13:                                      # zero-length / tiny / oversized segment length
        if segs:
            i, m, L = rng.choice(segs)
            d[i + 2:i + 4] = rng.choice([b"\x00\x00", b"\x00\x01", b"\x00\x02", b"\xff\xff",
                                         ((L + rng.randint(-3, 3)) & 0xFFFF).to_bytes(2, "big")])
    elif k == 14:                                      # wrong component counts (SOF Nf, SOS Ns)
        sel = [s for s in segs if s[1] in (0xC0, 0xC1, 0xDA)]
        if sel:
            i, m, L = rng.choice(sel)
            j = i + 4 + (5 if m != 0xDA else 0)
            if j < n:
                d[j] = rng.choice([0, 1, 2, 3, 4, 5, 255])
    elif k == 15:                                      # fill bytes before markers
        if segs:
            i = rng.choice(segs)[0]
            d[i:i] = b"\xff" * rng.randint(1, 5)
    elif k == 16:                                      # corrupt a DHT table body
        dht = [s for s in segs if s[1] == 0xC4]
        if dht:
            i, m, L = rng.choice(dht)
            for _ in range(rng.randint(1, 3)):
                j = i + 4 + rng.randrange(max(L - 2, 1))
                if j < n:
                    d[j] = rng.choice([0, 0xFF, 16, 17, rng.randrange(256)])
    elif k == 17:                                      # change the restart interval
        dri = [s for s in segs if s[1] == 0xDD]
        if dri:
            i = dri[0][0]
            d[i + 4:i + 6] = rng.choice([b"\x00\x00", b"\x00\x01", b"\xff\xff", rng.randbytes(2)])
        else:
            d[2:2] = b"\xff\xdd\x00\x04" + rng.choice([b"\x00\x01", b"\x00\x05", b"\xff\xff"])
    elif k == 18:                                      # a marker in the middle of the scan
        i = rng.randrange(s0, n + 1)
        d[i:i] = bytes([0xFF, rng.choice([0xD9, 0xDA, 0xC4, 0xDB, 0xE1, 0xFE, 0xDD, 0x01])]) + rng.randbytes(rng.randint(0, 6))
    else:                                              # several mutations
        for _ in range(rng.randint(2, 5)):
            d = bytearray(_mutate(bytes(d), rng))
    return bytes(d)


# ---------------- worker ----------------

def write_status(path, snap, final=False):
    """Atomic replace; on Windows it fails while the parent has the file open, so retry (always, for the last)."""
    with open(path + ".tmp", "w") as f:
        json.dump(snap, f)
    for k in range(200 if final else 5):
        try:
            os.replace(path + ".tmp", path)
            return
        except PermissionError:
            time.sleep(0.05)


def reference(data):
    """libjpeg-turbo's image: simplejpeg (strict=False), or Pillow (also libjpeg-turbo, same defaults) when
    simplejpeg raises on a harmless header warning (e.g. an unknown JFIF version). None if libjpeg-turbo rejects
    the file (see _pillow_reference: Pillow's image is not taken for a file libjpeg-turbo fails on)."""
    import simplejpeg
    try:
        return simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False, strict=False)
    except Exception:                                            # noqa: BLE001
        pass
    return _pillow_reference(data, "RGB")


def strict_refuses(data):
    """True if libjpeg-turbo's strict decode (simplejpeg, strict=True) refuses the file for what libjpeg reports
    (a warning or an error), not for a limit of TurboJPEG's API (a sampling layout it has no name for). A file
    the GPU path decodes under strict=True must not be one of these."""
    import simplejpeg
    try:
        simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False)
        return False
    except Exception as e:                                       # noqa: BLE001
        return "subsampling level" not in str(e)


def _pillow_reference(data, mode):
    """Pillow's decode (libjpeg-turbo inside) with Pillow's own default for damaged files, so that a file
    libjpeg-turbo rejects (a bad table after the scan, a second scan) is None here. Only when Pillow's complaint
    is that the data ends early is it decoded again with LOAD_TRUNCATED_IMAGES: libjpeg-turbo treats a missing
    end (EOI) as a warning and supplies it. The switch is put back as it was."""
    import io
    import numpy as np
    from PIL import Image, ImageFile
    was = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        for truncated in (False, True):
            ImageFile.LOAD_TRUNCATED_IMAGES = truncated
            try:
                with Image.open(io.BytesIO(data), formats=("JPEG", "MPO")) as im:
                    if mode == "L":
                        im.draft("L", im.size)
                    return np.asarray(im.convert(mode))
            except OSError as e:
                if truncated or "truncated" not in str(e):
                    return None
            except Exception:                                    # noqa: BLE001
                return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = was
    return None


def worker(out_dir, start, iters, seed, threads, seeds_dir):
    import cupy as cp
    import numpy as np

    import exact_gpu_jpeg as egj
    seeds = make_seeds(random.Random(seed))
    if seeds_dir:
        seeds += dir_seeds(seeds_dir)
    # known-good files and their references, for the leak check
    goods = [(d, reference(d)) for d in seeds[::7]]
    goods = [(d, r) for d, r in goods if r is not None and egj.inspect(d)["supported"]][:8]
    c = collections.Counter()
    reasons = collections.Counter()
    lock = threading.Lock()
    worst = [0.0]
    fdir = os.path.join(out_dir, "findings")
    os.makedirs(fdir, exist_ok=True)
    status = os.path.join(out_dir, "status.json")
    stop = threading.Event()
    done_it = [start]

    def finding(kind, data, note, it):
        with lock:
            c[kind] += 1
            p = os.path.join(fdir, f"{kind}_{seed}_{it}.jpg")
            with open(p, "wb") as f:
                f.write(data)
            with open(os.path.join(fdir, "notes.txt"), "a") as f:
                f.write(f"{os.path.basename(p)}\t{note}\n")

    def run(t):
        try:
            run_inner(t)
        except Exception as e:                                   # noqa: BLE001  (a fuzzer bug, not a finding)
            with lock:
                c["fuzzer_errors"] += 1
            with open(os.path.join(out_dir, "fuzzer_errors.txt"), "a") as f:
                f.write(repr(e) + "\n")
            stop.set()

    def run_inner(t):
        dec = egj.Decoder()
        rng = random.Random((seed * 1000003 + start) * 31 + t)
        for j in range(t, iters, threads):
            if stop.is_set():
                return
            it = start + j
            data = mutate(rng.choice(seeds), rng)
            with lock:
                c["mutants"] += 1
            with open(os.path.join(out_dir, f"current_{t}.jpg"), "wb") as f:
                f.write(data)
            t0 = time.perf_counter()
            try:
                out = dec.decode(data)
            except RuntimeError as e:
                finding("canary" if "outside device buffer" in str(e) else "exception", data, repr(e), it)
                continue
            except Exception as e:                               # noqa: BLE001
                finding("exception", data, repr(e), it)
                continue
            try:
                cp.cuda.runtime.deviceSynchronize()
            except Exception as e:                               # noqa: BLE001
                finding("cuda_error", data, repr(e), it)
                stop.set()
                return
            dt = time.perf_counter() - t0
            with lock:
                worst[0] = max(worst[0], dt)
                c["inputs"] += 1
                done_it[0] = max(done_it[0], it)
            if out is None:
                with lock:
                    c["declined"] += 1
                    reasons[(dec.reason or "?").split(";")[0].split(" (")[0]] += 1
                g, r = goods[j % len(goods)]                     # leak check: a good file right after a bad one
                o = dec.decode(g)
                if o is None or not np.array_equal(o.get(), r):
                    finding("leak", data, f"good file failed after this input: {dec.reason}", it)
                else:
                    with lock:
                        c["leak_checks_ok"] += 1
                continue
            gimg = out.get()
            ref = reference(data)
            if ref is None:
                finding("mismatch", data, "GPU decoded a file libjpeg-turbo rejects", it)
            elif gimg.shape != ref.shape or not np.array_equal(gimg, ref):
                finding("mismatch", data, "differs from libjpeg-turbo", it)
            elif strict_refuses(data):                           # the default decoder is strict
                finding("mismatch", data, "GPU decoded under strict=True a file libjpeg-turbo warns about", it)
            else:
                with lock:
                    c["gpu_exact"] += 1
            if j % 10 == 0:
                o2 = dec.decode(data)
                if o2 is None or not np.array_equal(o2.get(), gimg):
                    finding("nondeterministic", data, "", it)

    ts = [threading.Thread(target=run, args=(t,), daemon=True) for t in range(threads)]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        time.sleep(1)
        with lock:
            snap = dict(it=done_it[0], counts=dict(c), reasons=reasons.most_common(60), worst_s=worst[0])
        write_status(status, snap)
    with lock:
        snap = dict(it=start + iters, counts=dict(c), reasons=reasons.most_common(60), worst_s=worst[0],
                    done=not stop.is_set() and c["mutants"] == iters)
    write_status(status, snap, final=True)


# ---------------- parent ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--seeds-dir")
    ap.add_argument("--worker", nargs=2, type=int)        # internal: start iters
    a = ap.parse_args()
    if a.worker:
        worker(a.out_dir, a.worker[0], a.worker[1], a.seed, a.threads, a.seeds_dir)
        return
    os.makedirs(os.path.join(a.out_dir, "findings"), exist_ok=True)
    status = os.path.join(a.out_dir, "status.json")
    total, reasons = collections.Counter(), collections.Counter()
    worst = 0.0
    start = 0
    t0 = time.time()
    while start < a.iters:
        if os.path.exists(status):
            os.remove(status)
        cmd = [sys.executable, __file__, a.out_dir, "--seed", str(a.seed), "--threads", str(a.threads),
               "--worker", str(start), str(a.iters - start)]
        if a.seeds_dir:
            cmd += ["--seeds-dir", a.seeds_dir]
        p = subprocess.Popen(cmd)
        last, last_t, killed = None, time.time(), False
        while p.poll() is None:
            time.sleep(2)
            try:
                st = json.load(open(status))
            except (OSError, ValueError):
                st = None
            key = (st["it"], st["counts"].get("inputs")) if st else None
            if key != last:
                last, last_t = key, time.time()
            elif time.time() - last_t > a.timeout:
                p.kill()
                p.wait()
                killed = True
                break
        st = None
        try:
            st = json.load(open(status))
        except (OSError, ValueError):
            pass
        if st:
            total.update(st["counts"])
            reasons.update(dict(st["reasons"]))
            worst = max(worst, st["worst_s"])
        if st and st.get("done"):
            break
        if st and st["counts"].get("fuzzer_errors") and not killed:
            kind = "fuzzer_error"                                # the fuzzer's own bug (see fuzzer_errors.txt)
        else:
            kind = "hang" if killed else "crash"                 # keep the inputs in flight, continue after them
        total[kind] += 1
        for t in range(a.threads):
            cur = os.path.join(a.out_dir, f"current_{t}.jpg")
            if os.path.exists(cur):
                os.replace(cur, os.path.join(a.out_dir, "findings", f"{kind}_{a.seed}_{start}_t{t}.jpg"))
        start = (st["it"] if st else start) + 2 * a.threads
    summary = dict(iterations=a.iters, seed=a.seed, threads=a.threads, seeds_dir=bool(a.seeds_dir),
                   canary=os.environ.get("EGJ_CANARY") == "1", counts=dict(total),
                   declined_reasons=dict(reasons.most_common()), worst_decode_s=round(worst, 3),
                   wall_s=round(time.time() - t0))
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    found = sum(total.get(k, 0) for k in ("exception", "mismatch", "nondeterministic", "leak", "cuda_error", "canary", "crash", "hang", "fuzzer_error"))
    return 1 if found or not total.get("inputs") else 0      # (a run that tried nothing is not a pass either)


if __name__ == "__main__":
    sys.exit(main())
