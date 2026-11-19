"""nvJPEG through its own C API (ctypes on the nvjpeg DLL that ships with PyTorch), every back end this GPU accepts,
against exact-gpu-jpeg and against torchvision's route to nvJPEG, on the same files in the same process.

usage: python bench/nvjpeg_direct.py <out.json> <dir of photos> [--sizes 0.25,0.5,1,2,6] [--n 64]

Needs torch (for its nvjpeg DLL) and torchvision in the environment. Nothing is downloaded.

Files: as bench/batch_bench.py (n different JPEGs per size from the photos, quality 90, 4:2:0), plus the photos
themselves one at a time. Timed: from the JPEG bytes in host memory to RGB pixels in GPU memory, stream synchronised.
nvJPEG decodes into buffers allocated beforehand and its batched decoder is initialised once before the timing
(both in its favour); exact-gpu-jpeg allocates inside the call.
 - nvjpegDecode: one call per image, the stream waited for after each (as torchvision does; a state reused before
   its last image has left the GPU gives wrong pixels); the same with 8 states in turn and a wait every 8 images;
   nvjpegDecodeBatched: one call for the whole set (8 CPU threads allowed);
 - back ends tried: DEFAULT, HYBRID (Huffman on the CPU), GPU_HYBRID (Huffman on the GPU), HARDWARE; those the GPU
   refuses are recorded with nvJPEG's status code;
 - libjpeg-turbo (simplejpeg, accurate IDCT, fancy upsampling) on 1 and on 8 CPU threads, to host memory, in the
   same rounds;
 - every nvJPEG result is compared with libjpeg-turbo's pixels (simplejpeg, accurate IDCT, fancy upsampling); a
   result more than 3 levels off on average is not a decode and gets no speed ("wrong pixels").
Rounds alternate the order. Counts and timings only (no file names).
"""
import ctypes as C
import glob
import io
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simplejpeg
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cupy as cp  # noqa: E402
import torch  # noqa: E402
from torchvision.io import decode_jpeg  # noqa: E402

import exact_gpu_jpeg as egj  # noqa: E402

BACKENDS = {"DEFAULT": 0, "HYBRID": 1, "GPU_HYBRID": 2, "HARDWARE": 3}
RGBI = 5                                   # NVJPEG_OUTPUT_RGBI: interleaved RGB in channel[0]
MIN_S = 1.5
WRONG = 3.0                                # mean levels off libjpeg-turbo beyond which a result is not a decode


class Img(C.Structure):                    # nvjpegImage_t
    _fields_ = [("channel", C.c_void_p * 4), ("pitch", C.c_size_t * 4)]


def load_nvjpeg():
    import torchvision
    dirs = [os.path.dirname(torchvision.__file__), os.path.join(os.path.dirname(torch.__file__), "lib")]
    names = []
    for lib in dirs:                       # the library torchvision itself loads (its own folder on Windows)
        if hasattr(os, "add_dll_directory") and os.path.isdir(lib):
            os.add_dll_directory(lib)
        names += sorted(glob.glob(os.path.join(lib, "nvjpeg64_*.dll")) + glob.glob(os.path.join(lib, "libnvjpeg.so*")))
    names = names[:1]
    if not names:
        raise SystemExit("no nvjpeg library next to torchvision or torch")
    nv = C.CDLL(names[-1])
    vp, vpp = C.c_void_p, C.POINTER(C.c_void_p)
    nv.nvjpegGetProperty.argtypes = [C.c_int, C.POINTER(C.c_int)]
    nv.nvjpegCreateEx.argtypes = [C.c_int, vp, vp, C.c_uint, vpp]
    nv.nvjpegJpegStateCreate.argtypes = [vp, vpp]
    nv.nvjpegDecode.argtypes = [vp, vp, vp, C.c_size_t, C.c_int, C.POINTER(Img), vp]
    nv.nvjpegDecodeBatchedInitialize.argtypes = [vp, vp, C.c_int, C.c_int, C.c_int]
    nv.nvjpegDecodeBatched.argtypes = [vp, vp, C.POINTER(vp), C.POINTER(C.c_size_t), C.POINTER(Img), vp]
    nv.nvjpegJpegStateDestroy.argtypes = [vp]
    nv.nvjpegDestroy.argtypes = [vp]
    for f in (nv.nvjpegGetProperty, nv.nvjpegCreateEx, nv.nvjpegJpegStateCreate, nv.nvjpegDecode,
              nv.nvjpegDecodeBatchedInitialize, nv.nvjpegDecodeBatched, nv.nvjpegJpegStateDestroy, nv.nvjpegDestroy):
        f.restype = C.c_int
    ver = []
    for k in range(3):
        v = C.c_int(-1)
        nv.nvjpegGetProperty(k, C.byref(v))
        ver.append(v.value)
    return nv, ".".join(map(str, ver)), os.path.basename(names[-1])


def rate(fn, n):
    fn()
    t0, k = time.perf_counter(), 0
    while k < 2 or time.perf_counter() - t0 < MIN_S:
        fn()
        k += 1
    return k * n / (time.perf_counter() - t0)


def make_set(photos, mp, n):
    w = int(round((mp * 1e6 * 1.5) ** 0.5 / 8)) * 8
    h = int(round(w / 1.5 / 8)) * 8
    data = []
    for k in range(n):
        im = photos[k % len(photos)]
        if k >= len(photos):
            s = 1 - 0.04 * (k // len(photos))
            im = im.crop((0, 0, int(im.width * s), int(im.height * s)))
        b = io.BytesIO()
        im.resize((w, h), Image.LANCZOS).save(b, "JPEG", quality=90, subsampling=2)
        data.append(b.getvalue())
    return data


def free_gpu_caches():
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()


def measure(nv, backends, data, stream, dec, batched):
    """One set, three rounds. In each round: nvJPEG (one back end loaded at a time, so the libraries never crowd
    each other out of the 8 GB), then torchvision, then exact-gpu-jpeg, then libjpeg-turbo on the CPU; the order
    flips every round. A decode that
    fails stops the run (a declined image must never count as a fast one)."""
    n = len(data)
    ref = [simplejpeg.decode_jpeg(x, fastdct=False, fastupsample=False) for x in data]
    keep = [C.c_char_p(x) for x in data]
    ptrs = (C.c_void_p * n)(*[C.cast(p, C.c_void_p) for p in keep])
    lens = (C.c_size_t * n)(*[len(x) for x in data])
    sp = C.c_void_p(stream.ptr)
    res, rates = {}, {}

    def note(name, extra=None, r=None):
        res.setdefault(name, {}).update(extra or {})
        if r is not None:
            rates.setdefault(name, []).append(r)

    def nvjpeg_phase():
        with stream:                                            # nvJPEG's output buffers, made beforehand
            outs = [cp.empty(r.shape, cp.uint8) for r in ref]
        imgs = (Img * n)()
        for k, o in enumerate(outs):
            imgs[k].channel[0] = o.data.ptr
            imgs[k].pitch[0] = o.shape[1] * 3

        def clear():
            with stream:                                        # on the decode's own stream, and finished
                for o in outs:
                    o.fill(0)
            stream.synchronize()

        def diff():
            d = [np.abs(o.get(stream=stream).astype(np.int16) - r) for o, r in zip(outs, ref)]
            return dict(max_abs_diff=int(max(int(x.max()) for x in d)),
                        mean_abs_diff=round(float(np.mean([x.mean() for x in d])), 3),
                        identical_images=int(sum(int(x.max()) == 0 for x in d)))
        for name, b in backends.items():
            h, st, stb = C.c_void_p(), C.c_void_p(), C.c_void_p()
            s = nv.nvjpegCreateEx(b, None, None, 0, C.byref(h))
            if s == 0:
                s = nv.nvjpegJpegStateCreate(h, C.byref(st)) or nv.nvjpegJpegStateCreate(h, C.byref(stb))
            if s:
                note(f"nvjpegDecode {name}", dict(status=int(s)))
                continue

            def single():                                       # one state, so the stream is waited for after
                for k in range(n):                              # every image: a state reused while its last image
                    s = nv.nvjpegDecode(h, st, ptrs[k], lens[k], RGBI, C.byref(imgs[k]), sp)   # is still on the
                    if s:                                       # GPU gives wrong pixels (mean 50+ levels off)
                        raise RuntimeError(s)
                    stream.synchronize()

            def several():                                      # 8 states in turn, the stream waited for after
                for k in range(n):                              # every 8 images: no state is reused while busy
                    s = nv.nvjpegDecode(h, sts[k % 8], ptrs[k], lens[k], RGBI, C.byref(imgs[k]), sp)
                    if s:
                        raise RuntimeError(s)
                    if k % 8 == 7:
                        stream.synchronize()
                stream.synchronize()

            def batch():
                s = nv.nvjpegDecodeBatched(h, stb, ptrs, lens, imgs, sp)
                if s:
                    raise RuntimeError(s)
                stream.synchronize()
            sts = [C.c_void_p() for _ in range(8)]
            s8 = 0
            for x in sts:
                s8 = s8 or nv.nvjpegJpegStateCreate(h, C.byref(x))
            jobs = [(f"nvjpegDecode {name}", single, 0), (f"nvjpegDecode {name}, 8 states", several, s8)]
            if batched:
                jobs.append((f"nvjpegDecodeBatched {name}", batch, nv.nvjpegDecodeBatchedInitialize(h, stb, n, 8, RGBI)))
            for jn, fn, s in jobs:
                try:
                    if s:
                        raise RuntimeError(s)
                    clear()
                    fn()
                    d = diff()
                    note(jn, d)
                    if d["mean_abs_diff"] > WRONG:              # not a decode of these files: no speed is recorded
                        note(jn, dict(status="wrong pixels"))
                        continue
                    note(jn, r=rate(fn, n))
                except RuntimeError as e:
                    note(jn, dict(status=int(e.args[0])))
            for x in [st, stb] + sts:
                if x:
                    nv.nvjpegJpegStateDestroy(x)
            nv.nvjpegDestroy(h)
        del outs, imgs

    def torchvision_phase():
        tens = [torch.frombuffer(bytearray(x), dtype=torch.uint8) for x in data]

        def tv_single():
            for t in tens:
                decode_jpeg(t, device="cuda")
            torch.cuda.synchronize()

        def tv_batch():
            decode_jpeg(tens, device="cuda")
            torch.cuda.synchronize()
        note("torchvision decode_jpeg, one call per image", r=rate(tv_single, n))
        if batched:
            note("torchvision decode_jpeg, batched", r=rate(tv_batch, n))

    def ours_phase():
        def ours_single():
            for x in data:
                if dec.decode(x) is None:
                    raise SystemExit("exact-gpu-jpeg declined a file: " + str(dec.reason))
            dec.stream.synchronize()

        def ours_batch():
            if any(g is None for g in dec.decode_batch(data)):
                raise SystemExit("exact-gpu-jpeg declined a file in a batch: " + str(dec.reasons))
            dec.stream.synchronize()
        exact = 0
        for a0 in range(0, n, 16 if batched else 1):             # (in pieces: the big photos would not fit at once)
            got = dec.decode_batch(data[a0:a0 + 16]) if batched else [dec.decode(data[a0])]
            exact += int(sum(g is not None and np.array_equal(g.get(), r) for g, r in zip(got, ref[a0:])))
            del got
        note("exact-gpu-jpeg decode, one call per image", dict(identical_images=exact), rate(ours_single, n))
        if batched:
            note("exact-gpu-jpeg decode_batch", dict(identical_images=exact), rate(ours_batch, n))
    def cpu_phase():
        def one(x):
            return simplejpeg.decode_jpeg(x, fastdct=False, fastupsample=False)

        def cpu1():
            for x in data:
                one(x)
        note("libjpeg-turbo, 1 thread", r=rate(cpu1, n))
        if batched:
            with ThreadPoolExecutor(8) as ex:
                note("libjpeg-turbo, 8 threads", r=rate(lambda: list(ex.map(one, data)), n))
    phases = [nvjpeg_phase, torchvision_phase, ours_phase, cpu_phase]
    for rnd in range(3):
        for ph in (phases if rnd % 2 == 0 else phases[::-1]):
            free_gpu_caches()
            ph()
    for k, v in rates.items():
        res[k]["images_per_s"] = round(statistics.median(v), 1)
        res[k]["rounds"] = [round(x, 1) for x in v]
    return res


def main():
    out, d = sys.argv[1], sys.argv[2]
    a = sys.argv[3:]
    sizes = [float(x) for x in a[a.index("--sizes") + 1].split(",")] if "--sizes" in a else [0.25, 0.5, 1, 2, 6]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 64
    files = sorted({os.path.normcase(f): f for f in glob.glob(os.path.join(d, "*.jpg"))}.values())
    nv, ver, dll = load_nvjpeg()
    cp.cuda.Device().use()
    stream = cp.cuda.Stream(non_blocking=True)
    accepted, refused = {}, {}
    for name, b in BACKENDS.items():                             # which back ends does this GPU accept?
        h = C.c_void_p()
        s = nv.nvjpegCreateEx(b, None, None, 0, C.byref(h))
        if s == 0:
            accepted[name] = b
            nv.nvjpegDestroy(h)
        else:
            refused[name] = s
    handles = accepted
    print("nvJPEG", ver, dll, "| back ends accepted:", list(handles), "| refused (status):", refused, flush=True)
    dev = cp.cuda.runtime.getDeviceProperties(0)
    res = dict(env=dict(gpu=dev["name"].decode(), nvjpeg=ver, nvjpeg_library=dll, torch=torch.__version__,
                        torchvision=__import__("torchvision").__version__, cupy=cp.__version__,
                        simplejpeg=simplejpeg.__version__, exact_gpu_jpeg=egj.__version__),
               measured=time.strftime("%Y-%m-%d"), backends_accepted=list(handles), backends_refused=refused,
               images_per_set=n, sizes={}, sets={})
    dec = egj.Decoder()
    photos = [Image.open(f).convert("RGB") for f in files]
    for mp in sizes:
        r = measure(nv, handles, make_set(photos, mp, n), stream, dec, True)
        res["sets"][f"{mp:g}mp"] = r
        w = int(round((mp * 1e6 * 1.5) ** 0.5 / 8)) * 8                # as make_set
        res["sizes"][f"{mp:g}mp"] = [w, int(round(w / 1.5 / 8)) * 8]
        print(f"{mp:g} MP", json.dumps({k: (v.get("images_per_s"), v.get("max_abs_diff"), v.get("status")) for k, v in r.items()}),
              flush=True)
    del photos
    cam = [open(f, "rb").read() for f in files]
    r = measure(nv, handles, cam, stream, dec, False)
    res["camera_photos"] = dict(files=len(cam), results=r)
    print("camera", json.dumps({k: (v.get("images_per_s"), v.get("max_abs_diff"), v.get("status")) for k, v in r.items()}),
          flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
