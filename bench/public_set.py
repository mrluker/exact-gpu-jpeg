"""The public image set: what the library does on images anyone can download, with no file in this repository.

The set is imagecompression.info's 8-bit RGB test images (rgb8bit.zip, 14 photos of 3.4 to 39 MP) and
libjpeg-turbo's three benchmark frames (3dtestimages.tar.bz2, 1.2 MP rendered frames). The script downloads the
two archives from their sources into a cache folder (or uses the ones already there), checks their SHA-256, and
makes every JPEG in memory with libjpeg-turbo through Pillow: quality 95 at 4:2:0, 4:2:2, 4:4:4 and gray
(the settings of libjpeg-turbo's own performance page), quality 90 at 4:2:0, and quality 95 at 4:2:0 with
optimised Huffman tables. 17 images x 6 settings = 102 JPEGs.

For every JPEG:
  decode: the route egj.decode takes, the synchronisation passes, and whether the GPU decoder's pixels equal
    libjpeg-turbo's (accurate IDCT, fancy upsampling; simplejpeg), byte for byte;
  encode (colour, standard tables): whether the library's file equals Pillow's, byte for byte;
  the packed copy: a decode from it against the same pixels, and its size;
  other decoders, when they are installed (torchvision for nvJPEG and its CPU decoder, Pillow, OpenCV, libvips,
    libjpeg-turbo's fast modes): files identical to libjpeg-turbo, and for the rest the largest difference in
    levels, the share of pixels that differ and the PSNR against the reference;
  nvJPEG's encoder through torchvision, when installed, at quality 95, 4:4:4: file size and PSNR against the
    source image next to libjpeg-turbo's, and whether the bytes are equal;
  decode times, ms for one image from bytes in memory, median of 9 calls after at least 2 calls and 0.3 s of
    calls back to back to warm up, each waited for:
    this library (egj.decode, its default route), libjpeg-turbo on one thread (simplejpeg, accurate IDCT, fancy
    upsampling), and nvJPEG through torchvision when installed. Run it alone on the machine (it times).

usage: python bench/public_set.py <out.json> [--cache DIR] [--no-download]
       python bench/public_set.py --refit <out.json>     the per-MB fit again from the file's rows (no timing)
The cache folder (default: public-images in the system's temporary folder) holds the archives and the images;
it is never inside the repository. The images' licence notice (readme.txt in rgb8bit.zip) stays with them there.
"""
import glob
import hashlib
import io
import json
import os
import sys
import statistics
import tarfile
import tempfile
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import simplejpeg  # noqa: E402
import cupy as cp  # noqa: E402
import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ARCHIVES = [
    dict(folder="imagecompression_info", name="rgb8bit.zip", bytes=346_126_330,
         url="https://imagecompression.info/test_images/rgb8bit.zip",
         sha256="16c273504425923259b1ddf8e39dfa9cb27585d249f6263cfc6bea39cfafa660", images=14),
    dict(folder="libjpeg_turbo_3dtest", name="3dtestimages.tar.bz2", bytes=764_181,
         url="https://libjpeg-turbo.org/pmwiki/uploads/About/3dtestimages.tar.bz2",
         sha256="60e5cac0da25712a9b49da60212f14266d07c586dfb5d6c968e34fec6937b667", images=3),
]
SETTINGS = [("q95_420", 95, 2, False, False), ("q95_422", 95, 1, False, False), ("q95_444", 95, 0, False, False),
            ("q95_gray", 95, None, False, True), ("q90_420", 90, 2, False, False), ("q95_420_opt", 95, 2, True, False)]
SUB = {0: "444", 1: "422", 2: "420"}
PUBLISHED = ("artificial", "nightshot_iso_100", "vgl_5674_0098", "vgl_6434_0018", "vgl_6548_0026")   # libjpeg-turbo's five


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def fetch(cache, download):
    """The PPM files of both archives, checked and unpacked in the cache."""
    files = []
    for a in ARCHIVES:
        d = os.path.join(cache, a["folder"])
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, a["name"])
        if not os.path.exists(p):
            if not download:
                raise SystemExit(f"{p} is not there and --no-download was given (source: {a['url']}, {a['bytes']:,} bytes)")
            print(f"downloading {a['url']} ({a['bytes'] / 1e6:.0f} MB) ...", flush=True)
            urllib.request.urlretrieve(a["url"], p + ".part")
            os.replace(p + ".part", p)
        got = sha256_of(p)
        if got != a["sha256"]:
            raise SystemExit(f"{p}: SHA-256 {got}, expected {a['sha256']}: not the archive this script was written for")
        have = sorted(glob.glob(os.path.join(d, "*.ppm")))
        if len(have) != a["images"]:
            if a["name"].endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    for m in z.namelist():
                        if m.lower().endswith((".ppm", ".txt")) and os.path.basename(m) == m:
                            z.extract(m, d)
            else:
                with tarfile.open(p) as t:
                    for m in t.getmembers():
                        if m.isfile() and m.name.lower().endswith(".ppm") and os.path.basename(m.name) == m.name:
                            t.extract(m, d)
            have = sorted(glob.glob(os.path.join(d, "*.ppm")))
        assert len(have) == a["images"], (a["name"], len(have))
        files += have
    return files


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse == 0 else round(10 * np.log10(255.0 ** 2 / mse), 2)


def compare(got, want):
    if got is None:
        return dict(decoded=False)
    if got.shape != want.shape:
        return dict(same=False, shape=list(got.shape))
    d = np.abs(got.astype(np.int16) - want.astype(np.int16))
    mx = int(d.max())
    if mx == 0:
        return dict(same=True)
    return dict(same=False, max_diff=mx, pixels_differ=round(float((d.max(axis=2) > 0).mean()), 5), psnr=psnr(got, want))


PER_MB_SETS = {"4:2:0": ("q95_420", "q90_420", "q95_420_opt"), "4:2:2": ("q95_422",), "4:4:4": ("q95_444",)}


def per_mb_fit(rows):
    """For each decoder and subsampling: ms = a + b x megapixels + c x coded MB (least squares over the images of 3 MP
    and more), c and b with their standard errors: what a megabyte of entropy-coded data and a megapixel cost."""
    out = {}
    for sub, sets in PER_MB_SETS.items():
        rs = [r for r in rows if r["setting"] in sets and r["megapixels"] >= 3 and "ms" in r]
        if len(rs) < 4:
            continue
        X = np.array([[1.0, r["megapixels"], r["bytes_per_pixel"] * r["megapixels"]] for r in rs])
        for dn in rs[0]["ms"]:
            y = np.array([r["ms"][dn] for r in rs])
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            res = y - X @ c
            se = np.sqrt(np.diag((res @ res) / (len(y) - 3) * np.linalg.inv(X.T @ X)))
            out.setdefault(dn, {})[sub] = dict(ms_per_coded_mb=round(float(c[2]), 2), se=round(float(se[2]), 2),
                                               ms_per_megapixel=round(float(c[1]), 2), se_mp=round(float(se[1]), 2),
                                               jpegs=len(y))
    return out


WARM_S = 0.3                     # every decoder is called back to back for this long before it is timed


def ms_of(f, n=9, warm=2, warm_s=WARM_S):
    """Median ms of n calls of f, after at least `warm` calls and `warm_s` seconds of calls back to back. The
    same for every decoder. Between rows the GPU idles for seconds while the CPU decoders and the comparisons
    run, and an idle card drops to a lower clock that takes 135 to 165 ms of work to leave
    (bench/results/20261118/idle_gpu.json); two warm calls were not enough (2026-11-18: our column moved by up to
    2x between two runs while libjpeg-turbo's stayed within 1 %, median)."""
    t_end, k = time.perf_counter() + warm_s, 0
    while k < warm or time.perf_counter() < t_end:
        f()
        k += 1
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 3)


def other_decoders():
    """{name: function(bytes) -> RGB array} for what is installed, and the GPU encoder if there is one."""
    out, enc, versions = {}, None, {}
    out["Pillow"] = lambda data: np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    out["libjpeg-turbo fast modes (fast IDCT + fast upsampling)"] = \
        lambda data: simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=True, fastupsample=True)
    try:
        import torch
        import torchvision
        from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg
        versions.update(torch=torch.__version__, torchvision=torchvision.__version__)

        def tv_cpu(data):
            return decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB).permute(1, 2, 0).numpy()
        out["torchvision.io.decode_jpeg on the CPU"] = tv_cpu
        if torch.cuda.is_available():
            def nv(data):
                x = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB, device="cuda")
                torch.cuda.synchronize()
                return x.permute(1, 2, 0).cpu().numpy()
            out["nvJPEG (torchvision.io.decode_jpeg, device='cuda', defaults)"] = nv

            def enc(rgb, quality):
                t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous().cuda()
                return bytes(encode_jpeg(t, quality=quality).cpu().numpy().tobytes())
    except ImportError:
        pass
    try:
        import cv2
        versions["opencv"] = cv2.__version__
        out[f"OpenCV imdecode"] = lambda data: cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    except ImportError:
        pass
    try:
        import pyvips
        versions["pyvips"] = pyvips.__version__

        def vips(data):
            im = pyvips.Image.new_from_buffer(data, "")
            a = np.ndarray(buffer=im.write_to_memory(), dtype=np.uint8, shape=[im.height, im.width, im.bands])
            return a if im.bands == 3 else np.repeat(a, 3, 2)
        out["libvips (pyvips)"] = vips
    except ImportError:
        pass
    return out, enc, versions


def main():
    if sys.argv[1] == "--refit":
        res = json.load(open(sys.argv[2]))
        res["summary"]["ms_per_coded_mb"] = per_mb_fit(res["files"])
        with open(sys.argv[2], "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res["summary"]["ms_per_coded_mb"], indent=1))
        return
    out_path = sys.argv[1]
    args = sys.argv[2:]
    cache = args[args.index("--cache") + 1] if "--cache" in args else os.path.join(tempfile.gettempdir(), "public-images")
    files = fetch(cache, "--no-download" not in args)
    egj.self_check()
    others, gpu_encode, versions = other_decoders()
    nv_time = None
    try:
        import torch
        from torchvision.io import ImageReadMode, decode_jpeg
        if torch.cuda.is_available():
            def nv_time(data):                                 # (the decode alone, waited for; no copy back)
                decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB, device="cuda")
                torch.cuda.synchronize()
    except ImportError:
        pass
    versions.update(pillow=Image.__version__, simplejpeg=simplejpeg.__version__, exact_gpu_jpeg=egj.__version__)
    res = dict(measured=time.strftime("%Y-%m-%d"), gpu=egj.device_report()["gpu"], versions=versions,
               archives=[{k: a[k] for k in ("name", "url", "bytes", "sha256")} for a in ARCHIVES],
               settings=[s[0] for s in SETTINGS], other_decoders=list(others), files=[], nvjpeg_encode=[])
    dec = egj.Decoder()
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        im = Image.open(path).convert("RGB")
        src = np.asarray(im)
        gim = None
        for tag, q, sub, opt, gray in SETTINGS:
            kw = dict(quality=q, optimize=opt)
            if not gray:
                kw["subsampling"] = sub
            b = io.BytesIO()
            (im.convert("L") if gray else im).save(b, "JPEG", **kw)
            data = b.getvalue()
            want = simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False)
            hd = egj.parse(np.frombuffer(data, np.uint8))
            scan = len(data) - hd.scan_start - 2
            row = dict(image=name, setting=tag, megapixels=round(im.size[0] * im.size[1] / 1e6, 2),
                       bytes_per_pixel=round(scan / (im.size[0] * im.size[1]), 4), blocks=D.layout(hd)["nblk"])
            info = {}
            egj.decode(data, info=info)
            row["route"] = info.get("path")
            dev = cp.cuda.Device()

            def ours():
                egj.decode(data)
                dev.synchronize()
            row["ms"] = {"this library": ms_of(ours),
                         "libjpeg-turbo, one thread": ms_of(lambda: simplejpeg.decode_jpeg(
                             data, colorspace="RGB", fastdct=False, fastupsample=False))}
            if nv_time is not None:
                row["ms"]["nvJPEG (torchvision)"] = ms_of(lambda: nv_time(data))
            dec.keep_coefs = "packed"
            img = dec.decode(data)
            row.update(gpu_decoded=img is not None, passes=dec.stats.get("passes") if img is not None else None,
                       same_pixels=bool(img is not None and np.array_equal(cp.asnumpy(img), want)))
            h = dec.take_coefs()
            dec.keep_coefs = None
            if h is not None:
                row["from_packed_same"] = bool(h.form == "packed" and np.array_equal(cp.asnumpy(dec.decode_from_handle(h)), want))
                if h.form == "packed":
                    dl = h._parts[2]
                    body = 4 * row["blocks"] + (int(dl.get()) if hasattr(dl, "get") else int(dl))
                    row.update(packed_bytes_per_block=round(body / row["blocks"], 1), packed_against_scan=round(body / scan, 3),
                               wide_blocks=round(float((h._parts[0] >> 31).sum().get()) / row["blocks"], 4))
                h.free()
            if not gray and not opt:
                if gim is None:
                    gim = cp.asarray(src)
                row["encode_same_bytes"] = bool(bytes(egj.encode(gim, quality=q, subsampling=SUB[sub])) == data)
            row["others"] = {}
            for dn, f in others.items():
                try:
                    row["others"][dn] = compare(f(data), want)
                except Exception as e:
                    row["others"][dn] = dict(error=f"{type(e).__name__}: {str(e)[:100]}")
            if gpu_encode is not None and tag == "q95_444":
                try:
                    eb = gpu_encode(src, q)
                    if egj.inspect(eb).get("subsampling") == "4:4:4":
                        res["nvjpeg_encode"].append(dict(
                            image=name, same_bytes=eb == data, size_against_libjpeg_turbo=round(len(eb) / len(data), 4),
                            psnr_against_source=psnr(simplejpeg.decode_jpeg(eb, colorspace="RGB", fastdct=False, fastupsample=False), src),
                            libjpeg_turbo_psnr_against_source=psnr(want, src)))
                except Exception as e:
                    res["nvjpeg_encode"].append(dict(image=name, error=f"{type(e).__name__}: {str(e)[:100]}"))
            res["files"].append(row)
            print(name, tag, "route", row["route"], "passes", row["passes"], "same pixels", row["same_pixels"],
                  "encode", row.get("encode_same_bytes"), flush=True)
        del im, src, gim
        cp.get_default_memory_pool().free_all_blocks()
    rows = res["files"]
    enc = [r for r in rows if "encode_same_bytes" in r]
    s = dict(jpegs=len(rows), on_the_gpu_route=sum(r["route"] == "gpu" for r in rows),
             decoded_on_the_gpu=sum(r["gpu_decoded"] for r in rows), pixels_identical=sum(r["same_pixels"] for r in rows),
             passes={str(k): sum(r["passes"] == k for r in rows) for k in sorted({r["passes"] for r in rows if r["passes"]})},
             encodes_compared=len(enc), encodes_byte_identical=sum(r["encode_same_bytes"] for r in enc),
             from_packed_identical=sum(bool(r.get("from_packed_same")) for r in rows),
             bytes_per_pixel=[min(r["bytes_per_pixel"] for r in rows), max(r["bytes_per_pixel"] for r in rows)], others={})
    s["ms_per_coded_mb"] = per_mb_fit(rows)
    for k in ("packed_bytes_per_block", "packed_against_scan", "wide_blocks"):
        v = [r[k] for r in rows if k in r]
        if v:
            s[k] = [min(v), max(v)]
    for dn in others:
        v = [r["others"][dn] for r in rows]
        d = [x for x in v if "max_diff" in x]
        o = dict(identical=sum(bool(x.get("same")) for x in v), differ=len(d), failed=sum("error" in x or x.get("decoded") is False for x in v))
        if d:
            o.update(max_diff=max(x["max_diff"] for x in d), pixels_differ=[min(x["pixels_differ"] for x in d), max(x["pixels_differ"] for x in d)],
                     psnr=[min(x["psnr"] for x in d), max(x["psnr"] for x in d)])
            o["by_setting"] = {t[0]: dict(identical=sum(bool(r["others"][dn].get("same")) for r in rows if r["setting"] == t[0]),
                                          max_diff=max((r["others"][dn].get("max_diff", 0) for r in rows if r["setting"] == t[0]), default=0))
                               for t in SETTINGS}
        s["others"][dn] = o
    e = [x for x in res["nvjpeg_encode"] if "same_bytes" in x]
    if e:
        s["nvjpeg_encode_q95_444"] = dict(
            images=len(e), byte_identical=sum(x["same_bytes"] for x in e),
            size_against_libjpeg_turbo=[min(x["size_against_libjpeg_turbo"] for x in e), max(x["size_against_libjpeg_turbo"] for x in e)],
            psnr_minus_libjpeg_turbo=[round(min(x["psnr_against_source"] - x["libjpeg_turbo_psnr_against_source"] for x in e), 2),
                                      round(max(x["psnr_against_source"] - x["libjpeg_turbo_psnr_against_source"] for x in e), 2)])
    res["summary"] = s
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
