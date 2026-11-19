"""The GPU encode path (CuPy): byte-identical to Pillow's JPEG writer.

Pillow 13 (libjpeg-turbo 3.2.0) with `Image.fromarray(rgb).save(f, "JPEG", quality=q, subsampling=s,
icc_profile=..., exif=...)`: baseline, standard Huffman tables (optimize off), islow DCT, no smoothing, JFIF 1.01
header without density, then EXIF (APP1) and the ICC profile (APP2 chunks), DQT, SOF0, DHT, SOS.
"""
from __future__ import annotations

import os
import traceback
import threading
from time import perf_counter
from importlib import resources

import cupy as cp
import cupyx
import numpy as np

from . import _selfcheck
from ._decoder import NAT, _ceil
from ._header import _STD

TILE = 4096
SUBSAMPLINGS = {"444": (1, 1), "422": (2, 1), "420": (2, 2), 0: (1, 1), 1: (2, 1), 2: (2, 2),
                "4:4:4": (1, 1), "4:2:2": (2, 1), "4:2:0": (2, 2)}
MAX_DIMENSION = 65500
CAP_FLOOR = 1 << 20                    # bytes of entropy-coded data every encode has room for (see Encoder._queue)
SMALL_BLOCKS = 16384                   # up to this many 8x8 blocks, an encode is 4 or 5 launches (enc_plan, stuff_small)
SCAN_BLOCKS = 1 << 16                  # up to this many values, one kernel makes the offsets (enc_offsets)
FIRST = 1 << 16                        # bytes of data downloaded together with the size words (one host wait)
MAX_MARKER = 65533                     # bytes of payload in one marker segment
ICC_CHUNK = MAX_MARKER - 14            # Pillow's ICC chunk size ("ICC_PROFILE\0" + 2 bytes of numbering)

# jcparam.c std_luminance_quant_tbl / std_chrominance_quant_tbl (JPEG standard Annex K.1), natural order
STD_LUM = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.int64)
STD_CHR = np.array([17, 18, 24, 47] + [99] * 4 + [18, 21, 26, 66] + [99] * 4 + [24, 26, 56] + [99] * 5 + [47, 66] +
                   [99] * 6 + [99] * 32, np.int64)


def quant_tables(quality):
    """jcparam.c jpeg_set_quality(quality, force_baseline=TRUE): two tables, natural order."""
    q = min(max(int(quality), 1), 100)
    scale = 5000 // q if q < 50 else 200 - 2 * q
    out = []
    for base in (STD_LUM, STD_CHR):
        t = (base * scale + 50) // 100
        out.append(np.clip(t, 1, 255).astype(np.int64))
    return out


def reciprocals(table):
    """jcdctmgr.c compute_reciprocal(q << 3) with a 16-bit DCTELEM (libjpeg-turbo SIMD builds): recip, corr, shift."""
    rec, cor, sh = [], [], []
    for qv in table:
        d = int(qv) << 3
        b = d.bit_length() - 1
        r = 16 + b
        fq, fr = (1 << r) // d, (1 << r) % d
        c = d // 2
        if fr == 0:
            fq >>= 1
            r -= 1
        elif fr <= d // 2:
            c += 1
        else:
            fq += 1
        rec.append(fq)
        cor.append(c)
        sh.append(r - 16)
    return np.array(rec + cor + sh, np.int32)


def huffman_codes(counts, vals):
    """jchuff.c jpeg_make_c_derived_tbl: code and size per symbol."""
    code = np.zeros(256, np.uint16)
    size = np.zeros(256, np.uint8)
    c, p = 0, 0
    for l in range(1, 17):
        for _ in range(counts[l - 1]):
            code[vals[p]] = c
            size[vals[p]] = l
            c += 1
            p += 1
        c <<= 1
    return code, size


def _seg(marker, payload):
    if len(payload) > MAX_MARKER:
        raise ValueError(f"marker {marker:#04x} payload too long ({len(payload)} bytes)")
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def header(width, height, quality, samp, icc_profile=None, exif=None):
    """The bytes Pillow writes before the entropy-coded data."""
    tq = quant_tables(quality)
    out = bytearray(b"\xff\xd8")
    out += _seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if exif:
        out += _seg(0xE1, bytes(exif))
    if icc_profile:
        icc = bytes(icc_profile)
        chunks = [icc[i:i + ICC_CHUNK] for i in range(0, len(icc), ICC_CHUNK)]
        if len(chunks) > 255:
            raise ValueError("ICC profile too long")
        for k, ch in enumerate(chunks):
            out += _seg(0xE2, b"ICC_PROFILE\x00" + bytes([k + 1, len(chunks)]) + ch)
    for k in (0, 1):
        out += _seg(0xDB, bytes([k]) + bytes(int(v) for v in tq[k][NAT]))
    h, v = samp
    out += _seg(0xC0, bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([3]) +
                bytes([1, (h << 4) | v, 0, 2, 0x11, 1, 3, 0x11, 1]))
    for tc, th in ((0, 0), (1, 0), (0, 1), (1, 1)):
        counts, vals = _STD[(tc, th)]
        out += _seg(0xC4, bytes([(tc << 4) | th]) + bytes(counts) + vals)
    out += _seg(0xDA, bytes([3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]))
    return bytes(out)


_headers = {}


def _header(width, height, quality, samp, icc_profile, exif):
    """header(), remembered for images without EXIF or an ICC profile (the same few sizes come again and again)."""
    if icc_profile or exif:
        return header(width, height, quality, samp, icc_profile, exif)
    key = (width, height, int(quality), samp)
    r = _headers.get(key)
    if r is None:
        if len(_headers) >= 256:
            _headers.clear()
        r = _headers[key] = header(width, height, int(quality), samp)
    return r


def _out_cap(cap, retry):
    """Bytes for the stuffed data of an encode whose entropy-coded data may take `cap` bytes. Stuffing puts a 0x00
    after every 0xFF byte, so it can at most double the data. First attempt: half as much again plus 64 kB (real
    data has few 0xFF bytes). The retry: the worst case, so that it cannot overflow a second time."""
    return 2 * cap + 65536 if retry else cap + cap // 2 + 65536


ATTEMPTS = 2                           # an encode is the first attempt and at most one retry with exact sizes


def _fix(x):
    return int(x * 65536 + 0.5)


_mods = {}
_lock = threading.Lock()


def _k(name):
    dev = cp.cuda.Device().id
    with _lock:
        m = _mods.get(dev)
        if m is None:
            src = resources.files(__package__).joinpath("_enc_kernels.cu").read_text()
            zz = np.argsort(NAT)
            for k, v in dict(ZZ=", ".join(str(int(x)) for x in zz), FY_R=_fix(0.29900), FY_G=_fix(0.58700),
                             FY_B=_fix(0.11400), FCB_R=_fix(0.16874), FCB_G=_fix(0.33126), F_HALF=_fix(0.5),
                             FCR_G=_fix(0.41869), FCR_B=_fix(0.08131)).items():
                src = src.replace(f"@{k}@", str(v))
            m = _mods[dev] = cp.RawModule(code=src, options=("--std=c++14",))
            for n in ("enc_dct", "enc_dummy_dc", "enc_dc_bits", "enc_write", "stuff_count", "stuff_scatter",
                      "enc_offsets", "enc_info", "enc_plan", "stuff_small"):
                m.get_function(n)
        return m.get_function(name)


_tables = {}


def _device_tables(width, height, quality, samp):
    key = (width, height, int(quality), samp, cp.cuda.Device().id)
    r = _tables.get(key)
    if r is not None:
        return r
    comps = [samp, (1, 1), (1, 1)]
    hmax, vmax = samp
    blocks = [(ci, dx, dy) for ci, (hs, vs) in enumerate(comps) for dy in range(vs) for dx in range(hs)]
    bpm = len(blocks)
    mcux, mcuy = _ceil(width, 8 * hmax), _ceil(height, 8 * vmax)
    el = np.zeros(10 * 3 + 1 + 6 + 3 + 6 + 2 + 9, np.int32)
    el[0] = bpm
    for k, (ci, dx, dy) in enumerate(blocks):
        el[1 + k], el[11 + k], el[21 + k] = ci, dx, dy
    el[31:34] = [c[0] for c in comps]
    el[34:37] = [c[1] for c in comps]
    el[37:40] = [hmax, vmax, mcux]
    el[40:43] = [_ceil(width * c[0], hmax * 8) for c in comps]           # width_in_blocks
    el[43:46] = [_ceil(height * c[1], vmax * 8) for c in comps]          # height_in_blocks
    el[46:48] = [width, height]
    for ci in range(3):
        js = [k for k, b in enumerate(blocks) if b[0] == ci]
        el[48 + ci], el[51 + ci] = js[0], js[-1]
    el[54:57] = [0, 1, 1]
    div = np.concatenate([reciprocals(t) for t in quant_tables(quality)])
    codes, sizes = zip(*(huffman_codes(*_STD[k]) for k in ((0, 0), (1, 0), (0, 1), (1, 1))))
    ht = np.concatenate([np.concatenate(codes).view(np.int32), np.concatenate(sizes).view(np.int32)])
    dummies = any(_ceil(width * c[0], hmax * 8) < mcux * c[0] or _ceil(height * c[1], vmax * 8) < mcuy * c[1]
                  for c in comps)
    r = dict(el=cp.asarray(el), div=cp.asarray(div), ht=cp.asarray(ht), bpm=bpm, nmcu=mcux * mcuy,
             nblk=mcux * mcuy * bpm, dummies=dummies)
    cp.cuda.get_current_stream().synchronize()     # uploaded on this stream; shared by every encoder from here on
    if len(_tables) > 64:
        _tables.clear()
    _tables[key] = r
    return r


def _as_cupy_rgb(img, dev):
    """Validate and wrap the input: (H, W, 3) uint8, C-contiguous; CuPy array, torch CUDA tensor or numpy array
    (uploaded to the current device). A GPU array must be on device `dev`, the encoder's."""
    if isinstance(img, np.ndarray):
        a = img
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"expected a (H, W, 3) uint8 array, got {a.shape} {a.dtype}")
        return cp.asarray(np.ascontiguousarray(a))
    if not isinstance(img, cp.ndarray):
        if hasattr(img, "__dlpack__"):
            if getattr(img, "is_cuda", True) is False:
                raise ValueError("torch tensor must be on the GPU (or pass a numpy array)")
            img = cp.from_dlpack(img)
        else:
            raise TypeError(f"expected a CuPy array, a CUDA torch tensor or a numpy array, got {type(img)}")
    if img.dtype != cp.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected a (H, W, 3) uint8 array, got {img.shape} {img.dtype}")
    if not img.flags.c_contiguous:
        raise ValueError("image must be C-contiguous (H, W, 3); use cupy.ascontiguousarray")
    if img.device.id != dev:
        raise ValueError(f"the image is on device {img.device.id}, the encoder on device {dev}")
    return img


def _wait_event(stream, ev):
    """Make `stream` wait for a recorded GPU event (cupy.cuda.Event or torch.cuda.Event): no host wait."""
    if isinstance(ev, cp.cuda.Event):
        stream.wait_event(ev)
        return
    ptr = getattr(ev, "cuda_event", None)            # torch.cuda.Event (a ctypes pointer once recorded)
    ptr = getattr(ptr, "value", ptr)
    if not isinstance(ptr, int) or not ptr:
        raise TypeError("ready must be a recorded cupy.cuda.Event or torch.cuda.Event")
    cp.cuda.runtime.streamWaitEvent(stream.ptr, ptr)


def _write_all(f, mv):
    """Write every byte: an unbuffered write is one system call and may write fewer bytes than given (on POSIX,
    a file system that fills up does that without an error); what is left is written again, and a write that
    makes no progress is an error."""
    while len(mv):
        n = f.write(mv)
        if not n:
            raise OSError(28, "could not write the rest of the file (no space left?)", getattr(f, "name", None))
        mv = mv[n:]


class EncodeJob:
    """A queued encode (Encoder.encode_async). event: the CuPy event the GPU reaches when the entropy-coded data is
    complete on the device. result(): waits for it, downloads the data and returns the JPEG bytes (with path=:
    writes the file and returns its length); the same bytes as Encoder.encode()."""

    def __init__(self, enc, st):
        self.enc, self._st = enc, st
        self.event = st["done"]
        self._done, self._res, self._exc = False, None, None

    def result(self):
        """The JPEG bytes (with path=: the file's length). If the encode failed (a file that could not be opened or
        written, a CUDA error), this raises, and so does every later call: the file was not written."""
        if self._done:
            if self._exc is not None:
                raise self._exc
            return self._res
        self._done = True
        e, st, self._st = self.enc, self._st, None
        if e._job is self:
            e._job = None
        try:
            with e.device, e.stream:
                self._res = e._collect(st)
        except BaseException as x:
            # what the failed encode held (device buffers, an open file) must not live on in the kept exception:
            # its frames are cleared, and they are the only holders of those buffers
            st.clear()
            traceback.clear_frames(x.__traceback__)
            self._exc = x
            raise
        return self._res


class Encoder:
    """Encodes RGB images on one GPU. Not thread-safe: one Encoder per thread (each has its own CUDA stream)."""

    def __init__(self, device=None):
        self.device = cp.cuda.Device(device if device is not None else cp.cuda.Device().id)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self._scan_blocks = SCAN_BLOCKS      # 0: the route of images above SCAN_BLOCKS, whatever the image (tests)
        self.stats = {}
        self._pin = None
        self._info = None                    # page-locked: the size words of the encode being collected
        self._job = None                     # the EncodeJob not yet collected
        self.profile = False                 # "host": the host's time per step in stats["host_ms"]
        self._hp = None

    def _host(self, name):
        if self._hp is not None:
            self._hp.append((name, perf_counter()))

    PIECE = 1 << 20                          # path=...: the file is written in pieces as they arrive from the GPU
    BLOCK_WAIT = True                        # path=...: wait for each piece asleep (True) or spinning (False)

    def encode(self, img, quality=75, subsampling="420", icc_profile=None, exif=None, timings=None, path=None,
               ready=None):
        """JPEG bytes for an (H, W, 3) uint8 RGB image (CuPy, CUDA torch tensor or numpy). Same arguments and the
        same bytes as Pillow's `Image.fromarray(img).save(f, "JPEG", quality=..., subsampling=..., icc_profile=...,
        exif=...)`. subsampling: "444", "422", "420" (or Pillow's 0, 1, 2). Raises ValueError/TypeError on bad
        input. path: write the JPEG to this file instead (each piece is written as soon as it has arrived from
        the GPU, so the download and the file write overlap) and return its length. The file is opened as
        open(path, "wb") does: if it cannot be opened (PermissionError when another program holds it or it is
        read-only) the error is raised with nothing changed, and if the encode fails after that the file is
        removed, so no partial file is left. ready: see encode_async."""
        try:
            return self._start(img, quality, subsampling, icc_profile, exif, timings, path, ready).result()
        except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError, cp.cuda.compiler.CompileException) as e:
            # a CUDA error or a kernel that does not compile: one clear reason, and the GPU path goes off
            raise RuntimeError("exact_gpu_jpeg: " + _selfcheck.gpu_error(self.device.id, e)) from e

    def encode_async(self, img, quality=75, subsampling="420", icc_profile=None, exif=None, path=None, ready=None):
        """Queue the whole encode on this encoder's stream and return an EncodeJob at once, without waiting for the
        GPU; job.result() then does the one small size readback, the download and (with path=) the file write, and
        returns what encode() returns. So several Encoders can have their scans queued back to back (for example a
        photo's exports) before any host wait.

        ready: a GPU event (cupy.cuda.Event, or a recorded torch.cuda.Event) that the image's producer recorded on
        its stream after the last kernel that writes `img`. The encoder's stream waits for it on the GPU; the host
        does not wait, except when the size, quality and subsampling need tables this GPU has not built yet (the
        first time they are used on it, and again once more than 64 others have been used since): building them
        waits on the host, so the host then waits for the producer. Without `ready`, a GPU image is waited for on the host first (the calling thread's current CuPy
        stream is synchronised), as encode() does. The image must not be changed until result() has returned.
        One job per Encoder at a time: the next encode() or encode_async() collects the previous job first (call
        its result() to get its bytes)."""
        return self._start(img, quality, subsampling, icc_profile, exif, None, path, ready)

    def _start(self, img, quality, subsampling, icc_profile, exif, timings, path, ready):
        self._hp = [("call", perf_counter())] if self.profile == "host" else None
        if subsampling not in SUBSAMPLINGS:
            raise ValueError(f"subsampling must be one of '444', '422', '420' (or 0, 1, 2), got {subsampling!r}")
        if isinstance(quality, bool) or not isinstance(quality, (int, np.integer)) or not 1 <= quality <= 100:
            raise ValueError(f"quality must be an integer 1..100, got {quality!r}")
        if exif is not None and not isinstance(exif, (bytes, bytearray, memoryview)):
            raise TypeError("exif must be bytes")
        if icc_profile is not None and not isinstance(icc_profile, (bytes, bytearray, memoryview)):
            raise TypeError("icc_profile must be bytes")
        samp = SUBSAMPLINGS[subsampling]
        why = _selfcheck.ensure(self.device.id, "encode")
        if why is not None:                        # the GPU path is off for this process (see _selfcheck)
            raise RuntimeError("exact_gpu_jpeg: " + why)
        if self._job is not None:
            try:
                self._job.result()
            except Exception as x:                 # the error is that earlier job's, not this call's: say so
                if hasattr(x, "add_note"):
                    x.add_note("exact_gpu_jpeg: raised by the previous queued encode (encode_async) of this "
                               "Encoder, which this call concluded; this call's image was not encoded")
                raise
        with self.device:
            if ready is None and not isinstance(img, np.ndarray):
                # The caller's GPU image may still be being written by work queued on the caller's stream (CuPy
                # uploads a NumPy array asynchronously): this encoder's kernels run on another stream, which
                # does not wait for it. (With `ready`, the stream waits for that event instead.)
                cp.cuda.get_current_stream().synchronize()
        self._host("checks, wait for the caller's stream")
        with self.device, self.stream:
            if ready is not None:
                _wait_event(self.stream, ready)
            img = _as_cupy_rgb(img, self.device.id)
            H, W = int(img.shape[0]), int(img.shape[1])
            if not (1 <= W <= MAX_DIMENSION and 1 <= H <= MAX_DIMENSION):
                raise ValueError(f"image size {W}x{H} outside 1..{MAX_DIMENSION}")
            hdr = _header(W, H, quality, samp, icc_profile, exif)
            self._host("file header")
            st = self._queue(img, W, H, int(quality), samp, timings, hdr, path)
        self._job = EncodeJob(self, st)
        return self._job

    def _pinned(self, n):
        if self._pin is None or self._pin.size < n:
            self.stream.synchronize()
            self._pin = cupyx.empty_pinned(int(n * 1.25) + 65536, dtype=np.uint8)
        return self._pin[:n]

    def _queue(self, img, W, H, quality, samp, timings, hdr, path, cap=None, attempt=0):
        """Queue the kernels that make the entropy-coded bytes (no host wait); _collect() finishes. Buffers are sized from a bound (cap bytes; by default 40 bytes per 8x8 block, far above
        a camera photo at quality 95), so the only host waits are one small readback of the size and the download;
        an image that needs more is redone with exact-size buffers."""
        s = self.stream
        ev = [cp.cuda.Event() for _ in range(4)] if timings is not None else None
        if ev:
            ev[0].record(s)
        T = _device_tables(W, H, quality, samp)
        self._host("tables")
        nblk, nmcu = T["nblk"], T["nmcu"]
        coef = cp.empty(nblk * 64, cp.int16)
        nb = cp.empty(nblk, cp.int32)
        _k("enc_dct")((_ceil(nblk, 32),), (256,), (img, T["el"], T["div"], np.int64(nblk), coef, T["ht"], nb))
        if T["dummies"]:
            _k("enc_dummy_dc")((_ceil(nmcu, 128),), (128,), (T["el"], np.int64(nmcu), coef))
        if ev:
            ev[1].record(s)
        self._host("dct")
        if cap is None:
            cap = max(CAP_FLOOR, nblk * 40)
        out_cap = _out_cap(cap, attempt > 0)
        if nblk <= SMALL_BLOCKS:
            # a small image: one launch for the DC bits, the offsets and the clearing, one for the bits, one for
            # the stuffing and the size words
            off = cp.empty(nblk + 1, cp.int64)
            flags = cp.empty(2, cp.int32)
            buf = cp.empty(_ceil(cap, 4) + 2, cp.uint32)
            _k("enc_plan")((1,), (1024,), (coef, T["el"], T["ht"], np.int32(nblk), nb, off, buf, np.int64(buf.size),
                                           flags))
            self._host("dc bits, offsets")
            _k("enc_write")((_ceil(nblk, 128),), (128,), (coef, T["el"], T["ht"], np.int64(nblk), off, buf,
                                                           np.int64(cap * 8), flags))
            if ev:
                ev[2].record(s)
            self._host("write")
            out = cp.empty(out_cap, cp.uint8)
            info = cp.empty(4, cp.int64)                       # bytes out, bits, the two overflow flags
            _k("stuff_small")((1,), (256,), (buf, off[nblk:], np.int64(cap), out, np.int64(out_cap), flags, info))
            keep = (coef, nb, off, buf, flags, T)
        else:
            _k("enc_dc_bits")((_ceil(nblk, 128),), (128,), (coef, T["el"], T["ht"], np.int64(nblk), nb))
            off = cp.empty(nblk + 1, cp.int64)
            if nblk <= self._scan_blocks:                          # one launch: off[0] = 0 and the running sums
                _k("enc_offsets")((1,), (1024,), (nb, np.int32(nblk), off))
            else:
                off[0] = 0
                cp.cumsum(nb, dtype=cp.int64, out=off[1:])
            self._host("dc bits, offsets")
            flags = cp.zeros(2, cp.int32)
            buf = cp.zeros(_ceil(cap, 4) + 2, cp.uint32)
            _k("enc_write")((_ceil(nblk, 128),), (128,), (coef, T["el"], T["ht"], np.int64(nblk), off, buf,
                                                           np.int64(cap * 8), flags))
            if ev:
                ev[2].record(s)
            self._host("write")
            ntile = cap // TILE + 1
            total = off[nblk:]
            if ntile <= self._scan_blocks:
                tc = cp.empty(ntile, cp.int32)
                _k("stuff_count")((ntile,), (256,), (buf, total, np.int64(cap), tc))
                toff = cp.empty(ntile + 1, cp.int64)
                _k("enc_offsets")((1,), (1024,), (tc, np.int32(ntile), toff))
            else:
                tc = cp.zeros(ntile + 1, cp.int32)
                _k("stuff_count")((ntile,), (256,), (buf, total, np.int64(cap), tc[1:]))
                toff = cp.cumsum(tc, dtype=cp.int64)
            out = cp.empty(out_cap, cp.uint8)
            _k("stuff_scatter")((ntile,), (256,), (buf, total, np.int64(cap), toff, out, np.int64(out_cap), flags[1:]))
            info = cp.empty(4, cp.int64)                           # bytes out, bits, the two overflow flags
            _k("enc_info")((1,), (32,), (toff[ntile:], total, flags, info))
            keep = (coef, nb, off, buf, tc, toff, flags, T)
        done = cp.cuda.Event(disable_timing=True)
        done.record(s)
        self._host("stuffing, size words")
        # every device buffer stays referenced until _collect: the kernels that use them are only queued
        return dict(info=info, out=out, done=done, ev=ev, timings=timings, hdr=hdr, path=path, nblk=nblk,
                    args=(img, W, H, quality, samp), keep=keep, attempt=attempt)

    def _collect(self, st, f=None):
        """The host side of an encode: one wait for the size words and the first piece of the data, the rest of
        the download if there is more, and the file write. With path=, the file is opened while the GPU works
        (f: the file, already opened by the first attempt of an encode that is being done again)."""
        s = self.stream
        out, ev, timings, hdr, path, nblk = st["out"], st["ev"], st["timings"], st["hdr"], st["path"], st["nblk"]
        nh = len(hdr)
        first = min(FIRST, out.size)
        opened = f is not None
        try:                                                   # (from here on a failure closes and removes the file,
            self._pinned(nh + first + 2)                       # also one this call was given already opened)
            pin0 = self._pin
            if self._info is None:
                self._info = cupyx.empty_pinned(4, np.int64)
            info = self._info
            st["info"].get(stream=s, out=info, blocking=False)
            out[:first].get(stream=s, out=pin0[nh:nh + first], blocking=False)
            if path is not None and f is None:
                self._host("queue the download")
                f = open(path, "wb", buffering=0)              # the file itself (no temporary name, no rename)
                opened = True
                self._host("open the file")
            s.synchronize()
            self._host("wait, size readback")
            nout, total_bits = int(info[0]), int(info[1])
            if info[2] or info[3]:                             # bigger than the bound: exact sizes this time
                img, W, H, quality, samp = st["args"]
                attempt = st["attempt"] + 1
                st.clear()
                del out
                if attempt >= ATTEMPTS:                        # cannot happen: the retry had room for the worst case
                    raise RuntimeError("exact_gpu_jpeg: the encoded data did not fit its buffer even when the "
                                       f"buffer was sized for the worst case ({total_bits} bits, flags "
                                       f"{int(info[2])}/{int(info[3])}); nothing was written. Please report this.")
                again = self._queue(img, W, H, quality, samp, timings, hdr, path, cap=(total_bits + 7) // 8 + 8,
                                    attempt=attempt)           # (a failure here: the handler below closes f)
                g, f = f, None                                 # the same open file: it is not opened again; the
                return self._collect(again, f=g)               # second call owns it (and closes it on a failure)
            # the file in one page-locked buffer: header, entropy-coded data (straight from the GPU), EOI
            total = nh + nout + 2
            host = self._pinned(total)
            a = first if self._pin is pin0 else 0              # (a new, larger buffer: all of it again)
            if path is None:
                if a < nout:
                    out[a:nout].get(stream=s, out=host[nh + a:nh + nout], blocking=False)
                host[:nh] = np.frombuffer(hdr, np.uint8)
                host[nh + nout:] = (0xFF, 0xD9)
                if a < nout:
                    s.synchronize()
                self._host("download, wait")
                data = host.tobytes()
            else:
                P = self.PIECE
                evs = []
                for x in range(a, nout, P):                    # what the first piece did not bring
                    b = min(nout, x + P)
                    out[x:b].get(stream=s, out=host[nh + x:nh + b], blocking=False)
                    e = cp.cuda.Event(block=self.BLOCK_WAIT, disable_timing=True)
                    e.record(s)
                    evs.append((e, nh + b))
                host[:nh] = np.frombuffer(hdr, np.uint8)
                host[nh + nout:] = (0xFF, 0xD9)
                done = 0
                for e, b in evs:                               # each piece is written as soon as it has arrived
                    e.synchronize()
                    self._host("wait for a piece")
                    _write_all(f, memoryview(host[done:b]))
                    self._host("write to the file")
                    done = b
                _write_all(f, memoryview(host[done:total]))
                self._host("write to the file")
                f.close()
                f = None
                self._host("close the file")
                data = total
        except BaseException:
            try:                                               # downloads into the page-locked buffers may still be
                s.synchronize()                                # queued: they must land before the buffers can go
            except Exception:                                  # noqa: BLE001 - (the error being raised stands)
                pass
            if opened:                                         # failed after the file was opened: no empty or
                try:                                           # partial file is left under that name
                    if f is not None:
                        f, g = None, f
                        g.close()
                    os.remove(path)
                except OSError:
                    pass
            raise
        finally:
            if f is not None:
                f.close()
        if ev:
            ev[3].record(s)
            s.synchronize()
            timings["dct"] = timings.get("dct", 0) + cp.cuda.get_elapsed_time(ev[0], ev[1]) / 1000
            timings["huffman"] = timings.get("huffman", 0) + cp.cuda.get_elapsed_time(ev[1], ev[2]) / 1000
            timings["stuff_download"] = timings.get("stuff_download", 0) + cp.cuda.get_elapsed_time(ev[2], ev[3]) / 1000
        self.stats = dict(blocks=nblk, scan_bytes=nout)
        if self._hp is not None:
            hp = self._hp + [("bytes", perf_counter())]
            self.stats["host_ms"] = [(hp[k][0], round((hp[k][1] - hp[k - 1][1]) * 1e3, 4)) for k in range(1, len(hp))]
        return data
