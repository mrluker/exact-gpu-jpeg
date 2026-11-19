"""The GPU decode path (CuPy)."""
from __future__ import annotations

import os
import threading
from time import perf_counter
from importlib import resources

import cupy as cp
import cupyx
import numpy as np

from . import _selfcheck
from ._header import WARN_EOF, WARN_EXTRA, Header, NeedMore, Unsupported, after_scan, check_trailer, env_int, has_eoi, parse
from ._selfcheck import GuardBandError

LB = 10                    # Huffman lookup-table bits (must match _kernels.cu)
NT = 6                     # Huffman table slots (must match _kernels.cu)


def chunking(ne):
    """(bits per chunk, checkpoints per chunk) for ne bytes of entropy-coded data. Chunks: one thread each in the
    synchronisation passes; the write pass decodes each piece between checkpoints with its own thread. Measured
    on a GTX 1070 (bench/results/speed/s6_*, bench/results/20261117/call_floor.md): small files want many short
    chunks, large ones fewer long ones. EGJ_CHUNK_BITS / EGJ_NCP override (testing)."""
    S = env_int("EGJ_CHUNK_BITS", 0, 64, 1 << 20) or (1024 if ne < 200_000 else 2048 if ne < 6_000_000 else 8192)
    ncp = env_int("EGJ_NCP", 0, 1, 255) or max(S // 512, 8) - 1
    return S, ncp


SMALL_TABLE_SEGS = 16384     # up to these restart intervals and chunks, one kernel builds the chunk table
SMALL_TABLE_CHUNKS = 131072
CHUNK_BITS = None          # None: chosen per image by chunking()
MAX_SYNC_PASSES = 5        # passes a decode redone with host checks gets, after the eager group (see below)
TABW = (1 << LB) // 2 + 18 + 18 + 64      # 32-bit words per Huffman table slot (must match _kernels.cu)
SYNC_GROUP = 3             # synchronisation passes launched per host check
SYNC_ROUNDS = 129          # rounds inside a CUDA block per synchronisation launch, at most (see sync_pass)
# How long a synchronisation launch runs must not be the file's to decide (a file can be made whose chunks never
# synchronise), and the limit must not depend on how fast the card is: it is a limit on work. The work of a launch
# is the entropy-coded data it decodes. In a launch of r rounds a chunk is decoded at most once before the rounds
# (the first pass only) and at most once a round, and in round k only the chunks from the k-th of their CUDA block
# on can be (a chunk decodes again only when the chunk before it changed in the round before): over a block of 128
# chunks that is at most 1 + r (255 - r) / 256 decodes a chunk (sync_decodes; the first pass is the larger case).
# So one launch decodes at most ne x sync_decodes(r) bytes, and r is the most rounds that keep that within
# SYNC_LAUNCH_WORK, whatever the file. Up to 9.3 MB of data nothing is cut (SYNC_ROUNDS rounds a launch); at the
# input limit (MAX_INPUT) a launch has one round.
# A file's whole synchronisation is limited the same way: the eager group of SYNC_GROUP passes, and, when that
# has not settled, a redo of at most MAX_SYNC_PASSES passes; then the file is declined and decode() uses the CPU
# decoder. So at most SYNC_GROUP + MAX_SYNC_PASSES launches, (SYNC_GROUP + MAX_SYNC_PASSES) x SYNC_LAUNCH_WORK
# bytes decoded, for any file. Every photo measured settles in 2 passes; the slowest image found that this path
# still decodes faster than the CPU decoder needs 5 (a 100 MP blank page with a few lines, 4:2:0). What needs
# more is data that does not synchronise by itself: a file made for it, and flat images coded with optimised
# Huffman tables or pure white at 4:2:0, which the CPU decoder decodes faster than this path did when it gave
# them 64 passes.
# Times are one card's measurement, not the rule: bench/sync_worst_launch.py.
SYNC_LAUNCH_WORK = 600_000_000
assert MAX_SYNC_PASSES >= SYNC_GROUP       # the eager group's flags are read as a group: a redo is at least as long


def sync_decodes(rounds):
    """256 x the most times a synchronisation launch of `rounds` rounds decodes a chunk, on average over a CUDA
    block (128 chunks). Never falls when rounds grow."""
    r = min(rounds, 127)
    return 256 + r * (255 - r)


def sync_budget(ne):
    """(rounds per synchronisation launch, passes before the file is declined) for ne bytes of entropy-coded data."""
    lo, hi = 1, SYNC_ROUNDS                    # the most rounds whose work is within SYNC_LAUNCH_WORK (at least 1)
    if ne * sync_decodes(hi) <= SYNC_LAUNCH_WORK * 256:
        return hi, MAX_SYNC_PASSES
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ne * sync_decodes(mid) <= SYNC_LAUNCH_WORK * 256:
            lo = mid
        else:
            hi = mid
    return lo, MAX_SYNC_PASSES


STATE = 8                  # the decode's state words (one device buffer, read back in one piece): error bits, end
#                            of the data, bit count, restart markers found, 4 spare; then one flag per sync pass
# A file of up to this many bytes given in memory is not copied to the GPU: the kernels read it where it is, in
# page-locked host memory mapped into the GPU's address space (a copy command costs more than these reads until
# the file is about this long). 0 turns it off (the environment variable EGJ_MAPPED_BELOW); devices without unified
# addressing or host-memory mapping always copy.
MAPPED_BELOW = env_int("EGJ_MAPPED_BELOW", 131072, 0, 1 << 30)
_can_map = {}


def _mapped_ok(dev):
    """Can kernels on this device read page-locked host memory at its host address?"""
    r = _can_map.get(dev)
    if r is None:
        try:
            ga = cp.cuda.runtime.deviceGetAttribute
            r = bool(ga(19, dev)) and bool(ga(41, dev))    # cudaDevAttrCanMapHostMemory, ...UnifiedAddressing
        except Exception:
            r = False
        _can_map[dev] = r
    return r


HEADER_MEMO = 4096         # bytes: a header up to this long is remembered for the next file (Decoder._parse)
DC_ONE_BLOCK = 4096        # up to this many MCUs, dc_scan is one CUDA block and one launch
DC_RUN = 4                 # larger images: MCUs per thread in dc_scan (more only to stay within DC_BLOCKS)
DC_BLOCKS = 256            # CUDA blocks of dc_scan at most
PAD = 4096                 # zero bytes after the entropy-coded data (see Rd in _kernels.cu)
TILE = 4096
MAX_INPUT = 1 << 28        # bytes; bit positions are 32-bit
MAX_FILE = 1 << 30         # bytes: a larger file (or bytes input) is declined before any buffer is made for it


def _too_large(n):
    return f"file of {n >> 20} MB, larger than 1 GB"


def _file_size(path):
    n = os.path.getsize(path)
    if n > MAX_FILE:
        raise Unsupported(_too_large(n))
    return n
# How decode(path) reads the file. "streamed": in pieces, each copied to the GPU while the next is read (faster for a
# lone decoder). "whole": the whole file first, then one copy (faster when several decoders work at once: measured
# 94 vs 86 photos/s with two threads, 30 MP photos). "auto" (the default): streamed unless another Decoder is inside
# decode() at that moment. EGJ_STREAM_READ=1 / 0 force streamed / whole.
READ_MODES = ("auto", "streamed", "whole")
READ_MODE = {"0": "whole", "1": "streamed"}.get(os.environ.get("EGJ_STREAM_READ", ""), "auto")
STREAM_READ = READ_MODE != "whole"
_busy = 0                  # Decoders inside decode() right now
_busy_lock = threading.Lock()

ERRORS = {1: "bad Huffman code", 2: "entropy-coded data too short", 4: "AC coefficient index past 63",
          8: "a block needs bits past the end of the data or of its restart interval", 16: "restart markers missing or out of order",
          32: "coefficients outside the range where all libjpeg-turbo IDCTs agree", 64: "DC value outside int16",
          # 128 is not an error: WARN_BIT, the kernels' "extraneous bytes" warning
          1024: "a packed copy with an entry out of range",
          2048: "a packed copy that is not laid out as one"}     # (2048: set by pk_check, a test helper)

# zig-zag position -> natural (row-major) index (jutils.c jpeg_natural_order)
NAT = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7,
                14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39,
                46, 53, 60, 61, 54, 47, 55, 62, 63], dtype=np.int64)


# jddctmgr.c: AA&N scale factors for the fast IDCT (JDCT_IFAST), natural order, scaled up by 14 bits
AANSCALES = np.array([
    16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520, 22725, 31521, 29692, 26722, 22725, 17855, 12299, 6270,
    21407, 29692, 27969, 25172, 21407, 16819, 11585, 5906, 19266, 26722, 25172, 22654, 19266, 15137, 10426, 5315,
    16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520, 12873, 17855, 16819, 15137, 12873, 10114, 6967, 3552,
    8867, 12299, 11585, 10426, 8867, 6967, 4799, 2446, 4520, 6270, 5906, 5315, 4520, 3552, 2446, 1247], np.int64)


def ifast_table(q):
    """libjpeg-turbo's IFAST multipliers from natural-order quantisers (any shape ending in 64):
    DESCALE(quant * aanscale, 14 - IFAST_SCALE_BITS) stored as IFAST_MULT_TYPE, which is short in SIMD builds
    (so a 16-bit quantiser above about 4250 wraps, as it does in libjpeg-turbo)."""
    v = (np.asarray(q, np.int64) * AANSCALES + (1 << 11)) >> 12
    return v.astype(np.int16).astype(np.int32)


def _fix(x):
    return int(x * 65536 + 0.5)


_modules = {}
_mod_lock = threading.Lock()


def _kernel_source():
    """_kernels.cu with its constants filled in."""
    src = resources.files(__package__).joinpath("_kernels.cu").read_text()
    zz = np.argsort(NAT)
    for k, v in dict(ZZ=", ".join(str(int(x)) for x in zz), CRR=_fix(1.40200), CBG=_fix(0.34414),
                     CRG=_fix(0.71414), CBB=_fix(1.77200)).items():
        src = src.replace(f"@{k}@", str(v))
    return src


def _module():
    dev = cp.cuda.Device().id
    with _mod_lock:
        m = _modules.get(dev)
        if m is None:
            m = _modules[dev] = cp.RawModule(code=_kernel_source(), options=("--std=c++14",))
            for name in ("make_chunks", "sync_pass", "decode_write", "unstuff_count", "unstuff_scatter",
                         "fix_segs", "scan2_small", "chunk_table", "first_blocks", "dc_scan", "state_init", "idct_islow",
                         "idct_ifast", "color", "color_ep"):
                m.get_function(name)          # compile now, under the lock
        return m


_fns = {}


def _k(name):
    """The kernel `name` for the current device (looked up in the module once, then kept)."""
    key = (cp.cuda.device.get_device_id(), name)
    f = _fns.get(key)
    if f is None:
        f = _fns[key] = _module().get_function(name)
    return f


def sym_entry(sym, l, dc):
    """A decoded symbol as the kernels use it: (code length + value bits) | value bits << 5 | z step << 9.
    z step: how far the coefficient index moves (DC: 1; AC: run + 1 with a value, 16 for ZRL, 64 = end of block
    for EOB and for any other run with no value bits, as the kernels have always treated them)."""
    s = sym & 15
    if dc:
        za = 1
    else:
        r = sym >> 4
        za = r + 1 if s else 16 if r == 15 else 64
    return (l + s) | (s << 5) | (za << 9)


def huffman_arrays(counts, vals, dc):
    """LUT (1 << LB) of sym_entry() for codes up to LB bits (0 = longer or invalid), maxcode[18], valoffset[18]
    (jdhuff.c jpeg_make_d_derived_tbl), huffval[256]."""
    lut = np.zeros(1 << LB, np.uint16)
    maxc = np.full(18, -1, np.int32)
    voff = np.zeros(18, np.int32)
    hv = np.zeros(256, np.uint8)
    hv[:len(vals)] = np.frombuffer(vals, np.uint8)
    code, p = 0, 0
    for l in range(1, 17):
        nl = counts[l - 1]
        if nl:
            voff[l] = p - code
            for k in range(nl):
                if l <= LB:
                    lo = (code + k) << (LB - l)
                    lut[lo:lo + (1 << (LB - l))] = sym_entry(vals[p + k], l, dc)
            p += nl
            code += nl
            maxc[l] = code - 1
        code <<= 1
    maxc[17] = 0x7FFFFFFF
    return lut, maxc, voff, hv


def _ceil(a, b):
    return -(-a // b)


def layout(h: Header, fancy_upsampling=True):
    """Host-side tables for a header: Huffman tables, MCU layout, plane geometry, quantisers, upsampling."""
    t = _tables(h)
    return {**t, **_geometry(h, t["blocks"], fancy_upsampling)}


def _tables(h: Header):
    """The part of layout() that depends only on the Huffman tables and the sampling factors (shared by the images
    of a batch): tab, ntab, hdr, bpm, blocks."""
    slots, arrays = {}, []
    dct, act = [], []
    for ci in range(h.ncomp):
        for kind, tid, tab, dst in (("dc", h.td[ci], h.dc_tabs[ci], dct), ("ac", h.ta[ci], h.ac_tabs[ci], act)):
            key = (kind, tid, tuple(tab[0]), tab[1])
            if key not in slots:
                slots[key] = len(arrays)
                arrays.append(huffman_arrays(*tab, kind == "dc"))
            dst.append(slots[key])
    ntab = len(arrays)                       # slots in use (the kernels load only these)
    tab = np.concatenate([np.concatenate([a[0] for a in arrays]).view(np.int32),
                          np.concatenate([a[1] for a in arrays]), np.concatenate([a[2] for a in arrays]),
                          np.concatenate([a[3] for a in arrays]).view(np.int32)])
    blocks = [(ci, dx, dy) for ci, (hs, vs) in enumerate(h.samp) for dy in range(vs) for dx in range(hs)]
    bpm = len(blocks)
    hdr = np.zeros(21, np.int32)
    hdr[0] = bpm
    hdr[1:1 + bpm] = [dct[b[0]] for b in blocks]
    hdr[11:11 + bpm] = [act[b[0]] for b in blocks]
    return dict(tab=tab, ntab=ntab, hdr=hdr, bpm=bpm, blocks=blocks)


def _geometry(h: Header, blocks, fancy_upsampling=True):
    """The per-image part of layout(): plane geometry (pix), quantisers (q), upsampling (up), block counts."""
    bpm = len(blocks)
    hmax, vmax = h.hmax, h.vmax
    mcux, mcuy = _ceil(h.width, 8 * hmax), _ceil(h.height, 8 * vmax)
    pitch = [mcux * s[0] * 8 for s in h.samp] + [0] * (3 - h.ncomp)
    rows = [mcuy * s[1] * 8 for s in h.samp] + [0] * (3 - h.ncomp)
    off = np.cumsum([0] + [p * r for p, r in zip(pitch, rows)])[:3]
    head = np.zeros(38, np.int32)
    head[0] = bpm
    for k, (ci, dx, dy) in enumerate(blocks):
        head[1 + k], head[11 + k], head[21 + k] = ci, dx, dy
    head[31:31 + h.ncomp] = [s[0] for s in h.samp]
    head[34:34 + h.ncomp] = [s[1] for s in h.samp]
    pix = np.concatenate([head, np.array(off, np.int64).view(np.int32), np.array(pitch + [mcux], np.int32)])
    q = np.zeros((3, 64), np.int32)
    for ci in range(h.ncomp):
        q[ci, NAT] = h.quant[ci]
    up = np.zeros((5, 3), np.int32)          # fh, fv, dw, dh, fancy
    for ci, (hs, vs) in enumerate(h.samp):
        fh, fv = hmax // hs, vmax // vs
        dw, dh = _ceil(h.width * hs, hmax), _ceil(h.height * vs, vmax)
        # jdsample.c jinit_upsampler with do_fancy_upsampling = TRUE; other integral factors: int_upsample (box).
        # fancy_upsampling=False: libjpeg-turbo's do_fancy_upsampling = FALSE (box / merged upsampling, the same bytes)
        fancy = {(2, 1): int(dw > 2), (1, 2): 1, (2, 2): int(dw > 2)}.get((fh, fv), 0) if fancy_upsampling else 0
        up[:, ci] = (fh, fv, dw, dh, fancy)
    return dict(pix=pix, q=q.reshape(-1), up=up.reshape(-1), mcux=mcux, mcuy=mcuy,
                nmcu=mcux * mcuy, nblk=mcux * mcuy * bpm, plane_bytes=int(sum(p * r for p, r in zip(pitch, rows))))


_dev_cache = {}
_dev_lock = threading.Lock()


def _device_layout(h: Header, fancy_upsampling=True):
    key = (tuple(tuple(t[0]) + (t[1],) for t in h.dc_tabs + h.ac_tabs), tuple(h.td), tuple(h.ta), tuple(h.samp),
           tuple(q.tobytes() for q in h.quant), h.width, h.height, h.ncomp, cp.cuda.Device().id, fancy_upsampling)
    with _dev_lock:
        r = _dev_cache.get(key)
    if r is not None:
        return r
    L = layout(h, fancy_upsampling)
    r = dict(L)
    for k in ("tab", "hdr", "pix", "q", "up"):
        r[k] = cp.asarray(L[k])
    cp.cuda.get_current_stream().synchronize()   # uploaded on this stream; shared by every decoder from here on
    with _dev_lock:
        if len(_dev_cache) > 64:
            _dev_cache.clear()
        _dev_cache[key] = r
    return r


def _ifast_q(T, h):
    """The IFAST multipliers of a cached device layout, made (and cached with it) the first time they are needed."""
    qf = T.get("qf")
    if qf is None:
        q = np.zeros((3, 64), np.int64)
        for ci in range(h.ncomp):
            q[ci, NAT] = h.quant[ci]
        with cp.cuda.Stream():
            qf = cp.asarray(ifast_table(q).reshape(-1))
            cp.cuda.get_current_stream().synchronize()     # shared by every decoder (and stream) from here on
        # Two threads can both get here: every caller must use the array the layout keeps, or the one that lost
        # would launch with an array that nothing holds once the launch is queued.
        qf = T.setdefault("qf", qf)
    return qf


_i32, _i64, _u32, _u64 = np.int32, np.int64, np.uint32, np.uint64
WARN_BIT = 128                         # the error word's bit for whole bytes after an interval's last block (a warning)


def warnings_of(e, data, end):
    """(the error bits without the warning bit, the libjpeg-turbo warnings a decode that found no error has: [] or
    WARN_EXTRA and / or WARN_EOF)."""
    w = [WARN_EXTRA] if e & WARN_BIT else []
    e &= ~WARN_BIT
    if not e:
        eoi, more = after_scan(data, end)
        w += more
        if not eoi:
            w.append(WARN_EOF)
    return e, w


def strict_reason(w):
    return f"libjpeg-turbo warns: {'; '.join(w)} (declined because strict=True; strict=False decodes it as libjpeg-turbo does)"


EXTRA_REASON = ("unused bytes after the last block of a restart interval: libjpeg-turbo warns about extraneous bytes "
                "for some such files and not for others (its bit buffer may already hold a short tail), so the GPU "
                "path declines them in both modes and the CPU decoder decides")


def warned_reason(w, strict):
    """Why a decode with the warnings w (warnings_of) is declined, or None if it is not (strict=False: then its
    warnings are reported). A tail of unused bytes is declined in both modes: whether libjpeg-turbo reports it
    depends on how far its bit buffer read ahead, which the GPU path does not model."""
    if WARN_EXTRA in w:
        return EXTRA_REASON
    return strict_reason(w) if strict else None


def _at(a, i=0):
    """Element i of a device array as a kernel's pointer argument (a slice of the array costs microseconds; the
    array must stay referenced until the launch is queued, as for any argument)."""
    return np.uint64(a.data.ptr + i * a.itemsize)


COLORSPACES = ("RGB", "BGR", "GRAY")


def _epilogue(dtype, layout, scale, bias, nch):
    """None for the default uint8 HWC output, else the colour kernel's epilogue settings."""
    if layout not in ("HWC", "CHW"):
        raise ValueError("layout must be 'HWC' or 'CHW'")
    if dtype == "uint8":
        if layout != "HWC" or scale is not None or bias is not None:
            raise ValueError("layout, scale and bias apply to float output (dtype='float32' or 'float16')")
        return None
    if dtype not in ("float32", "float16"):
        raise ValueError("dtype must be 'uint8', 'float32' or 'float16'")

    def per_channel(v, default):
        a = np.broadcast_to(np.asarray(default if v is None else v, np.float32), (nch,)).astype(np.float32)
        return np.concatenate([a, np.zeros(3 - nch, np.float32)])
    s, b = per_channel(scale, 1.0), per_channel(bias, 0.0)
    if not (np.isfinite(s).all() and np.isfinite(b).all()):
        # (the GPU's NaN is not NumPy's bit pattern, and inf * 0 is NaN: the bytes would not be NumPy's)
        raise ValueError("scale and bias must be finite numbers (after rounding to float32)")
    return dict(dtype=dtype, chw=layout == "CHW", scale=s, bias=b)


def apply_epilogue(img, dtype="float32", layout="HWC", scale=None, bias=None):
    """The decode epilogue on an (H, W, C) uint8 NumPy or CuPy array (the CPU fallback's output), with the same
    float32 rounding as the GPU kernel."""
    epi = _epilogue(dtype, layout, scale, bias, img.shape[2])
    if epi is None:
        return img
    xp = cp.get_array_module(img)
    nch = img.shape[2]
    s, b = xp.asarray(epi["scale"][:nch]), xp.asarray(epi["bias"][:nch])
    v = img.astype(xp.float32) * s + b
    v = v.astype(xp.float16) if dtype == "float16" else v
    return xp.ascontiguousarray(v.transpose(2, 0, 1)) if epi["chw"] else v


KEEP_FORMS = ("raw", "packed")          # the forms of a handle; both live in GPU memory
_HANDLE_KEY = object()                  # CoefHandle's constructor takes it: handles are made by decodes only
HANDLE_FORMS = KEEP_FORMS               # Decoder.keep_coefs


class CoefHandle:
    """What a decode holds just before its pixel stage (Decoder.keep_coefs, take_coefs): every block's quantised
    coefficients. Decoder.decode_from_handle() makes the image from it again, the same bytes, without the file
    and without the Huffman decode.

    form: "raw" = the decoder's three arrays as they are on the GPU (133 bytes a block); "packed" = the same
    values in a smaller layout (13 to 54 bytes a block, about 27 for a camera photo), unpacked on the GPU when the
    handle is decoded from. nbytes: the GPU memory the handle holds. width, height: the image's. device: the GPU it
    lives on. free() gives the memory back (so does dropping the handle); freed tells whether that happened.

    A handle is opaque and lives in GPU memory only: it is valid in the process and on the device that made it, it
    cannot be saved, sent or rebuilt from bytes, and its layout is not part of the API."""

    def __init__(self, form, header, device, parts, nblk, *, _key=None):
        if _key is not _HANDLE_KEY:            # (only a decode makes one: the GPU kernels trust what a handle holds)
            raise TypeError("a CoefHandle is made by Decoder.take_coefs(), not by hand")
        self._form, self._device = form, device
        self._width, self._height = header.width, header.height
        self._h, self._parts, self._nblk = header, parts, nblk     # raw: (coef, nq, dc); packed: (idx, data, dlen)
        self._nbytes = int(sum(p.nbytes for p in parts if hasattr(p, "nbytes")))

    form = property(lambda self: self._form, doc='"raw" or "packed"')
    device = property(lambda self: self._device, doc="the GPU it lives on")
    width = property(lambda self: self._width, doc="the image's width")
    height = property(lambda self: self._height, doc="the image's height")
    nbytes = property(lambda self: self._nbytes, doc="the GPU memory it holds (0 once freed)")

    @property
    def freed(self):
        return self._parts is None

    def free(self):
        self._parts = None
        self._nbytes = 0


class _Queued:
    """A decode whose GPU work is queued (decode_async). out: the array the job hands out. how: what a second
    decode of it needs to be the same decode (the output options and the two mode switches as they were when it
    was queued). tables: the device tables its queued kernels read. They live in a cache shared by every decoder,
    which another decoder can clear at any time; held here, they cannot go back to the memory pool (and be given
    to another stream's work) before result() has waited for the job."""
    def __init__(self, done, resd, out, ctx, how, tables):
        self.done, self.resd, self.out, self.ctx, self.how, self.tables = done, resd, out, ctx, how, tables


class DecodeJob:
    """A queued decode (Decoder.decode_async). event: the CuPy event the GPU reaches when the pixels are written.
    result(): waits for it and returns the image (the `out` array if one was given), or None when the GPU path
    declined the JPEG (decoder.reason). redone: True when the fast synchronisation did not converge and the image
    was decoded again after the event, into the same array (job.out, which result() returns): work queued behind
    the event read unfinished pixels and must be run again. Photos do not take this path; images with large flat
    areas and optimised tables, and very large blank pages, do. reason: why result() gave None (the decoder's own
    `reason` belongs to its latest call, which may be a later one that concluded this job). An exception raised by
    result() is raised again by every later call of it."""
    def __init__(self, dec, queued, out):
        self.dec, self._q, self.out = dec, queued, out
        self.event = queued.done if queued is not None else None
        self.redone = False
        self.reason = None
        self.warning = dec.warning if queued is None else None
        self._res = out if queued is None else None
        self._done = queued is None
        self._exc = None

    def result(self):
        if self._done:
            if self._exc is not None:
                raise self._exc
            return self._res
        self._done = True
        try:
            return self._result()
        except BaseException as e:
            self._exc = e
            raise

    def _result(self):
        d, q = self.dec, self._q
        if d._job is self:
            d._job = None
        buf, h, colorspace, timings, s0, marks, ev = q.ctx
        d._redone = False
        # If the eager passes did not settle, _conclude decodes again. That decode must be this job's: into the
        # array the job handed out (work queued behind job.event reads that array), with the output options and
        # modes the job was queued with, whatever the decoder has been asked for since.
        now = d._out, d._epi, d.fast_dct, d.fast_upsample, d.strict
        d._out, (d._epi, d.fast_dct, d.fast_upsample, d.strict) = q.out, q.how      # (strict too: the job is judged
                                                                                    # by the rule it was queued under)
        try:
            with d.device:
                q.done.synchronize()            # a CUDA error that surfaces here is a decline, as anywhere else
                r = d._conclude(q.resd.get(), q.out, buf, h, colorspace, timings, s0, marks, ev, True)
        except cp.cuda.memory.OutOfMemoryError:
            d.stream.synchronize()
            d.reason, r = "out of GPU memory", None
        except GuardBandError:
            raise
        except (RuntimeError, ImportError, cp.cuda.compiler.CompileException) as e:
            d._guards = []
            d.reason, r = _selfcheck.gpu_error(d.device.id, e), None
        finally:
            d._out, d._epi, d.fast_dct, d.fast_upsample, d.strict = now
        self.redone = d._redone
        self.reason = d.reason if r is None else None
        self.warning = d.warning if r is not None else None
        self._res = r
        q.tables = None                        # the job's kernels have finished: the tables may go
        return r


class Decoder:
    """Decodes JPEGs on one GPU. Not thread-safe: use one Decoder per thread (each has its own CUDA stream).

    decode() returns a CuPy array (H, W, 3) uint8 for RGB/BGR or (H, W, 1) for GRAY, or None when this JPEG
    is not one the GPU path decodes exactly (or is corrupt); `reason` then says why.
    """

    def __init__(self, device=None, *, max_pixels=1 << 28, max_gpu_bytes=None, chunk_bits=CHUNK_BITS, canary=None,
                 fast_dct=False, fast_upsample=False, read=None, keep_coefs=None, cache=None, strict=True):
        """max_pixels: larger images are declined. max_gpu_bytes: decline images whose decode would need more GPU
        memory (default: 90% of what is free). canary: surround every device buffer with guard bands and check
        them after each decode (testing; default from the environment variable EGJ_CANARY=1).
        fast_dct: libjpeg-turbo's fast integer IDCT (JDCT_IFAST: TJFLAG_FASTDCT, simplejpeg fastdct=True)
        instead of the accurate one; less accurate, byte-identical to that mode of libjpeg-turbo. Not the default."""
        self.device = cp.cuda.Device(device if device is not None else cp.cuda.Device().id)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.S = chunk_bits                  # None: per image (chunking)
        self._large_route = False            # True: the route of files with many chunks, whatever the file (tests)
        self.max_pixels = max_pixels
        self.max_gpu_bytes = max_gpu_bytes
        self.fast_dct = bool(fast_dct)
        # read: how decode(path) reads the file, "auto" (default), "streamed" or "whole" (see READ_MODE); the bytes
        # decoded are the same
        self.read = READ_MODE if read is None else read
        if self.read not in READ_MODES:
            raise ValueError(f"read must be one of {READ_MODES}")
        self.canary = (os.environ.get("EGJ_CANARY") == "1") if canary is None else canary
        self._guards = []
        # True: per-step GPU times in stats["profile_ms"]. "host": instead, the host's time between the same steps
        # in stats["host_ms"] (how long the calling thread took to queue each step, and to wait at the end)
        self.profile = False
        self._hp = None
        self._out = None                     # decode_async(out=...)
        self._epi = None                     # decode(dtype=..., layout=..., scale=..., bias=...)
        self._async = False
        self._job = None                     # the DecodeJob not yet concluded
        self._redone = False
        self._last = None                   # (header bytes, Header) of the last file parsed from memory
        # opt-in: libjpeg-turbo's fast upsampling (do_fancy_upsampling = FALSE; simplejpeg fastupsample=True):
        # chroma is repeated instead of interpolated. Byte-identical to libjpeg-turbo in that mode (a compatibility
        # option: not faster on a GTX 1070).
        self.fast_upsample = fast_upsample
        self._pin = None
        # strict (default True): a file libjpeg-turbo decodes only with a warning (no end marker, extraneous bytes
        # before a marker, an unknown JFIF version or Adobe transform) is declined, with the warning as the reason.
        # strict=False: such a file gives libjpeg-turbo's image and `warning` names the warning.
        self.strict = bool(strict)
        self.warning = None                  # the last call's libjpeg-turbo warning, with strict=False
        self.warnings = []                   # decode_batch: one per image (None, or the warning)
        self._pin_dev = None                # the same buffer as the GPU sees it (when it is mapped)
        self._pad = 0                       # where the file starts in it
        self._pending = None                # a streamed file still being read (see _open_streamed)
        # opt-in: None (default: nothing is kept), or "raw" / "packed": each decode() also keeps its coefficients on the GPU;
        # take_coefs() hands them out as a CoefHandle for decode_from_handle()
        self.keep_coefs = keep_coefs
        self._coefs = None                  # the last concluded decode's handle, until take_coefs()
        # opt-in: a RepeatCache (None: none). decode() of one file then looks for a copy of it first and keeps one
        # after a normal decode; queued decodes and batches do not use it
        self.cache = cache
        self._cache_busy = False            # the cache itself is calling decode()
        self._err_bits = 0                  # decode_from_handle: the error bits of a declined decode, -1 = a CUDA error
        self._copychk = None                # the copy kernels' check on this GPU (_copy_check), once it has run
        self._coefs_pending = None          # ... of a decode not yet concluded
        self._ahead = None                  # upload_ahead(): (src, buffer, header, device copy of the scan)
        self._raw_ahead = None              # ... its device copy, for the decode that takes it
        self.reason = None
        self.reasons = []                    # decode_batch: why each declined image was declined
        self.stats = {}

    FIRST_PIECE = 1 << 18                    # streamed reads (_open_streamed): first piece, for the header
    PIECE = 1 << 21                          # then pieces of this size, each copied to the GPU once read

    def _load(self, src):
        self._pad = 0
        if isinstance(src, (str, os.PathLike)):
            n = _file_size(src)
            self._peek(src, n)
            buf = self._pinned(n)
            with open(src, "rb", buffering=0) as f:
                got = f.readinto(memoryview(buf))
            if got != n:
                raise OSError(f"short read: {src}")
            return buf
        mv = memoryview(src).cast("B")
        if len(mv) > MAX_FILE:
            raise Unsupported(_too_large(len(mv)))
        if len(mv) <= MAPPED_BELOW and _mapped_ok(self.device.id):
            # read in place: the kernels load whole words, so the entropy-coded data must start on a 64-byte
            # boundary of the page-locked buffer
            a = np.frombuffer(mv, np.uint8)
            try:
                self._pad = -self._parse(a).scan_start % 64
            except Unsupported:
                pass
            buf = self._pinned(len(mv) + self._pad)[self._pad:]
            buf[:] = a
            return buf
        buf = self._pinned(len(mv))
        buf[:] = np.frombuffer(mv, np.uint8)
        return buf

    GUARD = 65536

    def _alloc(self, shape, dtype, zero=False):
        """A device buffer. In canary mode it sits between two GUARD-byte bands of 0xA5 (checked after the
        decode), and its own bytes start as 0x5A instead of whatever the pool held."""
        if not self.canary:
            return cp.zeros(shape, dtype) if zero else cp.empty(shape, dtype)
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        G = self.GUARD
        raw = cp.empty(nbytes + 2 * G, cp.uint8)
        raw[:G].fill(0xA5)
        raw[G + nbytes:].fill(0xA5)
        raw[G:G + nbytes].fill(0 if zero else 0x5A)
        self._guards.append((raw, nbytes))
        return raw[G:G + nbytes].view(dtype).reshape(shape)

    def _check_guards(self):
        G = self.GUARD
        for k, (raw, nbytes) in enumerate(self._guards):
            if bool((raw[:G] != 0xA5).any()) or bool((raw[G + nbytes:] != 0xA5).any()):
                raise GuardBandError(f"exact_gpu_jpeg: a kernel wrote outside device buffer #{k} ({nbytes} bytes)")
        self._guards = []

    PEEK_ABOVE = 32 << 20                    # a file larger than this has its header read before it is page-locked
    KEEP_PINNED = 256 << 20                  # a page-locked buffer above this and 4 times the file's size is given back

    def _peek(self, path, n):
        """For a large file: its header, read into ordinary memory and parsed before a page-locked buffer of the
        file's size is made, so a file that is not a JPEG the GPU path decodes (a video, a 900 MB anything) is
        declined without page-locking its size. Unsupported if the header is declined."""
        if n <= self.PEEK_ABOVE:
            return
        with open(path, "rb", buffering=0) as f:
            head = bytearray()
            want = self.FIRST_PIECE
            while True:
                part = f.read(min(n, want) - len(head))
                if not part:
                    raise OSError(f"short read: {path}")
                head += part
                try:
                    parse(head)
                    return
                except NeedMore:                # (parse stops at 16 MB of header: a few rounds at most)
                    if len(head) >= n:
                        raise
                want = len(head) + self.PIECE

    def _pinned(self, n):
        if (self._pin is not None and self._pin.size > max(4 * n, self.KEEP_PINNED)
                and self._ahead is None and self._pending is None):
            # a buffer made for a much larger file: given back (page-locked memory is the whole machine's)
            self.stream.synchronize()
            self._pin = self._pin_dev = None
        if self._pin is None or self._pin.size < n:
            self.stream.synchronize()
            size = int(n * 1.25) + 4096
            try:
                return self._pin_new(size)[:n]
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                # page-locked host memory is short: this file is declined (the CPU decoder can still read it)
                raise Unsupported(f"could not page-lock {size >> 20} MB of host memory for the file ({e})") from e
        return self._pin[:n]

    def _pin_new(self, size):
        """A new page-locked buffer of `size` bytes; the decoder's buffer and its device view change together,
        and only once both exist."""
        if MAPPED_BELOW and _mapped_ok(self.device.id):
            # page-locked memory mapped into the GPU's address space (cudaHostAllocMapped): with unified
            # addressing a kernel can read it at the same address
            mem = cp.cuda.PinnedMemory(size, 2)
            pin = np.ndarray((size,), np.uint8, buffer=cp.cuda.PinnedMemoryPointer(mem, 0))
            dev = cp.ndarray((size,), cp.uint8, cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(mem.ptr, size, mem, self.device.id), 0))
        else:
            pin, dev = cupyx.empty_pinned(size, dtype=np.uint8), None
        self._pin, self._pin_dev = pin, dev
        return pin

    def _open_streamed(self, path):
        """Read only as much of the file as the header needs; the rest is read by _upload, each piece copied to
        the GPU as soon as it is in memory, so reading the file and copying it overlap. Returns (buffer, header).
        The header is parsed from the bytes read so far, and more are read while that fails; parse never reads
        past the SOS segment, so a successful parse of a prefix is the same as the parse of the whole file, and
        a failure is only final once the whole file is in."""
        n = _file_size(path)
        self._pad = 0                       # the file starts at the buffer's start (not where the last bytes input did)
        self._peek(path, n)
        buf = self._pinned(n)
        f = open(path, "rb", buffering=0)
        try:
            got, want = 0, self.FIRST_PIECE
            while True:
                while got < min(n, want):
                    k = f.readinto(memoryview(buf[got:min(n, want)]))
                    if not k:
                        raise OSError(f"short read: {path}")
                    got += k
                try:
                    h = parse(buf[:got])
                    break
                except NeedMore:                # the header goes on past what is read: read more (it is at
                    if got >= n:                # most 16 MB, so this is a few rounds at most)
                        raise
                want = got + self.PIECE
        except BaseException:
            f.close()
            raise
        if got >= n:
            f.close()
        else:
            self._pending = (f, got)
        return buf, h

    def _upload(self, raw, buf, s0, s):
        """Copy the entropy-coded data buf[s0:] to raw, on stream s. With a streamed file, read the rest of it
        piece by piece, queueing each piece's copy as soon as it is read."""
        if self._pending is None:
            raw.set(buf[s0:], stream=s)
            return
        f, got = self._pending
        self._pending = None
        n = len(buf)
        try:
            pos = s0
            if got > s0:
                raw[:got - s0].set(buf[s0:got], stream=s)
                pos = got
            while got < n:
                k = f.readinto(memoryview(buf[got:min(n, got + self.PIECE)]))
                if not k:
                    raise OSError("short read")
                got += k
                if got > pos and got > s0:
                    a = max(pos, s0)
                    raw[a - s0:got - s0].set(buf[a:got], stream=s)
                    pos = got
        finally:
            f.close()

    def decode_async(self, src, colorspace="RGB", out=None, **output):
        """Queue a decode on this decoder's stream (`self.stream`) and return a DecodeJob at once, without waiting
        for the GPU: work that waits for `job.event` (e.g. `my_stream.wait_event(job.event)`) starts the moment
        the pixels are written. out: optional C-contiguous CuPy array of the output's shape and dtype (default
        uint8 (H, W, 3), GRAY (H, W, 1); see decode's output options) to decode into, e.g. your kernels' input. Returns None when the JPEG is declined before any GPU work
        (`self.reason` says why). Call job.result() before using the pixels for anything that must be correct
        (it returns the image, or None for a JPEG the GPU path does not decode exactly); one job per decoder at
        a time (the next decode concludes it first)."""
        self._conclude_job()
        self._async, self._out, self._redone = True, out, False
        try:
            r = self.decode(src, colorspace, **output)
        finally:
            self._async, self._out = False, None
        if r is None:
            return None
        if not isinstance(r, _Queued):            # concluded already (e.g. a decode with host checks)
            return DecodeJob(self, None, r)
        self._job = DecodeJob(self, r, r.out)
        return self._job

    def decode_batch(self, srcs, colorspace="RGB", *, dtype="uint8", layout="HWC", scale=None, bias=None):
        """Decode several JPEGs (paths or bytes-like) at once: a list with, for each, the image as decode() returns
        it, or None where the GPU path declines it (`self.reasons[i]` says why). JPEGs that share their Huffman
        tables and sampling factors (the usual case for files from one camera, phone or encoder) are decoded
        together by one set of kernel launches, which is several times faster for small images; the pixels are
        the same as decode()'s. The others go through decode() one by one. Images decoded together are views of
        one GPU buffer: keeping one of them alive keeps that buffer (copy it to release the rest).
        Output options: as decode()."""
        from . import _batch
        global _busy
        if colorspace not in COLORSPACES:          # (also when the GPU path is off: the same call, the same error)
            raise ValueError(f"colorspace must be one of {COLORSPACES}")
        why = (_selfcheck.ensure(self.device.id, "decode") or _selfcheck.ensure(self.device.id, "batch")
               or _selfcheck.ensure_modes(self.device.id, _selfcheck.mode_parts(
                   self.fast_dct, self.fast_upsample, _epilogue(dtype, layout, scale, bias, 3), colorspace)))
        if why is not None:                       # the GPU path is off for this process (see _selfcheck)
            srcs = list(srcs)
            self.reason, self.reasons, self.stats = why, [why] * len(srcs), {}
            self.warnings = [None] * len(srcs)
            return [None] * len(srcs)
        from . import _route
        entered = _route.enter()
        with _busy_lock:
            _busy += 1                # a batch reads whole files and keeps the GPU busy: other Decoders read whole too
        keep, self.keep_coefs = self.keep_coefs, None          # a batch keeps no coefficients
        try:
            return _batch.decode_batch(self, srcs, colorspace, dtype, layout, scale, bias)
        finally:
            self.keep_coefs, self._coefs, self._coefs_pending = keep, None, None
            with _busy_lock:
                _busy -= 1
            _route.leave(entered)

    def _conclude_job(self):
        if self._job is not None:
            self._job.result()

    def upload_ahead(self, src):
        """Opt-in, for pipelines: read `src` (a path or bytes-like) now, parse its header and queue the copy of its
        entropy-coded data to the GPU on this decoder's stream, without waiting for it. The next decode() or
        decode_async() of the same `src` (the same object, or the same path) then starts from that copy: no file
        read, no parse and no copy at its start. Meant for a reader thread that stages the next file on a second
        Decoder while the GPU works on the current one (a Decoder is still used by one thread at a time).

        Returns True when the file is staged, False when it is not (`reason` says why, as decode() would; the later
        decode() of it then declines in the usual way). One staged file per decoder: staging another, or decoding
        anything else first, drops it. The pixels are decode()'s; the file is read once, here, so a change to it
        on disk before the decode is not seen, and a bytes-like object (a bytearray, a NumPy array, an mmap)
        changed in place after this call is decoded as it was when it was staged. A path is matched as it names a
        file now: a relative path is resolved here, so a change of the working directory does not make it name
        another file."""
        self._conclude_job()
        self._finish_pending()
        self._ahead = self._raw_ahead = None
        self.reason = _selfcheck.ensure(self.device.id, "decode")
        if self.reason is not None:
            return False
        try:
            buf = self._load(src)
            h = parse(buf)
        except Unsupported as e:
            self.reason = str(e)
            return False
        ne = len(buf) - h.scan_start
        if ne < 1:                            # decode() declines these three the same way
            self.reason = "entropy-coded data too short for the image size"
            return False
        if ne > MAX_INPUT:
            self.reason = "entropy-coded data larger than 256 MB"
            return False
        if self.max_pixels and h.width * h.height > self.max_pixels:
            self.reason = f"more than max_pixels ({self.max_pixels})"
            return False
        try:
            with self.device, self.stream:
                raw = cp.empty(ne, cp.uint8)
                raw.set(buf[h.scan_start:], stream=self.stream)
                self.stream.done              # (a stream query: under WDDM this submits the queued copy now)
        except cp.cuda.memory.OutOfMemoryError:
            self.stream.synchronize()
            self.reason = "out of GPU memory"
            return False
        except (RuntimeError, ImportError, cp.cuda.compiler.CompileException) as e:
            self.reason = _selfcheck.gpu_error(self.device.id, e)
            return False
        self._ahead = (self._source_key(src), buf, h, raw)
        return True

    @staticmethod
    def _source_key(src):
        """What a staged file is matched by: a path as an absolute path, resolved now; any other object itself."""
        if isinstance(src, (str, os.PathLike)):
            return ("path", os.path.normcase(os.path.abspath(os.fspath(src))))
        return ("object", src)

    @classmethod
    def _same_source(cls, staged, src):
        """staged: _source_key() of the staged file; src: what decode() was given."""
        kind, v = staged
        if kind == "object":
            return v is src
        return isinstance(src, (str, os.PathLike)) and cls._source_key(src)[1] == v

    def decode(self, src, colorspace="RGB", timings=None, *, dtype="uint8", layout="HWC", scale=None, bias=None,
               check="full"):
        """src: path, or bytes-like holding the JPEG. colorspace: 'RGB', 'BGR' or 'GRAY'.

        Output options (an epilogue in the colour kernel: one pass, no separate conversion): dtype 'uint8' (the
        default, libjpeg-turbo's bytes), 'float32' or 'float16'; layout 'HWC' (default) or 'CHW' (channel planes,
        as PyTorch models take them); scale, bias: per channel (or one number), applied to float output as
        v * scale + bias with float32 rounding after each step (the same values as NumPy's
        img.astype(np.float32) * scale + bias); float16 rounds that to nearest even. E.g. ImageNet normalisation:
        scale = 1 / (255 * std), bias = -mean / std.

        check: only with a RepeatCache (Decoder(cache=...)): how a copy is known to be current, "full" (default)
        or "light" (see RepeatCache)."""
        global _busy
        if not self._async:
            self._conclude_job()
        self._coefs = self._coefs_pending = None    # first: a call that raises below leaves no older handle behind
        if self.cache is not None and not self._cache_busy and not self._async:
            return self.cache._decode(self, src, colorspace, timings, check, dtype, layout, scale, bias)[0]
        self._hp = [("call", perf_counter())] if self.profile == "host" else None
        from . import _route
        entered = _route.enter()
        with _busy_lock:
            others, _busy = _busy, _busy + 1               # other Decoders decoding right now
        try:
            return self._decode_src(src, colorspace, timings, dtype, layout, scale, bias, others)
        finally:
            with _busy_lock:
                _busy -= 1
            _route.leave(entered)

    def _decode_src(self, src, colorspace, timings, dtype, layout, scale, bias, others):
        self._epi = _epilogue(dtype, layout, scale, bias, 1 if colorspace == "GRAY" else 3)
        if not self._async:
            self._conclude_job()
        if colorspace not in COLORSPACES:
            raise ValueError(f"colorspace must be one of {COLORSPACES}")
        self.stats = {}
        self.warning = None
        self._finish_pending()
        self._coefs = self._coefs_pending = None
        if self.keep_coefs is not None and self.keep_coefs not in HANDLE_FORMS:
            raise ValueError(f"keep_coefs must be None or one of {HANDLE_FORMS}, got {self.keep_coefs!r}")
        self.reason = _selfcheck.ensure(self.device.id, "decode") or _selfcheck.ensure_modes(
            self.device.id, _selfcheck.mode_parts(self.fast_dct, self.fast_upsample, self._epi, colorspace))
        if self.reason is not None:                # the GPU path is off for this process (see _selfcheck)
            self._ahead = self._raw_ahead = None
            return None
        ahead, self._ahead, self._raw_ahead = self._ahead, None, None
        try:
            if ahead is not None and self._same_source(ahead[0], src):
                buf, h, self._raw_ahead = ahead[1], ahead[2], ahead[3]     # staged by upload_ahead()
                self.stats["read"] = "staged"
            elif isinstance(src, (str, os.PathLike)) and (self.read == "streamed" or (self.read == "auto" and not others)):
                self.stats["read"] = "streamed"
                buf, h = self._open_streamed(src)
            else:
                self.stats["read"] = "whole" if isinstance(src, (str, os.PathLike)) else "memory"
                buf = self._load(src)
                h = self._parse(buf)
        except Unsupported as e:
            self.reason = str(e)
            return None
        try:
            return self._decode_parsed(buf, h, colorspace, timings)
        finally:
            self._raw_ahead = None
            self._finish_pending()

    def _host(self, name):
        if self._hp is not None:
            self._hp.append((name, perf_counter()))

    def _parse(self, buf):
        """parse(buf), skipped when the file begins with the very header bytes of the last file parsed here (files
        from one writer at one size share them: the size, the tables and the scan's layout are all in those bytes,
        and parse() reads nothing after them)."""
        last = self._last
        if last is not None:
            k = len(last[0])
            if len(buf) >= k and buf[:k].tobytes() == last[0]:
                return last[1]
        h = parse(buf)
        self._last = (buf[:h.scan_start].tobytes(), h) if h.scan_start <= HEADER_MEMO else None
        return h

    def _finish_pending(self):
        """A streamed file whose upload never started (declined before the GPU work): close it."""
        if getattr(self, "_pending", None) is not None:
            self._pending[0].close()
        self._pending = None

    def _decode_parsed(self, buf, h, colorspace, timings):
        if len(buf) - h.scan_start > MAX_INPUT:
            self.reason = "entropy-coded data larger than 256 MB"
            return None
        if self.max_pixels and h.width * h.height > self.max_pixels:
            self.reason = f"more than max_pixels ({self.max_pixels})"
            return None
        self._host("read, parse")
        self.stats["subsampling"] = h.subsampling
        self.stats["restart_interval"] = h.dri
        try:
            with self.device, self.stream:
                out = self._decode(buf, h, colorspace, timings)
        except cp.cuda.memory.OutOfMemoryError:
            self.stream.synchronize()
            self.reason = "out of GPU memory"
            return None
        except GuardBandError:
            raise
        except (RuntimeError, ImportError, cp.cuda.compiler.CompileException) as e:
            # a CUDA error, a kernel that does not compile, a missing NVRTC: a decline, and the GPU path goes off
            self._guards = []
            self.reason = _selfcheck.gpu_error(self.device.id, e)
            return None
        return out

    def _decode(self, buf, h, colorspace, timings, eager=True):
        s = self.stream
        ev = [cp.cuda.Event() for _ in range(4)] if timings is not None else None
        if ev:
            ev[0].record(s)
        hp = self._hp
        marks = [] if self.profile and hp is None else None

        def mark(name):
            if hp is not None:
                hp.append((name, perf_counter()))
            elif marks is not None:
                e = cp.cuda.Event()
                e.record(s)
                marks.append((name, e))
        mark("start")
        T = _device_layout(h, not self.fast_upsample)
        mark("tables")
        n = len(buf)
        s0 = h.scan_start
        ne = n - s0
        S, NCP = chunking(ne)
        if self.S:
            S = int(self.S)
            NCP = env_int("EGJ_NCP", 0, 1, 255) or max(S // 512, 8) - 1
        nblk, bpm = T["nblk"], T["bpm"]
        if 2 * nblk > 8 * ne:                      # every block takes at least 2 bits (DC code + EOB)
            self.reason = "entropy-coded data too short for the image size"
            return None
        dri = h.dri
        nseg = _ceil(T["nmcu"], dri) if dri else 1
        nmax = (8 * ne) // S + nseg + 2
        W, H = h.width, h.height
        mode = {"RGB": 0, "BGR": 1, "GRAY": 2}[colorspace]
        if h.ncomp == 1 and mode != 2:
            mode = 3
        nout = H * W * (1 if mode == 2 else 3) * (np.dtype(self._epi["dtype"]).itemsize if self._epi else 1)
        # peak GPU memory, checked before anything is allocated
        need = (2 * ne + PAD + nmax * (60 + 12 * NCP) + nblk * 133 + T["plane_bytes"] + nout + nseg * 16 + (ne // TILE + 2) * 16)
        if self.canary:
            need += 24 * 2 * self.GUARD            # the test mode's guard bands, two for each of about 24 buffers
        budget = self.max_gpu_bytes
        if budget is None:
            budget = _selfcheck.budget(self.device.id)     # 90% of the free GPU memory
        if need > budget:
            self.reason = f"needs about {need >> 20} MB of GPU memory, more than allowed ({budget >> 20} MB)"
            return None
        mark("memory check")
        self._guards = []
        A = self._alloc
        raw, self._raw_ahead = self._raw_ahead, None       # the copy upload_ahead() queued (None: not staged)
        if raw is None:
            if (ne <= MAPPED_BELOW and self._pin_dev is not None and self._pending is None
                    and getattr(buf, "base", None) is self._pin and (self._pad + s0) % 64 == 0):
                raw = self._pin_dev[self._pad + s0:self._pad + n]      # read in place
                self.stats["upload"] = "mapped"
            else:
                raw = A(max(ne, 1), cp.uint8)
                self._upload(raw, buf, s0, s)
        if ev:
            ev[1].record(s)
        mark("upload")
        ntile = ne // TILE + 1
        small = nseg <= SMALL_TABLE_SEGS and nmax <= SMALL_TABLE_CHUNKS and not self._large_route
        # the state words: one launch sets them all (zeros, and the data's length as the end) and they are read
        # back in one piece at the end
        rounds, max_passes = sync_budget(ne)
        nstate = STATE + (SYNC_GROUP if eager else max_passes)
        state = A(nstate, cp.int32)
        _k("state_init")((1,), (128,), (state, _i32(nstate), _i32(ne)))
        err, end, nbits, found = _at(state), _at(state, 1), _at(state, 2), _at(state, 3)
        tct = A(2 * (ntile + 1), cp.int32)         # scan2_small takes their first elements as 0: no clearing
        toff, roff = _at(tct), _at(tct, ntile + 1)   # tile counts of kept bytes and of restart markers, then their scans
        _k("unstuff_count")((ntile,), (256,), (raw, _i32(ne), _i32(dri), _at(tct, 1), _at(tct, ntile + 2), end))
        _k("scan2_small")((1,), (1024,), (toff, roff, _i32(ntile + 1)))    # both scans, in place, one launch
        words = A(_ceil(ne + PAD, 4) * 4, cp.uint8)    # unstuff_scatter writes every byte of it, zeros included
        seg = A(nseg + 1, cp.uint32, zero=not small)   # small: chunk_table writes what unstuff_scatter does not
        _k("unstuff_scatter")((ntile,), (256,), (raw, _i32(ne), _i32(dri), toff, roff, end, words, nbits, seg,
                                                 _i32(nseg), found, err, _i32(words.size)))
        del raw, tct
        mark("unstuff")
        cstart = A(nmax, cp.uint32)
        cend = A(nmax, cp.uint32)
        if small:
            # the chunk table in one single-block launch: fix_segs, chunk counts, their scan, make_chunks, cseg = 0
            cfirst = A(nseg + 1, cp.int32)
            cseg = A(nmax, cp.int32)
            _k("chunk_table")((1,), (1024,), (seg, _i32(nseg), found, nbits, err, _i32(S), cfirst, cstart, cend, cseg,
                                              _i32(nmax)))
        else:
            _k("fix_segs")((_ceil(nseg + 1, 256),), (256,), (seg, _i32(nseg), found, nbits, err))
            lens = (seg[1:] - seg[:-1]).astype(cp.int64)
            ck = cp.maximum((lens + S - 1) // S, 1).astype(cp.int32)
            cfirst = A(nseg + 1, cp.int32, zero=True)
            cfirst[1:] = cp.cumsum(ck)
            del lens, ck
            cseg = A(nmax, cp.int32, zero=True)
            grid = (_ceil(nmax, 128),)
            _k("make_chunks")(grid, (128,), (seg, _i32(nseg), _i32(S), cfirst, cstart, cend, cseg, _i32(nmax)))
        nch_p = _at(cfirst, nseg)
        st_in = A(nmax, cp.uint64)
        st_out = A(nmax, cp.uint64)
        cnt = A(nmax, cp.int32, zero=not small)    # small: first_blocks reads only the chunks there are
        cps = A(nmax * NCP, cp.uint64)
        cpc = A(nmax * NCP, cp.int32)
        tabs, hdr, ntab = T["tab"], T["hdr"], T["ntab"]
        tsh = ntab * TABW * 4                  # dynamic shared memory: the Huffman tables in use
        nw = _u32(words.size // 4)
        mark("chunk table")
        # synchronisation passes. Eager mode: launch SYNC_GROUP passes and carry on without waiting; whether they
        # converged is read back with the error word at the end (a pass after the fixed point exits at once). If
        # they did not, the whole decode is redone with host checks after every group of passes.
        sync = _k("sync_pass")
        sgrid = (_ceil(nmax, 128),)
        passes, k = None, 0
        while passes is None and k < max_passes:
            g = min(SYNC_GROUP, max_passes - k)
            for j in range(k, k + g):
                prev = _at(state, STATE + j - 1) if j else _u64(0)
                sync(sgrid, (128,), (words, nw, tabs, hdr, nch_p, cstart, cend, cseg, seg, _i32(S), st_in, st_out,
                                     cnt, cps, cpc, prev, _at(state, STATE + j), _i32(nmax), _i32(ntab), _i32(NCP),
                                     _i32(rounds)),
                     shared_mem=tsh)
                mark(f"sync pass {j}")
            if eager:
                passes = -1                    # decided at the end
                break
            ch = state[STATE + k:STATE + k + g].get(stream=s)
            for j, v in enumerate(ch):
                if not v:
                    passes = k + j + 1
                    break
            k += g
        if passes is None:
            self.reason = "Huffman chunks did not synchronise"
            return None
        self.stats["passes"] = passes
        del st_out
        seglen = dri * bpm if dri else nblk
        if small:                                  # one launch (cnt is scanned in place)
            first = A(nmax, cp.int64)
            _k("first_blocks")((1,), (1024,), (cnt, nch_p, cseg, cfirst, _i64(seglen), first, _i32(nmax)))
        else:
            excl = A(nmax + 1, cp.int64, zero=True)
            cp.cumsum(cnt, dtype=cp.int64, out=excl[1:])
            first = cseg.astype(cp.int64) * seglen + excl[:nmax] - excl[cfirst[cseg]]
            del excl
        del cnt
        mark("prefix sums")
        coef = A(nblk * 64, cp.int16)
        dcd = A(nblk, cp.int32)
        nq = A(nblk, cp.uint8)
        _k("decode_write")((_ceil(nmax * (NCP + 1), 256),), (256,), (words, nw, tabs, hdr, nch_p, cstart, cend, cseg, seg,
                                                                    cfirst, st_in, cps, cpc, _i32(S), first, _i64(seglen),
                                                                    _i64(nblk), coef, dcd, nq, err, _i32(nmax),
                                                                    _i32(ntab), _i32(NCP)), shared_mem=tsh)
        del words, st_in, first, cstart, cend, cseg, cfirst, cps, cpc
        mark("write pass")
        self._dc_prediction(dcd, T, dri)
        if ev:
            ev[2].record(s)
        mark("dc prediction")
        arrays = [coef, nq, dcd]
        del coef, dcd, nq                          # (the pixel stage lets go of them once the IDCT is queued)
        if self.keep_coefs is not None:            # handed out by take_coefs() once this decode has concluded
            self._coefs_pending = (self.keep_coefs, h, self.device.id, *arrays)
        out = self._pixels(T, h, arrays, err, mode, mark)
        if ev:
            ev[3].record(s)
        resd = state[:STATE + SYNC_GROUP]
        if self._async and eager:
            # decode_async: the GPU work is queued; DecodeJob.result() reads the error word later
            done = cp.cuda.Event(disable_timing=True)
            done.record(s)
            return _Queued(done, resd, out, (buf, h, colorspace, timings, s0, marks, ev),
                           (self._epi, self.fast_dct, self.fast_upsample, self.strict), T)
        self._host("result words")
        res = resd.get(stream=s)
        self._host("wait, readback")
        return self._conclude(res, out, buf, h, colorspace, timings, s0, marks, ev, eager)

    def _pixels(self, T, h, arrays, err, mode, mark):
        """The pixel stage, queued on the current stream: IDCT of the coefficient arrays into planes, then
        upsampling and colour conversion into the output (the caller's buffer, or a new array), which is returned.
        arrays: [coef, nq, dc]; the list is emptied once the IDCT is queued, so that arrays nobody else holds go
        back to the pool before the output is allocated. err: the device word that takes the IDCT's error bits."""
        A = self._alloc
        nblk, W, H = T["nblk"], h.width, h.height
        coef, nq, dcd = arrays
        planes = A(T["plane_bytes"], cp.uint8)
        if self.fast_dct:
            _k("idct_ifast")((_ceil(nblk, 32),), (256,), (coef, nq, dcd, _ifast_q(T, h), T["pix"], _i64(nblk), planes,
                                                          err))
        else:
            _k("idct_islow")((_ceil(nblk, 32),), (256,), (coef, nq, dcd, T["q"], T["pix"], _i64(nblk), planes, err))
        del coef, dcd, nq
        arrays.clear()
        mark("idct")
        C = 1 if mode == 2 else 3
        epi = self._epi
        shape = (C, H, W) if epi and epi["chw"] else (H, W, C)
        dt = cp.dtype(epi["dtype"]) if epi else cp.dtype(cp.uint8)
        if self._out is not None:                  # the caller's buffer (decode_async(out=...))
            o = self._out
            if (o.shape != shape or o.dtype != dt or not o.flags.c_contiguous or o.device.id != self.device.id
                    or o.data.ptr % dt.itemsize):           # (a misaligned float array would make the kernel fault)
                raise ValueError(f"out must be a C-contiguous {dt} array of shape {shape} on device {self.device.id}, "
                                 "aligned to its element size")
            out = o
        else:
            out = A(shape, dt)
        if epi:
            e = np.zeros(8, np.float32)
            e[0:3], e[3:6] = epi["scale"], epi["bias"]
            ei = e.view(np.int32)
            ei[6], ei[7] = int(epi["dtype"] == "float16"), int(epi["chw"])
            ed = cp.asarray(e)
            _k("color_ep")((_ceil(W, 1024), H), (128,), (planes, T["pix"], T["up"], _i32(W), _i32(H), _i32(mode), out, ed))
        else:
            _k("color")((_ceil(W, 1024), H), (128,), (planes, T["pix"], T["up"], _i32(W), _i32(H), _i32(mode), out))
        mark("colour")
        del planes
        return out

    def _conclude(self, res, out, buf, h, colorspace, timings, s0, marks, ev, eager):
        """The checks after the GPU work: convergence of the eager passes, guard bands, the error word, the trailer."""
        kept, self._coefs_pending = self._coefs_pending, None
        e, endpos = int(res[0]), int(res[1])
        if eager:
            flags = [int(v) for v in res[STATE:]]
            if flags[-1]:                      # not converged within the eager passes: redo with host checks
                self._guards = []
                self._redone = True
                with self.device, self.stream:
                    return self._decode(buf, h, colorspace, timings, eager=False)
            self.stats["passes"] = flags.index(0) + 1
        if self.canary:
            self._check_guards()
        e, warned = warnings_of(e, buf, s0 + endpos)
        if e:
            self.reason = "; ".join(v for b, v in ERRORS.items() if e & b)
            return None
        why = check_trailer(buf, s0 + endpos)
        if why:
            self.reason = why
            return None
        if warned:
            why = warned_reason(warned, self.strict)
            if why is not None:
                self.reason = why
                return None
            self.warning = "; ".join(warned)
        if self._hp is not None:
            hp = self._hp + [("checks", perf_counter())]
            self.stats["host_ms"] = [(hp[k][0], round((hp[k][1] - hp[k - 1][1]) * 1e3, 4)) for k in range(1, len(hp))]
        if marks:
            self.stats["profile_ms"] = {marks[k][0]: round(cp.cuda.get_elapsed_time(marks[k - 1][1], marks[k][1]), 3)
                                        for k in range(1, len(marks))}
        if ev:
            timings["upload"] = timings.get("upload", 0) + cp.cuda.get_elapsed_time(ev[0], ev[1]) / 1000
            timings["huffman"] = timings.get("huffman", 0) + cp.cuda.get_elapsed_time(ev[1], ev[2]) / 1000
            timings["pixels"] = timings.get("pixels", 0) + cp.cuda.get_elapsed_time(ev[2], ev[3]) / 1000
        if kept is not None and not warned:        # the decode is good: its coefficients may be handed out (not for
            self._coefs = self._handle(*kept)              # a file with a warning: a strict decoder might use the copy)
        return out

    def _handle(self, form, h, dev, coef, nq, dcd):
        """The CoefHandle of a concluded decode's arrays, in the form asked for. "packed": made on the GPU here
        (about 2 ms for a 30 MP photo, with one wait). An image the packed form cannot hold keeps the raw form, and
        one there is no memory to pack is not kept."""
        nblk = int(nq.size)
        if form != "raw" and nblk * 138 >= 1 << 32:
            form = "raw"                           # the packer's 32-bit offsets would wrap
        if form != "raw" and self._copy_check()["packed"] is not None:
            form = "raw"                           # the packed form's kernels failed their check on this GPU (or
                                                   # memory was short for the check)
        if form == "packed":
            from . import _packed
            try:
                with self.device, self.stream:
                    made = _packed.pack(self._alloc, coef, nq, dcd, nblk, self.stream)
            except cp.cuda.memory.OutOfMemoryError:
                self.stream.synchronize()
                self._guards = []
                return None
            if self.canary:
                self._check_guards()
            if made is not None:
                return CoefHandle("packed", h, dev, made, nblk, _key=_HANDLE_KEY)
        return CoefHandle("raw", h, dev, (coef, nq, dcd), nblk, _key=_HANDLE_KEY)

    def _copy_check(self):
        """The check of the copy kernels on this decoder's GPU ({part: None or why it failed}; _copycheck). It runs
        once per GPU, at the first use of a form other than "raw"."""
        c = self._copychk
        if c is None:
            from . import _copycheck
            c = _copycheck.state(self.device.id)
            if _copycheck.NO_MEMORY not in c.values():         # (memory was short: not checked, so asked again at
                self._copychk = c                              # the next use, as the process-wide state is)
        return c

    def take_coefs(self):
        """The CoefHandle of the last decode() or decode_async() of this decoder, if keep_coefs was set and the
        decode succeeded (a queued decode is concluded first); else None. The decoder keeps no reference: the
        handle, and the GPU memory it holds (handle.nbytes), are the caller's."""
        self._conclude_job()
        handle, self._coefs = self._coefs, None
        return handle

    def decode_from_handle(self, handle, colorspace="RGB", timings=None, *, dtype="uint8", layout="HWC", scale=None,
                           bias=None):
        """The image a CoefHandle holds, as decode() returns it for the file the handle came from (the same
        bytes), without reading the file and without the entropy decode: only the pixel stage runs. colorspace and
        the output options are decode()'s; the decoder's fast_dct and fast_upsample settings apply as they do to
        a file. timings: optional dict, gets 'pixels' (seconds of GPU time). The handle is not changed and can be
        used again, by any Decoder on its device. Returns None when the GPU path is off or memory is short
        (`reason` says why)."""
        self._conclude_job()
        self._coefs = self._coefs_pending = None    # first: a call that raises below leaves no older handle behind
        if not isinstance(handle, CoefHandle):
            raise TypeError("handle must be a CoefHandle (Decoder.take_coefs())")
        if handle.freed:
            raise ValueError("the handle was freed")
        if handle.device != self.device.id:
            raise ValueError(f"the handle is on device {handle.device}, this decoder on device {self.device.id}")
        if colorspace not in COLORSPACES:
            raise ValueError(f"colorspace must be one of {COLORSPACES}")
        self._hp = [("call", perf_counter())] if self.profile == "host" else None
        self._epi = _epilogue(dtype, layout, scale, bias, 1 if colorspace == "GRAY" else 3)
        self._finish_pending()
        self._ahead = self._raw_ahead = None
        self.stats = {"read": "handle"}
        self.warning = None
        self._err_bits = 0
        if self.keep_coefs is not None and self.keep_coefs not in HANDLE_FORMS:
            raise ValueError(f"keep_coefs must be None or one of {HANDLE_FORMS}, got {self.keep_coefs!r}")
        self.reason = _selfcheck.ensure(self.device.id, "decode") or _selfcheck.ensure_modes(
            self.device.id, _selfcheck.mode_parts(self.fast_dct, self.fast_upsample, self._epi, colorspace))
        if self.reason is not None:                # the GPU path is off for this process (see _selfcheck)
            return None
        hh = handle._h
        if self.max_pixels and hh.width * hh.height > self.max_pixels:
            self.reason = f"more than max_pixels ({self.max_pixels})"
            return None
        try:
            with self.device, self.stream:
                return self._from_handle(handle, colorspace, timings)
        except cp.cuda.memory.OutOfMemoryError:
            self.stream.synchronize()
            self.reason = "out of GPU memory"
            return None
        except GuardBandError:
            raise
        except (RuntimeError, ImportError, cp.cuda.compiler.CompileException) as e:
            self._guards = []
            self.reason = _selfcheck.gpu_error(self.device.id, e)
            self._err_bits = -1
            return None

    def _from_handle(self, handle, colorspace, timings):
        s, h = self.stream, handle._h
        ev = [cp.cuda.Event() for _ in range(2)] if timings is not None else None
        if ev:
            ev[0].record(s)
        hp = self._hp
        marks = [] if self.profile and hp is None else None

        def mark(name):
            if hp is not None:
                hp.append((name, perf_counter()))
            elif marks is not None:
                e = cp.cuda.Event()
                e.record(s)
                marks.append((name, e))
        mark("start")
        T = _device_layout(h, not self.fast_upsample)
        mark("tables")
        W, H = h.width, h.height
        mode = {"RGB": 0, "BGR": 1, "GRAY": 2}[colorspace]
        if h.ncomp == 1 and mode != 2:
            mode = 3
        nout = H * W * (1 if mode == 2 else 3) * (np.dtype(self._epi["dtype"]).itemsize if self._epi else 1)
        packed = handle.form == "packed"
        need = T["plane_bytes"] + nout             # peak GPU memory, checked before anything is allocated
        if packed:
            need += handle._nblk * 133             # the three arrays the copy is unpacked into
        if self.canary:
            need += 6 * 2 * self.GUARD
        budget = self.max_gpu_bytes
        if budget is None:
            budget = _selfcheck.budget(self.device.id)
        if need > budget:
            self.reason = f"needs about {need >> 20} MB of GPU memory, more than allowed ({budget >> 20} MB)"
            return None
        mark("memory check")
        self._guards = []
        err = self._alloc(1, cp.int32, zero=True)
        if packed:
            from . import _packed
            idx, data, dlen = handle._parts
            nblk = handle._nblk
            arrays = [self._alloc(nblk * 64, cp.int16), self._alloc(nblk, cp.uint8), self._alloc(nblk, cp.int32)]
            _packed.unpack(idx, data, dlen, nblk, arrays[0], arrays[2], arrays[1], err)
            mark("unpack")
        else:
            arrays = list(handle._parts)
        # keep_coefs in another form than the handle's: take_coefs() gives the coefficients in that form
        held = tuple(arrays) if self.keep_coefs not in (None, handle.form) else None
        out = self._pixels(T, h, arrays, err, mode, mark)
        if ev:
            ev[1].record(s)
        self._host("result word")
        e = int(err.get(stream=s)[0])
        self._host("wait, readback")
        if self.canary:
            self._check_guards()
        if e:
            self._err_bits = e
            self.reason = "; ".join(v for b, v in ERRORS.items() if e & b)
            return None
        if hp is not None:
            hp = hp + [("checks", perf_counter())]
            self.stats["host_ms"] = [(hp[k][0], round((hp[k][1] - hp[k - 1][1]) * 1e3, 4)) for k in range(1, len(hp))]
        if marks:
            self.stats["profile_ms"] = {marks[k][0]: round(cp.cuda.get_elapsed_time(marks[k - 1][1], marks[k][1]), 3)
                                        for k in range(1, len(marks))}
        if ev:
            timings["pixels"] = timings.get("pixels", 0) + cp.cuda.get_elapsed_time(ev[0], ev[1]) / 1000
        if held is not None:
            self._coefs = self._handle(self.keep_coefs, h, handle.device, *held)
        return out

    def _dc_prediction(self, dcd, T, dri):
        """DC values from differences, in place: a running sum per component, restarting at each restart
        interval (dc_scan). One launch for a small image, two (sums per CUDA block, then the values) otherwise."""
        nmcu = T["nmcu"]
        comp = T.get("comp2")
        if comp is None:
            comp = T["comp2"] = _u32(sum(b[0] << (2 * k) for k, b in enumerate(T["blocks"])))
        if nmcu <= DC_ONE_BLOCK:
            per, nb = _ceil(nmcu, 1024), 1
        else:
            per = max(DC_RUN, _ceil(nmcu, 1024 * DC_BLOCKS))
            nb = _ceil(nmcu, 1024 * per)
        f = _k("dc_scan")
        args = (dcd, _i32(nmcu), _i32(T["bpm"]), comp, _i32(dri), _i32(per))
        bs = dcd                               # (one block: not used)
        if nb > 1:
            bs = self._alloc(4 * nb, cp.uint32)
            f((nb,), (1024,), args + (bs, _i32(0)))
        f((nb,), (1024,), args + (bs, _i32(1)))
