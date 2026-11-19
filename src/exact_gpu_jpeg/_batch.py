"""Batched GPU decode (Decoder.decode_batch): JPEGs that share their Huffman tables and sampling factors are decoded
as one stream by one set of kernel launches (_batch_kernels.cu), instead of about 45 small launches per image.
The arithmetic is the single-image path's, so the pixels are the same."""
from __future__ import annotations

import math
import os
from dataclasses import replace
import threading
from importlib import resources

import cupy as cp
import numpy as np

from . import _decoder as _d
from ._header import MAX_DIMENSION, MAX_HEADER, MAX_SEGMENTS, Unsupported, check_trailer, env_int, parse

MIN_GROUP = 2                              # fewer images sharing their tables: decode() one by one
MAX_IMAGES = 1024                          # per set of launches
MAX_RAW = 64 << 20                         # bytes of entropy-coded data per set of launches, at most (cards with
#                                            less than 8 GB get less: _selfcheck.settings)
MAX_SEGS = _d.SMALL_TABLE_SEGS - 8         # chunk_table_b is a single-block kernel
MAX_CHUNKS = _d.SMALL_TABLE_CHUNKS
SEG_GAP = 1 << 30
TILE = _d.TILE
KERNELS = ("unstuff_count_b", "scan2_small", "unstuff_scatter_b", "chunk_table_b", "sync_pass", "decode_write_b",
           "dc_apply_b", "idct_islow_b", "color_b")

_bmods = {}
_tabs = {}
_tab_lock = threading.Lock()
_i32, _u32, _u64 = np.int32, np.uint32, np.uint64


class _Single(Exception):
    """This set of images cannot be decoded as one batch (or did not synchronise): decode them one by one."""


def _bk(name):
    dev = cp.cuda.Device().id
    with _d._mod_lock:
        m = _bmods.get(dev)
        if m is None:
            src = _d._kernel_source() + "\n" + resources.files(__package__).joinpath("_batch_kernels.cu").read_text()
            m = _bmods[dev] = cp.RawModule(code=src, options=("--std=c++14",))
            for n in KERNELS:
                m.get_function(n)             # compile now, under the lock
        return m.get_function(name)


def _group_key(h):
    return (tuple(tuple(t[0]) + (t[1],) for t in h.dc_tabs + h.ac_tabs), tuple(h.td), tuple(h.ta), tuple(h.samp),
            h.ncomp)


def _group_tables(h, key):
    """The Huffman tables and MCU layout of a group, on the device (shared by every decoder, like _device_layout)."""
    dk = (key, cp.cuda.Device().id)
    with _tab_lock:
        r = _tabs.get(dk)
    if r is not None:
        return r
    r = _d._tables(h)
    r["tab"], r["hdr"] = cp.asarray(r["tab"]), cp.asarray(r["hdr"])
    cp.cuda.get_current_stream().synchronize()   # uploaded on this stream; shared by every decoder from here on
    with _tab_lock:
        if len(_tabs) > 64:
            _tabs.clear()
        _tabs[dk] = r
    return r


_hdrs = {}                      # significant header bytes -> Header (see _parse)
_recent = []                    # (header bytes up to the scan, Header) of the last few files, most recent first
_quants = {}
_SKIP = frozenset({0xE1} | set(range(0xE3, 0xEE))) | {0xEF, 0xFE}   # APP1, APP3..APP13, APP15, COM: parse() ignores
                                                                     # their content
_KEEP = frozenset((0xDB, 0xC4, 0xDD, 0xE0, 0xEE))        # DQT, DHT, DRI, APP0 (JFIF), APP14 (Adobe)


def _parse(data):
    """parse(), remembering headers. What parse() returns depends only on the DQT, DHT, SOF, DRI, APP0, APP14 and
    SOS segments, in their order; and on the SOF's width and height only through Header.width / height (after the
    range check). So files whose segments are equal apart from the size (and apart from EXIF and other APPn / COM
    content, which parse() skips) share one parsed Header, with their own width, height and scan start: files from
    one camera or encoder are parsed once. Only plainly structured headers take this path (marker after marker, no
    fill bytes); anything else, and every header not seen before, goes through parse() itself."""
    if not isinstance(data, (bytes, bytearray)):
        return parse(data)
    for pre, h in _recent:                               # the same bytes up to the scan: the same Header
        if data.startswith(pre):
            return h
    h = _parse_keyed(data)
    with _tab_lock:
        _recent.insert(0, (bytes(data[:h.scan_start]), h))
        del _recent[4:]
    return h


def _parse_keyed(data):
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return parse(data)
    i, parts, W, H, segments = 2, [], 0, 0, 0
    while True:
        segments += 1
        if i > MAX_HEADER or segments > MAX_SEGMENTS or i + 4 > n or data[i] != 0xFF:
            return parse(data)                           # (which applies the same bounds, and raises)
        m = data[i + 1]
        L = (data[i + 2] << 8) | data[i + 3]
        e = i + 2 + L
        if L < 2 or e > n:
            return parse(data)
        if m in _SKIP:
            pass
        elif m == 0xE2:                                  # APP2: parse() reads an ICC chunk's numbers (its rule), so the
            parts.append(data[i + 1:i + 2] + data[i + 4:min(e, i + 18)])     # identifier and the two numbers count
        elif m in _KEEP:
            parts.append(data[i + 1:e])
        elif m == 0xC0 or m == 0xC1:                     # SOF: everything but the height and width
            if L < 8:
                return parse(data)
            H, W = (data[i + 5] << 8) | data[i + 6], (data[i + 7] << 8) | data[i + 8]
            parts.append(data[i + 1:i + 5])
            parts.append(data[i + 9:e])
        elif m == 0xDA:                                  # SOS: the header ends here
            parts.append(data[i + 1:e])
            break
        else:
            return parse(data)
        i = e
    if not (0 < W <= MAX_DIMENSION and 0 < H <= MAX_DIMENSION):
        return parse(data)
    key = b"".join(parts)
    h0 = _hdrs.get(key)
    if h0 is None:
        h = parse(data)                                  # raises for what the GPU path does not decode
        if h.width == W and h.height == H and h.scan_start == e:
            if len(_hdrs) > 256:
                _hdrs.clear()
            _hdrs[key] = h
        return h
    return replace(h0, width=W, height=H, scan_start=e)


def _geom(h, T, fancy):
    """_decoder._geometry() for the batch tables, without NumPy arrays per image: pix and up as lists of ints (the
    same numbers), q kept per set of quantisers."""
    head = T.get("head")
    blocks = T["blocks"]
    bpm = len(blocks)
    if head is None:
        head = [0] * 38
        head[0] = bpm
        for k, (ci, dx, dy) in enumerate(blocks):
            head[1 + k], head[11 + k], head[21 + k] = ci, dx, dy
        for ci, (hs, vs) in enumerate(h.samp):
            head[31 + ci], head[34 + ci] = hs, vs
        T["head"] = head
    samp, nc, W, H = h.samp, h.ncomp, h.width, h.height
    hmax, vmax = max(x[0] for x in samp), max(x[1] for x in samp)
    mcux, mcuy = -(-W // (8 * hmax)), -(-H // (8 * vmax))
    pitch = [mcux * x[0] * 8 for x in samp] + [0] * (3 - nc)
    rows = [mcuy * x[1] * 8 for x in samp] + [0] * (3 - nc)
    o1 = pitch[0] * rows[0]
    o2 = o1 + pitch[1] * rows[1]
    off = []
    for o in (0, o1, o2):                                # int64 as two int32 words, low word first
        lo = o & 0xFFFFFFFF
        off += [lo - (1 << 32) if lo >> 31 else lo, o >> 32]
    up = [0] * 15                                        # fh[3], fv[3], dw[3], dh[3], fancy[3]
    for ci, (hs, vs) in enumerate(samp):
        fh, fv = hmax // hs, vmax // vs
        dw, dh = -(-(W * hs) // hmax), -(-(H * vs) // vmax)
        fy = {(2, 1): int(dw > 2), (1, 2): 1, (2, 2): int(dw > 2)}.get((fh, fv), 0) if fancy else 0
        up[ci], up[3 + ci], up[6 + ci], up[9 + ci], up[12 + ci] = fh, fv, dw, dh, fy
    qk = tuple(x.tobytes() for x in h.quant)
    q = _quants.get(qk)
    if q is None:
        q = np.zeros((3, 64), np.int32)
        for ci in range(nc):
            q[ci, _d.NAT] = h.quant[ci]
        if len(_quants) > 256:
            _quants.clear()
        q = _quants[qk] = q.reshape(-1)
    nmcu = mcux * mcuy
    return dict(pix=head + off + pitch + [mcux], up=up, q=q, nmcu=nmcu, nblk=nmcu * bpm,
                plane_bytes=o2 + pitch[2] * rows[2])


def _read(src):
    """The file's bytes; Unsupported for a file (or bytes input) over the size limit, before it is read."""
    if isinstance(src, (str, os.PathLike)):
        _d._file_size(src)
        with open(src, "rb", buffering=0) as f:
            return f.read()
    n = memoryview(src).nbytes
    if n > _d.MAX_FILE:
        raise Unsupported(_d._too_large(n))
    return src


def decode_batch(dec, srcs, colorspace, dtype, layout, scale, bias):
    if colorspace not in _d.COLORSPACES:
        raise ValueError(f"colorspace must be one of {_d.COLORSPACES}")
    epi = _d._epilogue(dtype, layout, scale, bias, 1 if colorspace == "GRAY" else 3)
    dec._conclude_job()
    dec._finish_pending()
    dec._ahead = dec._raw_ahead = None          # a file staged by upload_ahead() is dropped (its buffer is reused)
    srcs = list(srcs)
    out, reasons = [None] * len(srcs), [None] * len(srcs)
    dec.warnings = [None] * len(srcs)
    dec.reason = None
    dec.stats = {}
    groups, single = {}, []
    for i, src in enumerate(srcs):
        try:
            data = _read(src)
            h = _parse(data)
        except Unsupported as e:
            reasons[i] = str(e)
            continue
        except OSError as e:                   # a file that cannot be read: that file is declined, not the list
            reasons[i] = f"could not read the file ({type(e).__name__}: {e})"
            continue
        if dec.fast_dct:                       # the fast IDCT has no batch kernel
            single.append((i, data))
            continue
        groups.setdefault(_group_key(h), []).append((i, data, h))
    fancy = not dec.fast_upsample
    nbatched = 0
    dev = dec.device.id
    with dec.device, dec.stream:
        budget = dec.max_gpu_bytes
        if budget is None:
            try:
                budget = _d._selfcheck.budget(dev)         # 90% of the free GPU memory
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
                budget = 0                                 # (no part fits: every image goes one by one)
        max_raw = min(MAX_RAW, _d._selfcheck.settings(dec.device.id)["batch_data_bytes"])   # by the card's memory

        guard = 24 * 2 * dec.GUARD if dec.canary else 0     # the test mode's guard bands

        def flush(T, part):
            nonlocal nbatched
            if len(part) < MIN_GROUP or _d._selfcheck.failed(dev):     # (a CUDA error in an earlier part: none
                single.extend((p[0], p[1]) for p in part)              # of the rest runs on the GPU)
                return
            try:
                _decode_part(dec, T, part, colorspace, epi, out, reasons)
                nbatched += len(part)
            except _Single:
                single.extend((p[0], p[1]) for p in part)
            except cp.cuda.memory.OutOfMemoryError:
                dec.stream.synchronize()
                dec._guards = []
                for p in part:
                    out[p[0]] = None
                single.extend((p[0], p[1]) for p in part)
            except _d.GuardBandError:
                raise
            except (RuntimeError, ImportError, cp.cuda.compiler.CompileException) as e:
                # a CUDA error or a kernel that does not compile: the GPU path goes off; decode() declines the rest
                dec._guards = []
                _d._selfcheck.gpu_error(dec.device.id, e)
                for p in part:
                    out[p[0]] = None
                single.extend((p[0], p[1]) for p in part)

        for key, items in groups.items():
            if len(items) < MIN_GROUP:
                single.extend((i, data) for i, data, _ in items)
                continue
            try:
                T = _group_tables(items[0][2], key)
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
                dec.stream.synchronize()
                single.extend((i, data) for i, data, _ in items)   # decode() says why, one by one
                continue
            bpm = T["bpm"]
            part, raw, segs, need = [], 0, 0, 0
            for i, data, h in items:
                ne = memoryview(data).nbytes - h.scan_start
                if ne > _d.MAX_INPUT or (dec.max_pixels and h.width * h.height > dec.max_pixels):
                    single.append((i, data))          # decode() gives the reason
                    continue
                g = _geom(h, T, fancy)
                if 2 * g["nblk"] > 8 * ne:            # every block takes at least 2 bits (DC code + EOB)
                    reasons[i] = "entropy-coded data too short for the image size"
                    continue
                nseg = _d._ceil(g["nmcu"], h.dri) if h.dri else 1
                nt = ne // TILE + 1
                nmax = (8 * ne) // 2048 + nseg + 3
                C = 1 if colorspace == "GRAY" else 3
                mine = (2 * ne + nt * TILE + _d.PAD + nmax * (60 + 12 * 15) + (g["nblk"] + 96) * 133 + g["plane_bytes"]
                        + h.width * h.height * C * (np.dtype(epi["dtype"]).itemsize if epi else 1) + nseg * 40 + nt * 16
                        + 4096)
                if part and (len(part) >= MAX_IMAGES or raw + nt * TILE > max_raw or segs + nseg + 1 > MAX_SEGS
                             or need + mine + guard > budget):
                    flush(T, part)
                    part, raw, segs, need = [], 0, 0, 0
                part.append((i, data, h, g, ne, nseg, nt))
                raw, segs, need = raw + nt * TILE, segs + nseg + 1, need + mine
            flush(T, part)
    busy, dec._cache_busy = dec._cache_busy, True    # a batch does not use or fill a RepeatCache (as documented)
    try:
        for i, data in single:
            out[i] = dec.decode(data, colorspace, dtype=dtype, layout=layout, scale=scale, bias=bias)
            reasons[i] = dec.reason
            dec.warnings[i] = dec.warning
        dec.warning = None
    finally:
        dec._cache_busy = busy
    dec.reason = None
    dec.reasons = reasons
    dec.stats["batched"] = nbatched
    return out


def _decode_part(dec, T, part, colorspace, epi, out, reasons):
    """One set of launches for `part`: [(index, data, header, geometry, ne, nseg, tiles)], all with the tables T."""
    s = dec.stream
    N = len(part)
    bpm = T["bpm"]
    AL = 32 * bpm // math.gcd(32, bpm)        # block ranges start at multiples of 32 (idct) and of bpm (dc)
    mode = {"RGB": 0, "BGR": 1, "GRAY": 2}[colorspace]
    if part[0][2].ncomp == 1 and mode != 2:
        mode = 3
    C = 1 if mode == 2 else 3
    esize = np.dtype(epi["dtype"]).itemsize if epi else 1
    rows, pbs, obs, simg, sblk, slim, views = [], [], [], [], [], [], []
    rb = seg = blk = unit = pb = ob = total_ne = 0
    for k, (i, data, h, g, ne, nseg, nt) in enumerate(part):
        W, H, nblk = h.width, h.height, g["nblk"]
        xb = _d._ceil(W, 1024)
        rows.append((rb, ne, h.dri, rb // TILE, seg, nseg, W, H, blk, nblk, unit, xb))
        pbs.append(pb)
        obs.append(ob)
        last = blk + nblk
        if h.dri:
            step = h.dri * bpm
            for a in range(blk, blk + nseg * step, step):
                sblk.append(a)
                slim.append(min(a + step, last))
            simg.extend([k] * nseg)
        else:
            sblk.append(blk)
            slim.append(last)
            simg.append(k)
        sblk.append(last)                                      # the gap segment: no blocks
        slim.append(last)
        simg.append(k | SEG_GAP)
        nel = H * W * C
        views.append((ob, nel, (C, H, W) if epi and epi["chw"] else (H, W, C)))
        rb += nt * TILE
        seg += nseg + 1
        blk += _d._ceil(nblk, AL) * AL
        unit += xb * H
        pb += g["plane_bytes"]
        ob += nel if epi else _d._ceil(nel, 16) * 16
        total_ne += ne
    NS, ntile, Btot = seg, rb // TILE, blk
    S, NCP = _d.chunking(total_ne)
    if dec.S:
        S = int(dec.S)
        NCP = env_int("EGJ_NCP", 0, 1, 255) or max(S // 512, 8) - 1
    nmax = sum((8 * p[4]) // S + p[5] + 3 for p in part)
    if nmax > MAX_CHUNKS or NS > _d.SMALL_TABLE_SEGS or Btot >= 1 << 31 or rb > _d.MAX_INPUT or unit >= 1 << 31:
        raise _Single()
    D = np.zeros((N, 16), np.int32)
    D[:, :12] = np.array(rows, np.int64)
    D64 = D.view(np.int64)
    D64[:, 6], D64[:, 7] = pbs, obs
    pix, q, up = (np.array([p[3][name] for p in part], np.int32) for name in ("pix", "q", "up"))
    endv, nts = np.array([p[4] for p in part], np.int32), [p[6] for p in part]
    marks = [] if dec.profile else None

    def mark(name):
        if marks is not None:
            e = cp.cuda.Event()
            e.record(s)
            marks.append((name, e))
    mark("start")
    dec._guards = []
    A = dec._alloc
    try:
        pin = dec._pinned(rb)
    except Unsupported:                        # page-locked memory is short: these images go one by one
        raise _Single()
    for k, (i, data, h, g, ne, nseg, nt) in enumerate(part):
        o = rows[k][0]
        pin[o:o + ne] = np.frombuffer(memoryview(data).cast("B"), np.uint8)[h.scan_start:]
    raw = A(rb, cp.uint8)
    raw.set(pin, stream=s)
    Dd, timg = cp.asarray(D), cp.asarray(np.repeat(np.arange(N, dtype=np.int32), nts))
    simg_d = cp.asarray(np.array(simg, np.int32))
    sblk_d, slim_d = cp.asarray(np.array(sblk, np.int64)), cp.asarray(np.array(slim, np.int64))
    pix_d, q_d, up_d = cp.asarray(pix), cp.asarray(q), cp.asarray(up)
    end = A(N, cp.int32)
    end.set(endv)
    mark("upload")
    k_ = _bk
    tc = A(ntile + 1, cp.int32, zero=True)
    tr = A(ntile + 1, cp.int32, zero=True)
    k_("unstuff_count_b")((ntile,), (256,), (raw, Dd, timg, tc[1:], tr[1:], end))
    k_("scan2_small")((1,), (1024,), (tc, tr, _i32(ntile + 1)))
    words = A(_d._ceil(total_ne + _d.PAD, 4) * 4, cp.uint8)
    nbits = A(N, cp.uint32, zero=True)
    segd = A(NS + 1, cp.uint32, zero=True)
    found = A(N, cp.int32, zero=True)
    err = A(N, cp.int32, zero=True)
    k_("unstuff_scatter_b")((ntile,), (256,), (raw, Dd, timg, tc, tr, end, words, nbits, segd, found, err,
                                               _i32(words.size)))
    del raw
    mark("unstuff")
    cstart, cend = A(nmax, cp.uint32), A(nmax, cp.uint32)
    cfirst, cseg = A(NS + 1, cp.int32), A(nmax, cp.int32)
    k_("chunk_table_b")((1,), (1024,), (segd, simg_d, _i32(NS), Dd, found, nbits, err, tc[ntile:], _i32(S), cfirst,
                                        cstart, cend, cseg, _i32(nmax)))
    del tr
    nch_p = cfirst[NS:]
    st_in, st_out = A(nmax, cp.uint64), A(nmax, cp.uint64)
    cnt = A(nmax, cp.int32, zero=True)
    cps, cpc = A(nmax * NCP, cp.uint64), A(nmax * NCP, cp.int32)
    tabs, hdr, ntab = T["tab"], T["hdr"], T["ntab"]
    tsh = ntab * _d.TABW * 4
    nw = _u32(words.size // 4)
    mark("chunk table")
    # the eager passes of the single-image path; if they did not converge, the images are decoded one by one
    changed = A(_d.SYNC_GROUP, cp.int32, zero=True)
    sync = k_("sync_pass")
    for j in range(_d.SYNC_GROUP):
        prev = changed[j - 1:] if j else _u64(0)
        sync((_d._ceil(nmax, 128),), (128,), (words, nw, tabs, hdr, nch_p, cstart, cend, cseg, segd, _i32(S), st_in,
                                              st_out, cnt, cps, cpc, prev, changed[j:], _i32(nmax), _i32(ntab),
                                              _i32(NCP), _i32(_d.sync_budget(total_ne)[0])), shared_mem=tsh)
        mark(f"sync pass {j}")
    del st_out
    excl = A(nmax + 1, cp.int64, zero=True)
    cp.cumsum(cnt, dtype=cp.int64, out=excl[1:])
    first = sblk_d[cseg] + excl[:nmax] - excl[cfirst[cseg]]
    del excl, cnt
    mark("prefix sums")
    coef, dcd, nq = A(Btot * 64, cp.int16), A(Btot, cp.int32), A(Btot, cp.uint8)
    k_("decode_write_b")((_d._ceil(nmax * (NCP + 1), 256),), (256,),
                         (words, nw, tabs, hdr, nch_p, cstart, cend, cseg, segd, cfirst, st_in, cps, cpc, _i32(S), first,
                          sblk_d, slim_d, simg_d, coef, dcd, nq, err, _i32(nmax), _i32(ntab), _i32(NCP)), shared_mem=tsh)
    del words, st_in, first, cstart, cend, cseg, cfirst, cps, cpc
    mark("write pass")
    d2 = dcd.reshape(Btot // bpm, bpm)
    comp_of = [b[0] for b in T["blocks"]]
    for ci in sorted(set(comp_of)):
        cols = [c for c in range(bpm) if comp_of[c] == ci]
        a, w = cols[0], len(cols)
        cs = cp.cumsum(d2[:, a:a + w].reshape(-1), dtype=cp.int32)
        k_("dc_apply_b")((_d._ceil(cs.size, 256),), (256,), (cs, _i32(w), _i32(a), _i32(bpm), Dd, _i32(N), dcd,
                                                             _u32(cs.size)))
        del cs
    mark("dc prediction")
    planes = A(pb, cp.uint8)
    k_("idct_islow_b")((Btot // 32,), (256,), (coef, nq, dcd, q_d, pix_d, Dd, _i32(N), planes, err))
    del coef, dcd, nq
    mark("idct")
    if epi:
        e = np.zeros(8, np.float32)
        e[0:3], e[3:6] = epi["scale"], epi["bias"]
        ei = e.view(np.int32)
        ei[6], ei[7] = int(epi["dtype"] == "float16"), int(epi["chw"])
        ed = cp.asarray(e)
        res_all = A(ob, cp.dtype(epi["dtype"]))
    else:
        ed = _u64(0)
        res_all = A(ob, cp.uint8)
    k_("color_b")((unit,), (128,), (planes, pix_d, up_d, Dd, _i32(N), _i32(mode), res_all, ed))
    mark("colour")
    del planes
    res = cp.concatenate([err, end, changed]).get(stream=s)
    if int(res[-1]):                           # not converged within the eager passes
        dec._guards = []
        raise _Single()
    if dec.canary:
        dec._check_guards()
    if marks:
        dec.stats["profile_ms"] = {marks[j][0]: round(cp.cuda.get_elapsed_time(marks[j - 1][1], marks[j][1]), 3)
                                   for j in range(1, len(marks))}
    for k, (i, data, h, g, ne, nseg, nt) in enumerate(part):
        e, warned = _d.warnings_of(int(res[k]), data, h.scan_start + int(res[N + k]))
        if e:
            reasons[i] = "; ".join(v for b, v in _d.ERRORS.items() if e & b)
            continue
        why = check_trailer(data, h.scan_start + int(res[N + k]))
        if why:
            reasons[i] = why
            continue
        if warned:
            why = _d.warned_reason(warned, dec.strict)
            if why is not None:
                reasons[i] = why
                continue
            dec.warnings[i] = "; ".join(warned)
        o, nel, shape = views[k]
        out[i] = res_all[o:o + nel].reshape(shape)
