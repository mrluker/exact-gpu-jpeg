"""Strict JPEG header parser.

It accepts only what the GPU path decodes exactly, and it is deliberately stricter than libjpeg-turbo:
anything unusual raises :class:`Unsupported`, so the caller falls back to a CPU decoder. The rules that
decide *what* libjpeg-turbo does (colour space guess, default Huffman tables, table validation) follow
libjpeg-turbo 3.1.2 (jdmarker.c, jdapimin.c, jdhuff.c, jstdhuff.c).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_DIMENSION = 65500          # JPEG_MAX_DIMENSION
MAX_BLOCKS_IN_MCU = 10         # D_MAX_BLOCKS_IN_MCU


MAX_HEADER = 16 << 20       # bytes before the scan: a longer header is declined (real files: kilobytes to a few MB)
MAX_SEGMENTS = 4096         # marker segments before the scan, and again after it (real files: tens)


_ENV_WARNED = set()


def env_int(name, default, lo, hi):
    """The integer environment variable `name` if it is set to a whole number in lo..hi; else `default`. A value
    that is set but not such a number gives one RuntimeWarning naming it (and the default is used): a mistyped
    setting must neither stop the import nor make every call raise."""
    import os
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        n = int(v.strip().replace("_", ""))
    except ValueError:
        n = None
    if n is None or not lo <= n <= hi:
        if name not in _ENV_WARNED:
            _ENV_WARNED.add(name)
            import warnings
            warnings.warn(f"exact_gpu_jpeg: {name}={v!r} is not a whole number from {lo} to {hi}; ignored",
                          RuntimeWarning, stacklevel=2)
        return default
    return n


ICC_BAD = "Corrupt JPEG data: bad ICC marker"     # JWRN_BOGUS_ICC


def icc_bogus(chunks):
    """True if the ICC_PROFILE APP2 segments before the scan, as (seq_no, num_markers) in file order, are ones
    libjpeg-turbo's jpeg_read_icc_profile() (jdicc.c) warns about: num_markers not the same in all, a seq_no of 0 or
    above num_markers, a seq_no twice, or one missing. (TurboJPEG reads the profile with the header.)"""
    if not chunks:
        return False
    num, seen = None, set()
    for seq, cnt in chunks:
        if num is None:
            num = cnt
        elif cnt != num:
            return True
        if seq == 0 or seq > num or seq in seen:
            return True
        seen.add(seq)
    return len(seen) != num


def _icc_chunk(seg):
    """(seq_no, num_markers) of an APP2 segment's body if it is an ICC_PROFILE chunk, else None."""
    if len(seg) >= 14 and bytes(seg[:12]) == b"ICC_PROFILE\x00":
        return seg[12], seg[13]
    return None


class Unsupported(Exception):
    """This JPEG is not one the GPU path decodes exactly (or it is malformed). The message says why."""


class NeedMore(Unsupported):
    """The header is cut off where the data given ends: the same call with more of the file may succeed."""


def _after_fill(mv, i, n):
    """The first offset p > i with mv[p] != 0xFF (or n): a run of fill bytes is skipped in steps of growing size,
    so a long run costs about as much as reading it, not one Python step per byte."""
    p = i + 1
    if p >= n or mv[p] != 0xFF:
        return p
    w = 64
    while p < n:
        e = min(n, p + w)
        nz = np.flatnonzero(np.frombuffer(mv[p:e], np.uint8) != 0xFF)
        if nz.size:
            return p + int(nz[0])
        p, w = e, w * 4
    return n


# libjpeg's standard Huffman tables (jstdhuff.c). libjpeg-turbo installs them when a JPEG has no DHT
# for table 0 or 1 (Motion-JPEG frames).
_STD = {
    (0, 0): ([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bytes(range(12))),
    (0, 1): ([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], bytes(range(12))),
    (1, 0): ([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d], bytes([
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa])),
    (1, 1): ([0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77], bytes([
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa])),
}


def check_huffman(counts, vals, is_dc):
    """libjpeg-turbo's table checks (jdmarker.c get_dht, jdhuff.c jpeg_make_d_derived_tbl)."""
    n = sum(counts)
    if n < 1 or n > 256 or len(vals) != n:
        raise Unsupported("bad Huffman table")
    code = 0
    for l in range(1, 17):
        code += counts[l - 1]
        if code >= (1 << l):                # jdhuff.c: also rejects an all-ones code
            raise Unsupported("bad Huffman table")
        code <<= 1
    if is_dc and max(vals) > 15:
        raise Unsupported("bad Huffman table")


@dataclass
class Header:
    width: int
    height: int
    ncomp: int                      # 1 (grayscale) or 3 (YCbCr)
    samp: list                      # per component (h, v) as used for decoding (1, 1 for a 1-component scan)
    quant: list                     # per component: int32[64] in zig-zag order
    dc_tabs: list                   # per component: (counts[16], values bytes)
    ac_tabs: list
    td: list                        # per component DC table id (for de-duplication)
    ta: list
    dri: int                        # restart interval in MCUs (0: none)
    scan_start: int                 # offset of the first entropy-coded byte
    progressive: bool = False
    info: dict = field(default_factory=dict)

    @property
    def hmax(self):
        return max(s[0] for s in self.samp)

    @property
    def vmax(self):
        return max(s[1] for s in self.samp)

    @property
    def subsampling(self):
        """'4:4:4', '4:2:2', '4:2:0', '4:4:0', 'gray', or the raw factors."""
        if self.ncomp == 1:
            return "gray"
        names = {(1, 1): "4:4:4", (2, 1): "4:2:2", (2, 2): "4:2:0", (1, 2): "4:4:0", (4, 1): "4:1:1"}
        (yh, yv), c1, c2 = self.samp
        if c1 == c2 and yh % c1[0] == 0 and yv % c1[1] == 0 and (yh // c1[0], yv // c1[1]) in names:
            return names[(yh // c1[0], yv // c1[1])]
        return "/".join(f"{h}x{v}" for h, v in self.samp)


def _u16(mv, i):
    return (mv[i] << 8) | mv[i + 1]


def parse(data) -> Header:
    """Parse the markers up to the first SOS. Raises Unsupported for anything the GPU path does not decode
    exactly, including malformed headers."""
    mv = memoryview(data).cast("B")
    n = len(mv)
    if n < 4 or mv[0] != 0xFF or mv[1] != 0xD8:
        raise Unsupported("not a JPEG")
    i = 2
    qt, dht = {}, {}
    sof = None
    dri = 0
    jfif = False
    adobe = None
    icc = []
    segments = 0
    while True:
        if i > MAX_HEADER:
            raise Unsupported("header longer than 16 MB")
        segments += 1
        if segments > MAX_SEGMENTS:
            raise Unsupported(f"more than {MAX_SEGMENTS} marker segments before the scan")
        if i + 1 >= n:
            raise NeedMore("no SOS")
        if mv[i] != 0xFF:
            raise Unsupported("garbage between markers")
        i = _after_fill(mv, i, min(n, MAX_HEADER + 2)) - 1       # fill bytes (a run is bounded by the header's
        if i > MAX_HEADER:                                       # length too: more data cannot cure it)
            raise Unsupported("header longer than 16 MB")
        if i + 1 >= n:
            raise NeedMore("no SOS")
        m = mv[i + 1]
        i += 2
        if m == 0x00 or m == 0x01 or 0xD0 <= m <= 0xD7:
            raise Unsupported(f"unexpected marker {m:#04x}")
        if m == 0xD8 or m == 0xD9:
            raise Unsupported("SOI/EOI before SOS")
        if i + 2 > n:
            raise NeedMore("truncated header")
        L = _u16(mv, i)
        if L < 2:
            raise Unsupported("bad marker length")
        if i + L > n:
            raise NeedMore("bad marker length")
        seg = mv[i + 2:i + L]
        sl = L - 2
        if m == 0xDB:                                    # DQT
            j = 0
            while j < sl:
                pq, tq = seg[j] >> 4, seg[j] & 15
                if tq > 3 or pq > 1:
                    raise Unsupported("bad DQT")
                w = 128 if pq else 64
                if j + 1 + w > sl:
                    raise Unsupported("bad DQT length")
                raw = bytes(seg[j + 1:j + 1 + w])
                qt[tq] = (np.frombuffer(raw, ">u2") if pq else np.frombuffer(raw, np.uint8)).astype(np.int32)
                j += 1 + w
        elif m == 0xC4:                                  # DHT
            j = 0
            while j < sl:
                if j + 17 > sl:
                    raise Unsupported("bad DHT length")
                tc, th = seg[j] >> 4, seg[j] & 15
                if tc > 1 or th > 3:
                    raise Unsupported("bad DHT index")
                counts = list(seg[j + 1:j + 17])
                nv = sum(counts)
                if nv > 256 or j + 17 + nv > sl:
                    raise Unsupported("bad Huffman table")
                vals = bytes(seg[j + 17:j + 17 + nv])
                check_huffman(counts, vals, tc == 0)
                dht[(tc, th)] = (counts, vals)
                j += 17 + nv
        elif m in (0xC0, 0xC1):                          # baseline / extended sequential, Huffman
            if sof is not None:
                raise Unsupported("duplicate SOF")
            if sl < 6:
                raise Unsupported("bad SOF")
            prec, H, W, nc = seg[0], _u16(seg, 1), _u16(seg, 3), seg[5]
            if sl != 6 + 3 * nc:
                raise Unsupported("bad SOF length")
            if prec != 8:
                raise Unsupported(f"{prec}-bit samples")
            if H == 0 or W == 0:
                raise Unsupported("empty image or DNL")
            if H > MAX_DIMENSION or W > MAX_DIMENSION:
                raise Unsupported("image too big")
            comps = []
            for k in range(nc):
                cid, hv, tq = seg[6 + 3 * k], seg[7 + 3 * k], seg[8 + 3 * k]
                h, v = hv >> 4, hv & 15
                if not (1 <= h <= 4 and 1 <= v <= 4) or tq > 3:
                    raise Unsupported("bad sampling factors")
                comps.append((cid, h, v, tq))
            if len({c[0] for c in comps}) != nc:
                raise Unsupported("duplicate component id")
            sof = dict(H=H, W=W, comps=comps, marker=m)
        elif m == 0xC2:
            raise Unsupported("progressive JPEG")
        elif m in (0xC3, 0xC5, 0xC6, 0xC7):
            raise Unsupported("lossless or hierarchical JPEG")
        elif 0xC9 <= m <= 0xCF and m != 0xCC or m == 0xC8:
            raise Unsupported("arithmetic coding" if m != 0xC8 else "JPG marker")
        elif m == 0xCC:
            raise Unsupported("arithmetic coding")
        elif m == 0xDD:                                  # DRI
            if sl != 2:
                raise Unsupported("bad DRI")
            dri = _u16(seg, 0)
        elif m == 0xE0:
            if sl >= 14 and bytes(seg[:5]) == b"JFIF\x00":
                jfif = True
                if seg[5] != 1:                          # libjpeg-turbo warns (JWRN_JFIF_MAJOR, any major but 1)
                    raise Unsupported(f"JFIF version {seg[5]}")      # and its TurboJPEG API refuses the file
        elif m == 0xEE:
            if sl >= 12 and bytes(seg[:5]) == b"Adobe":
                adobe = seg[11]
        elif m == 0xE2:
            c = _icc_chunk(seg)
            if c is not None:
                icc.append(c)
        elif 0xE1 <= m <= 0xEF or m == 0xFE:             # other APPn, COM
            pass
        elif m == 0xDA:                                  # SOS
            if sof is None:
                raise Unsupported("SOS before SOF")
            if sl < 1:
                raise Unsupported("bad SOS")
            ns = seg[0]
            if sl != 2 * ns + 4 or not 1 <= ns <= 4:
                raise Unsupported("bad SOS length")
            scomps = [(seg[1 + 2 * k], seg[2 + 2 * k] >> 4, seg[2 + 2 * k] & 15) for k in range(ns)]
            ss, se, ahal = seg[1 + 2 * ns], seg[2 + 2 * ns], seg[3 + 2 * ns]
            if icc_bogus(icc):                           # libjpeg-turbo warns and its TurboJPEG API refuses the file
                raise Unsupported("a bad ICC profile segment (libjpeg-turbo: bad ICC marker)")
            return _finish(sof, scomps, (ss, se, ahal), qt, dht, dri, jfif, adobe, i + L)
        else:
            raise Unsupported(f"marker {m:#04x}")
        i += L


def _finish(sof, scomps, spec, qt, dht, dri, jfif, adobe, scan_start):
    comps = sof["comps"]
    nc = len(comps)
    ids = [c[0] for c in comps]
    if nc == 4:
        raise Unsupported("4 components (CMYK or YCCK): decoded by the CPU decoder with its own colour conversion, "
                          "outside the identical-pixels claim (libjpeg-turbo has no conversion to RGB for them)")
    if nc not in (1, 3):
        raise Unsupported(f"{nc} components")
    if spec != (0, 63, 0):
        raise Unsupported("not a sequential scan")
    if [s[0] for s in scomps] != ids:
        raise Unsupported("scan does not hold every component in frame order")
    if nc == 3:
        # jdapimin.c default_decompress_parms: JFIF > Adobe transform > component ids
        if jfif:
            rgb = False
        elif adobe is not None:
            if adobe not in (0, 1):                      # libjpeg-turbo warns (JWRN_ADOBE_XFORM): declined here
                raise Unsupported(f"Adobe transform {adobe}")
            rgb = adobe == 0
        else:
            rgb = ids == [82, 71, 66]
        if rgb:
            raise Unsupported("RGB-coded JPEG")
        samp = [(c[1], c[2]) for c in comps]
        hmax, vmax = max(s[0] for s in samp), max(s[1] for s in samp)
        for h, v in samp:
            if hmax % h or vmax % v:                   # jdsample.c: fractional upsampling is not implemented
                raise Unsupported(f"sampling {samp}")
        if sum(h * v for h, v in samp) > MAX_BLOCKS_IN_MCU:
            raise Unsupported("too many blocks per MCU")
    else:
        samp = [(1, 1)]                     # a single-component scan is non-interleaved: one block per MCU
    q, dct, act = [], [], []
    for c in comps:
        if c[3] not in qt:
            raise Unsupported("missing quantisation table")
        q.append(qt[c[3]])
    for s in scomps:
        tabs = []
        for tc, th in ((0, s[1]), (1, s[2])):
            if th > 3:
                raise Unsupported("bad Huffman table id")
            t = dht.get((tc, th)) or _STD.get((tc, th))
            if t is None:
                raise Unsupported("missing Huffman table")
            tabs.append(t)
        dct.append(tabs[0])
        act.append(tabs[1])
    return Header(width=sof["W"], height=sof["H"], ncomp=nc, samp=samp, quant=q, dc_tabs=dct, ac_tabs=act,
                  td=[s[1] for s in scomps], ta=[s[2] for s in scomps], dri=dri, scan_start=scan_start)


WARN_EOF = "Premature end of JPEG file (no end marker)"     # libjpeg-turbo's JWRN_JPEG_EOF
WARN_EXTRA = "Corrupt JPEG data: extraneous bytes before a marker"     # JWRN_EXTRANEOUS_DATA


def header_warning(data):
    """The libjpeg-turbo warnings a file's header gives, or None: a JFIF APP0 with a major version other than 1
    (JWRN_JFIF_MAJOR; every JFIF APP0 is examined), an Adobe transform libjpeg-turbo does not know for the number of
    components (JWRN_ADOBE_XFORM: three components without JFIF, 0 or 1; four, 0 or 2), and ICC profile segments
    jpeg_read_icc_profile() warns about (JWRN_BOGUS_ICC). Reads the marker segments before the scan only (at most
    MAX_SEGMENTS)."""
    mv = memoryview(data).cast("B")
    n = len(mv)
    if n < 4 or mv[0] != 0xFF or mv[1] != 0xD8:
        return None
    i, majors, jfif, adobe, nc, icc = 2, [], False, None, None, []
    for _ in range(MAX_SEGMENTS):
        if i + 4 > n or mv[i] != 0xFF:
            break
        i = _after_fill(mv, i, n) - 1
        if i + 4 > n:
            break
        m, L = mv[i + 1], _u16(mv, i + 2)
        if m in (0xDA, 0xD9) or L < 2:
            break
        seg = mv[i + 4:min(n, i + 2 + L)]
        if m == 0xE0 and len(seg) >= 14 and bytes(seg[:5]) == b"JFIF\x00":
            jfif = True
            if seg[5] != 1:
                majors.append(f"{seg[5]}.{seg[6]:02d}")
        elif m == 0xE2 and _icc_chunk(seg) is not None:
            icc.append(_icc_chunk(seg))
        elif m == 0xEE and len(seg) >= 12 and bytes(seg[:5]) == b"Adobe":
            adobe = seg[11]
        elif 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC) and len(seg) >= 6:
            nc = seg[5]
        i += 2 + L
    w = [f"Warning: unknown JFIF revision number {v}" for v in majors]
    if adobe is not None and ((nc == 3 and not jfif and adobe not in (0, 1)) or (nc == 4 and adobe not in (0, 2))):
        w.append(f"Unknown Adobe color transform code {adobe}")
    if icc_bogus(icc):
        w.append(ICC_BAD)
    return "; ".join(w) or None


def after_scan(data, end):
    """The markers after the entropy-coded data, walked as check_trailer() walks them (fill bytes, APPn / COM
    segments): (True if the end marker follows, False if the data just ends: libjpeg-turbo then warns, WARN_EOF;
    the warnings libjpeg-turbo gives for the segments on the way: a JFIF APP0 with a major version other than 1,
    which jpeg_finish_decompress() reads and warns about as in the header)."""
    mv = memoryview(data).cast("B")
    n = len(mv)
    i = end
    segments = 0
    w = []
    while i + 1 < n and mv[i] == 0xFF:
        m = mv[i + 1]
        if m == 0xFF:
            i = _after_fill(mv, i, n) - 1
            continue
        if m == 0xD9:
            return True, w
        segments += 1
        if segments > MAX_SEGMENTS:                  # (check_trailer() declines such a file: no need to walk on)
            return False, w
        if (0xE0 <= m <= 0xEF or m == 0xFE) and i + 4 <= n:
            L = _u16(mv, i + 2)
            if m == 0xE0 and L >= 16 and i + 2 + L <= n and bytes(mv[i + 4:i + 9]) == b"JFIF\x00" and mv[i + 9] != 1:
                w.append(f"Warning: unknown JFIF revision number {mv[i + 9]}.{mv[i + 10]:02d}")
            i += 2 + L
            continue
        return False, w
    return False, w


def has_eoi(data, end):
    """True if the end marker follows the entropy-coded data (after_scan())."""
    return after_scan(data, end)[0]


def check_trailer(data, end):
    """After the entropy-coded data, which ends at offset `end` (a marker's 0xFF, or len(data)): accept only fill
    bytes, complete APPn/COM segments (libjpeg-turbo skips them unread), then EOI or the end of the data. Anything
    else (tables, a second scan, a truncated segment) is declined, because libjpeg-turbo parses it and may fail.
    Returns None if fine, else a reason."""
    mv = memoryview(data).cast("B")
    n = len(mv)
    i = end
    segments = 0
    while i < n:
        if mv[i] != 0xFF or i + 1 >= n:
            return "data after the scan"
        m = mv[i + 1]
        if m == 0xFF:
            i = _after_fill(mv, i, n) - 1
            continue
        segments += 1
        if segments > MAX_SEGMENTS:
            return f"more than {MAX_SEGMENTS} marker segments after the scan"
        if m == 0xD9:
            return None
        if 0xE0 <= m <= 0xEF or m == 0xFE:
            if i + 4 > n:
                return "truncated marker after the scan"
            L = _u16(mv, i + 2)
            if L < 2 or i + 2 + L > n:
                return "bad marker after the scan"
            i += 2 + L
            continue
        return f"marker {m:#04x} after the scan"
    return None
