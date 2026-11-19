"""The packed coefficient form (see _repeat_kernels.cu): made on the GPU from the arrays a decode leaves there,
unpacked on the GPU into the same arrays, and never written to a file by this library.

A packed copy is two device buffers, each at the start of an allocation: the index words (uint32, one per block)
and the data bytes followed by TAIL zero bytes. Its body (the index words then the data bytes, without the tail)
is what the tests compare with tests/packref.py, a second implementation. The layout is internal: handles are
opaque, and it may change in any release. check_body() and check_layout() validate a body made elsewhere; only
the tests use them.
"""
import threading
from importlib import resources

import cupy as cp
import numpy as np

ERR_PK_RANGE, ERR_PK_LAYOUT = 1024, 2048
TAIL = 192                     # zero bytes after the data (a block's reads stay within 160 bytes of its offset)
MAX_DATA = 1 << 28             # bytes of data a copy may hold (offsets are 28 bits)
MAX_BLOCK = 138                # bytes of the largest block: 2 + 4 x 2 + 63 x 2, padded to even
_mods = {}
_lock = threading.Lock()
_i32, _u32 = np.int32, np.uint32


def _k(name):
    dev = cp.cuda.device.get_device_id()
    m = _mods.get(dev)
    if m is None:
        with _lock:
            m = _mods.get(dev)
            if m is None:
                src = resources.files(__package__).joinpath("_repeat_kernels.cu").read_text()
                m = cp.RawModule(code=src, options=("--std=c++14",))
                for n in ("unpack", "pk_check", "pk_size", "pk_write"):
                    m.get_function(n)                          # compile now, under the lock
                _mods[dev] = m
    return m.get_function(name)


def pack(alloc, coef, nq, dc, nblk, stream):
    """A packed copy of the three arrays of nblk blocks: (idx, data, dlen), device arrays allocated by `alloc`
    (the decoder's allocator), or None when the image is not packed (a DC outside 16 bits, or too much data).
    Queued on the current stream, with one wait for the size."""
    sz = alloc(nblk, cp.int32)
    meta = alloc(nblk, cp.uint32)
    off = alloc(nblk + 1, cp.uint32)
    err = alloc(1, cp.int32, zero=True)
    off[:1] = 0
    _k("pk_size")((-(-nblk // 256),), (256,), (coef, nq, dc, _i32(0), _i32(nblk), sz, meta, err))
    cp.cumsum(sz, dtype=cp.uint32, out=off[1:])
    total = int(off[nblk:].get(stream=stream)[0])
    if int(err.get(stream=stream)[0]) or total >= MAX_DATA:
        return None
    idx = alloc(nblk, cp.uint32)
    data = alloc(total + TAIL, cp.uint8)
    data[total:] = 0
    _k("pk_write")((-(-nblk * 4 // 256),), (256,), (coef, dc, off, meta, _i32(0), _i32(nblk), idx, data))
    stream.synchronize()                           # written before any other stream may read the copy
    return idx, data, total


def unpack(idx, data, dlen, nblk, coef, dc, nq, err):
    """Queue the unpack of a copy into the three arrays (err: the device word that takes ERR_PK_RANGE)."""
    _k("unpack")((-(-nblk * 4 // 256),), (256,), (idx, data, _i32(nblk), _u32(dlen), coef, dc, nq, err))


def check_layout(idx, data, dlen, nblk, err):
    """Queue the layout check of a copy (err: the device word that takes ERR_PK_RANGE / ERR_PK_LAYOUT)."""
    _k("pk_check")((-(-nblk // 256),), (256,), (idx, data, _i32(nblk), _u32(dlen), err))


def body(idx, data, dlen):
    """The copy's body as bytes: the index words (little-endian uint32), then the data bytes."""
    return cp.asnumpy(idx).astype("<u4").tobytes() + cp.asnumpy(data[:dlen]).tobytes()


def check_body(idx, data, nblk=None):
    """The validity rules of the packed form, on the host (NumPy): None if `idx` (uint32 per block) and `data`
    (bytes) are a body exactly as a packer writes it, else the first rule that fails. Every rule the kernels rely
    on, and the canonical form, in the order of the layout's rules."""
    idx = np.ascontiguousarray(idx, dtype=np.uint32)
    d = np.frombuffer(memoryview(data).cast("B"), np.uint8)
    dlen, B = d.size, idx.size
    if nblk is not None and B != nblk:
        return "the number of blocks is not the image's"
    if dlen & 1 or dlen >= MAX_DATA:
        return "the data length is odd or too large"
    if dlen > MAX_BLOCK * B:
        return "more data than the blocks can hold"
    off = (idx & np.uint32(0x0FFFFFFF)).astype(np.int64)
    n = ((idx >> np.uint32(28)) & np.uint32(7)).astype(np.int64)
    wide = (idx >> np.uint32(31)).astype(bool)
    if (off > dlen).any():
        return "a block offset past the data"
    if (off & 1).any():
        return "an odd block offset"
    if (n > 4).any():
        return "more than 4 quads in a block"
    if B == 0:
        return None if dlen == 0 else "data without blocks"
    if off[0] != 0:
        return "the first block does not start at 0"
    pad = np.concatenate([d, np.zeros(TAIL, np.uint8)]).astype(np.int64)

    def u16(at):
        return pad[at] | (pad[at + 1] << 8)
    cnt = np.zeros(B, np.int64)
    last = np.zeros(B, np.int64)
    for j in range(4):
        have = n > j
        m = np.where(have, u16(off + 2 + 2 * j), 0)
        bits = np.zeros(B, np.int64)
        for i in range(16):
            bits += (m >> i) & 1
        cnt += bits
        last = np.where(n == j + 1, m, last)
        if j == 0 and ((m & 1) != 0).any():
            return "position 0 (the DC's place) stored as a value"
    used = 2 + 2 * n + cnt * np.where(wide, 2, 1)
    size = (used + 1) & ~1
    nxt = np.concatenate([off[1:], [dlen]])
    if (off + size != nxt).any():
        return "a block does not end where the next one starts"
    if ((n > 0) & (last == 0)).any():
        return "the last quad of a block has no value"
    if ((n == 0) & wide).any():
        return "a block without values flagged wide"
    odd = (used & 1) == 1
    if (pad[(off + used)[odd]] != 0).any():
        return "a pad byte that is not 0"
    for b in np.flatnonzero(wide & (n > 0)):                   # a wide block holds a value outside -128..127
        v = int(off[b] + 2 + 2 * n[b])
        vals = d[v:v + 2 * int(cnt[b])].view("<i2")
        if not ((vals < -128) | (vals > 127)).any():
            return "a block flagged wide that holds no wide value"
    return None
