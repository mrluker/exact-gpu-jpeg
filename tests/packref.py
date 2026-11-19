"""The packed coefficient form on the CPU, a second implementation of the layout written apart from the kernels:
the arrays of the formula test vectors, a packer, an unpacker. The reference the GPU kernels are compared with
(slow: plain Python loops). The layout is internal to the library: handles are opaque."""
import struct

import numpy as np

M = (1 << 64) - 1

# seed, blocks, wide, wide blocks, body bytes, SHA-256 of the arrays, SHA-256 of the body, identity of the body (32 bytes)
VECTORS = [
    (1, 64, False, 8, 1832, "41c119404c6401a4a512901b14839de86b507cd7f9a81ea90c35802f18a77f86",
     "3fc65fcfba9b23f77e70aa124ab8ee5058c24c5d9ecd7b78828da61968571b4d", "d05ba40691df5945b1a2eb3eaea321608ad4b8c08d1ae95a9e3c1f8a421109e7"),
    (2, 4096, False, 546, 125136, "45ad154bf136b45ab6bcc3b0c18e7c50bb74b6f804dbe71f27050d7dadf4c57c",
     "315b1c9cebb033d829b4a1571543f216f38bbb6e3251950cafae1e95fec4f943", "f1f348d559e854336de9b88690400a675f8f0c10f40e71a330a979148bc38015"),
    (3, 65536, False, 8135, 2017814, "1a6f44b7dff81b600294341c8a7207d9b88833a6ae7e36cc1e10c3c210299264",
     "f3bdd6c80f3e49b3094f270caefb0c6fb1a0f165feab9b94d3da7d2dd30f07b5", "54dfad96bae768d6c19cd2fb7b92e5d0f5a1e478ba814fe45e9db1295e220320"),
    (4, 4096, True, 521, 124686, "c1639c9406f3a92ef55696b6e7bcb3049fa61d9d0f243f0dd403c00ca53c9f17",
     "d0ff39ff9f9bcb04cfe281ac0d7ebb757ce7b90f4e9ea4201c333e0dd6678ab2", "ffbfbaf1c852ca939f2810168bd6817e76c79c2ba95ff6bc41dbda75ec3cab2a"),
]


def arrays(seed, B, wide):
    """(coef (B, 64) int16, nq (B,) uint8, dc (B,) int32) of a formula vector: xorshift64 from `seed`."""
    s = seed

    def nxt():
        nonlocal s
        s ^= (s << 13) & M
        s ^= s >> 7
        s ^= (s << 17) & M
        return s
    coef = np.zeros((B, 64), np.int16)
    nq = np.zeros(B, np.uint8)
    dc = np.zeros(B, np.int32)
    for b in range(B):
        r = nxt()
        dc[b] = (r % 65536) - 32768 if wide else (r % 2047) - 1023
        k = (r >> 16) % 8
        count = 0 if k == 0 else 63 if k == 1 else 1 + ((r >> 24) % 24)
        limit = (32767 if wide else 1023) if k == 2 else 127 if k == 3 else 7
        top = 0
        for j in range(count):
            r2 = nxt()
            pos = 1 + j if k == 1 else 1 + (r2 % 63)
            mag = 1 + ((r2 >> 8) % limit)
            coef[b, pos] = -mag if (r2 >> 40) & 1 else mag
            top = max(top, pos)
        nq[b] = 0 if count == 0 else top // 16 + 1
    return coef, nq, dc


def pack(coef, nq, dc):
    """(index words uint32, data bytes, wide blocks) for coef (B, 64), nq, dc: the packed layout."""
    coef = np.asarray(coef).reshape(-1, 64)
    idx = np.zeros(len(nq), np.uint32)
    data = bytearray()
    wides = 0
    for b in range(len(nq)):
        n = int(nq[b])
        vals = coef[b, :16 * n]
        nzv = vals[vals != 0]
        wide = bool(nzv.size and (nzv.min() < -128 or nzv.max() > 127))
        wides += wide
        off = len(data)
        data += struct.pack("<h", int(dc[b]))
        for q in range(n):
            m = 0
            for i in np.flatnonzero(vals[16 * q:16 * q + 16]):
                m |= 1 << int(i)
            data += struct.pack("<H", m)
        data += nzv.astype("<i2" if wide else "i1").tobytes()
        if len(data) % 2:
            data.append(0)
        idx[b] = off | (n << 28) | (int(wide) << 31)
    return idx, bytes(data), wides


def unpack(idx, data, B):
    """coef (B, 64), nq, dc from a body's index words and data (quads at or past nq[b] are zero)."""
    coef = np.zeros((B, 64), np.int16)
    nq = np.zeros(B, np.uint8)
    dc = np.zeros(B, np.int32)
    for b in range(B):
        h = int(idx[b])
        n, wide, off = (h >> 28) & 7, h >> 31, h & 0x0FFFFFFF
        dc[b] = struct.unpack_from("<h", data, off)[0]
        nq[b] = n
        masks = struct.unpack_from(f"<{n}H", data, off + 2)
        v = off + 2 + 2 * n
        for q in range(n):
            for i in range(16):
                if masks[q] >> i & 1:
                    coef[b, 16 * q + i] = struct.unpack_from("<h" if wide else "<b", data, v)[0]
                    v += 2 if wide else 1
    return coef, nq, dc


def zero_past_nq(coef, nq):
    """coef (B, 64) with the quads at or past nq[b] set to zero (the IDCT does not read them)."""
    coef = np.array(coef, np.int16).reshape(-1, 64)
    keep = np.arange(64)[None, :] < (np.asarray(nq, np.int64)[:, None] * 16)
    return np.where(keep, coef, 0).astype(np.int16)
