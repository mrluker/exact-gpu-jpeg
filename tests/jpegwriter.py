"""A small baseline JPEG writer from quantised DCT coefficients, for tests: it can write coefficient values and
streams no real encoder would produce (extreme values, odd tables, restart intervals, broken markers).

write_jpeg(blocks, quant, samp, width, height, dri=0) -> bytes
  blocks: per component, int array (by, bx, 64) of quantised coefficients in natural (row-major) order, with
          by/bx the component's block grid padded to whole MCUs; quant: per component, 64 values (natural order).
"""
import numpy as np

NAT = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7,
                14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39,
                46, 53, 60, 61, 54, 47, 55, 62, 63])


class Bits:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, code, length):
        if length == 0:
            return
        self.acc = (self.acc << length) | (code & ((1 << length) - 1))
        self.n += length
        while self.n >= 8:
            self.n -= 8
            b = (self.acc >> self.n) & 0xFF
            self.out.append(b)
            if b == 0xFF:
                self.out.append(0)
        self.acc &= (1 << self.n) - 1

    def flush(self):                      # pad with 1-bits to a byte boundary
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)


def _size(v):
    v = abs(int(v))
    return v.bit_length()


def _vbits(v, s):
    return v if v >= 0 else v + (1 << s) - 1


def table_for(symbols):
    """A valid canonical table holding `symbols`: all codes the same length, never the all-ones code."""
    syms = sorted(set(symbols)) or [0]
    L = max(1, (len(syms)).bit_length())       # 2**L > len(syms): code len(syms)-1 is not all ones
    counts = [0] * 16
    counts[L - 1] = len(syms)
    codes = {s: (i, L) for i, s in enumerate(syms)}
    return counts, bytes(syms), codes


def _seg(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _symbols(blocks_in_order):
    """DC size symbols, AC run/size symbols and the per-block symbol lists."""
    out = []
    for comp, blk, dcdiff in blocks_in_order:
        z = blk[NAT]
        syms = [("dc", _size(dcdiff), dcdiff)]
        run = 0
        last = 63
        while last > 0 and z[last] == 0:
            last -= 1
        for k in range(1, last + 1):
            v = int(z[k])
            if v == 0:
                run += 1
                continue
            while run > 15:
                syms.append(("ac", 0xF0, None))
                run -= 16
            s = _size(v)
            syms.append(("ac", (run << 4) | s, v))
            run = 0
        if last < 63:
            syms.append(("ac", 0x00, None))
        out.append((comp, syms))
    return out


def write_jpeg(blocks, quant, samp, width, height, dri=0, ids=(1, 2, 3), app0=True, rst_numbers=None,
               extra_after_scan=b""):
    nc = len(blocks)
    hmax = max(s[0] for s in samp)
    vmax = max(s[1] for s in samp)
    if nc == 1:
        samp_scan = [(1, 1)]
        by, bx = blocks[0].shape[:2]
        mcux, mcuy = bx, by
    else:
        samp_scan = samp
        mcux = -(-width // (8 * hmax))
        mcuy = -(-height // (8 * vmax))
    # blocks in decode order with DC differences (prediction restarts at each restart interval)
    order = []
    pred = [0] * nc
    mcu = 0
    restarts = []
    for my in range(mcuy):
        for mx in range(mcux):
            if dri and mcu and mcu % dri == 0:
                restarts.append(len(order))
                pred = [0] * nc
            for ci in range(nc):
                h, v = samp_scan[ci]
                for dy in range(v):
                    for dx in range(h):
                        b = blocks[ci][my * v + dy, mx * h + dx]
                        dc = int(b[0])
                        order.append((ci, b, dc - pred[ci]))
                        pred[ci] = dc
            mcu += 1
    syms = _symbols(order)
    dcs = {0: [], 1: []}
    acs = {0: [], 1: []}
    for comp, sl in syms:
        t = 0 if comp == 0 else 1
        for kind, sym, _ in sl:
            (dcs if kind == "dc" else acs)[t].append(sym)
    tabs = {}
    for t in (0, 1):
        if t == 1 and nc == 1:
            continue
        tabs[("dc", t)] = table_for(dcs[t])
        tabs[("ac", t)] = table_for(acs[t])
    bits = Bits()
    data = bytearray()
    rst = 0
    rpos = set(restarts)
    for i, (comp, sl) in enumerate(syms):
        if i in rpos:
            bits.flush()
            data += bits.out
            bits = Bits()
            num = rst & 7 if rst_numbers is None else rst_numbers(rst)
            data += bytes([0xFF, 0xD0 + num])
            rst += 1
        t = 0 if comp == 0 else 1
        for kind, sym, v in sl:
            code, L = tabs[(kind, t)][2][sym]
            bits.put(code, L)
            s = sym & 15
            if s:
                bits.put(_vbits(v, s), s)
    bits.flush()
    data += bits.out
    out = bytearray(b"\xff\xd8")
    if app0:
        out += _seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    for ci in range(nc):
        q = np.asarray(quant[ci]).reshape(-1)[NAT]
        if q.max() > 255:
            out += _seg(0xDB, bytes([0x10 | ci]) + b"".join(int(x).to_bytes(2, "big") for x in q))
        else:
            out += _seg(0xDB, bytes([ci]) + bytes(int(x) for x in q))
    sof = bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([nc])
    for ci in range(nc):
        sof += bytes([ids[ci], (samp[ci][0] << 4) | samp[ci][1], ci])
    out += _seg(0xC0, sof)
    for (kind, t), (counts, vals, _) in tabs.items():
        out += _seg(0xC4, bytes([(0 if kind == "dc" else 0x10) | t]) + bytes(counts) + vals)
    if dri:
        out += _seg(0xDD, dri.to_bytes(2, "big"))
    sos = bytes([nc])
    for ci in range(nc):
        t = 0 if ci == 0 else 1
        sos += bytes([ids[ci], (t << 4) | t])
    sos += bytes([0, 63, 0])
    out += _seg(0xDA, sos)
    out += data + extra_after_scan + b"\xff\xd9"
    return bytes(out)


def grid(samp, width, height, ci):
    """(by, bx) block grid of component ci, padded to whole MCUs."""
    hmax = max(s[0] for s in samp)
    vmax = max(s[1] for s in samp)
    if len(samp) == 1:
        return -(-height // 8), -(-width // 8)
    mcux = -(-width // (8 * hmax))
    mcuy = -(-height // (8 * vmax))
    return mcuy * samp[ci][1], mcux * samp[ci][0]
