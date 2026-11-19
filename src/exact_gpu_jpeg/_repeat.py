"""Decoding the same file again: a cache of coefficient handles (opt-in; nothing here runs unless a RepeatCache is
made and passed to decode() or to a Decoder).

A copy is found by what the file holds, never by its name alone: its key is the identity of the JPEG's bytes, a
SHA-256 over them (check_value). So a copy is only ever used for a file with exactly the bytes it was decoded from,
also when someone else made the file.
"""
import hashlib
import os
import struct
import threading
import zlib
from collections import OrderedDict

CHUNK = 65536                  # bytes per leaf of the identity (check_value); also the light check's head
CHECKS = ("full", "light")
# Images below this many pixels are not kept. 0: every image is. Measured on a GTX 1070 (bench/repeat_sizes.py): a
# decode from a copy was faster than a normal GPU decode at every size from 0.1 MP up, with the full check too.
MIN_PIXELS = 0
_pool = None
_pool_lock = threading.Lock()


def _threads():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from concurrent.futures import ThreadPoolExecutor
                _pool = ThreadPoolExecutor(min(8, os.cpu_count() or 1), thread_name_prefix="egj-check")
    return _pool


ID_BYTES = 32


def check_value(data):
    """The 32-byte identity of a JPEG's bytes (the cache's key): the SHA-256 of each 65,536-byte chunk,
    the last one simply shorter (none for empty data); then the SHA-256 of the length as a little-endian uint64
    followed by those digests in order. Two different byte strings with the same identity are a SHA-256
    collision, so it holds for files someone else made, not only against damage. The chunks are hashed on
    several threads (hashlib releases the GIL)."""
    mv = memoryview(data).cast("B")
    n = len(mv)
    if n <= 4 * CHUNK:
        leaves = [hashlib.sha256(mv[i:i + CHUNK]).digest() for i in range(0, n, CHUNK)]
    else:
        step = 16 * CHUNK
        parts = _threads().map(lambda a: [hashlib.sha256(mv[i:i + CHUNK]).digest()
                                          for i in range(a, min(a + step, n), CHUNK)], range(0, n, step))
        leaves = [d for p in parts for d in p]
    return hashlib.sha256(struct.pack("<Q", n) + b"".join(leaves)).digest()


def _identity(path):
    """What the light check compares: the file's size and modification time, and the CRC-32 of its first 64 KiB."""
    with open(path, "rb", buffering=0) as f:
        st = os.fstat(f.fileno())
        head = f.read(CHUNK)
    return st.st_size, st.st_mtime_ns, zlib.crc32(head)


def _read(path):
    """(the file's bytes, its identity as _identity() gives it, taken from the same open file, or None when the
    file changed while it was read)."""
    with open(path, "rb", buffering=0) as f:
        st = os.fstat(f.fileno())
        data = f.read()
        st2 = os.fstat(f.fileno())
    same = (st.st_size, st.st_mtime_ns) == (st2.st_size, st2.st_mtime_ns) == (len(data), st2.st_mtime_ns)
    return data, ((st.st_size, st.st_mtime_ns, zlib.crc32(data[:CHUNK])) if same else None)


class RepeatCache:
    """Fast copies of decoded JPEGs, so that decoding the same file again skips the entropy decode.

    cache = RepeatCache(memory_bytes=1 << 30); then egj.decode(src, cache=cache) or egj.Decoder(cache=cache). The
    first decode of a file is a normal decode whose coefficients are kept on the GPU; a later decode of a file with
    the same bytes is made from that copy: the same pixels. The least recently used copies are dropped when the
    copies together would hold more than memory_bytes of GPU memory. One image per call: lists (batches), queued
    decodes and images below min_pixels do not use the cache.

    How a copy is known to be current (the `check` argument of decode()):
      "full" (default): the file is read and its bytes' check value must be the copy's. A changed file is always
        decoded afresh. The file read and the check are paid on every call.
      "light": for a path, the file's size, modification time and first 64 KiB are compared with what they were
        when the file was last read in full; if they are the same the copy is used without reading the file. A
        file changed past its first 64 KiB with its size and time kept is NOT noticed. For pixels that only go to
        a screen. Bytes input is always checked in full.
    No file is locked or kept open.

    form: how a copy is kept, "packed" (default: 13 to 54 bytes for every 8x8 block, about 27 for a camera photo; unpacked on the
    GPU when it is used) or "raw" (the decoder's arrays as they are: 133 bytes a block, nothing to unpack).

    A RepeatCache may be shared by Decoders on several threads. stats: counts since it was made."""

    def __init__(self, memory_bytes=1 << 30, *, min_pixels=None, form="packed"):
        from ._decoder import KEEP_FORMS
        if form not in KEEP_FORMS:
            raise ValueError(f"form must be one of {KEEP_FORMS}, got {form!r}")
        self.form = form                           # how copies are kept: "packed" (default) or "raw"
        self.memory_bytes = int(memory_bytes)
        self.min_pixels = MIN_PIXELS if min_pixels is None else int(min_pixels)
        self._lock = threading.Lock()
        self._copies = OrderedDict()               # (device, check value) -> CoefHandle, least recently used first
        self._paths = {}                           # normalised path -> (identity, check value)
        self._held = 0                             # bytes of GPU memory the copies hold
        self._n = dict(used=0, made=0, evicted=0, changed=0, too_small=0, over_budget=0)

    @property
    def stats(self):
        """used: decodes made from a copy; made: copies kept; evicted: copies dropped for the budget; changed:
        files the light check found changed; too_small / over_budget: decodes whose copy was not kept; copies and
        bytes: what is held now."""
        with self._lock:
            return dict(self._n, copies=len(self._copies), bytes=self._held)

    def clear(self):
        """Drop every copy (the GPU memory goes back as soon as no decode is using it)."""
        with self._lock:
            self._copies.clear()
            self._paths.clear()
            self._held = 0

    def drop(self, src):
        """Drop the copies of one file (a path or the JPEG's bytes). True if there was one."""
        if isinstance(src, (str, os.PathLike)):
            path = os.path.normcase(os.path.abspath(os.fspath(src)))
            with self._lock:
                rec = self._paths.pop(path, None)
            if rec is not None:
                cv = rec[1]
            else:
                with open(path, "rb") as f:
                    cv = check_value(f.read())
        else:
            cv = check_value(src)
        found = False
        with self._lock:
            for key in [k for k in self._copies if k[1] == cv]:
                self._held -= self._copies.pop(key).nbytes
                found = True
        return found

    def prepare(self, srcs, decoder=None):
        """Make the copies of these files now (at import, or while idle), so that their next decode is made from
        a copy. decoder: the Decoder to use (default: this thread's, as decode() uses). Returns how many of the
        files have a copy afterwards."""
        if decoder is None:
            from . import _thread_decoder
            decoder = _thread_decoder(None, False)
        n = 0
        for src in srcs:
            img, kept = self._decode(decoder, src, "RGB", None, "full", "uint8", "HWC", None, None, want_image=False)
            n += kept
        return n

    def _take(self, dev, cv):
        with self._lock:
            h = self._copies.get((dev, cv))
            if h is not None:
                self._copies.move_to_end((dev, cv))
            return h

    def _keep(self, dev, cv, handle):
        n = handle.nbytes
        with self._lock:
            if handle.width * handle.height < self.min_pixels:
                self._n["too_small"] += 1
                return False
            if n > self.memory_bytes:
                self._n["over_budget"] += 1
                return False
            old = self._copies.pop((dev, cv), None)
            if old is not None:
                self._held -= old.nbytes
            self._copies[(dev, cv)] = handle
            self._held += n
            self._n["made"] += 1
            while self._held > self.memory_bytes:              # the least recently used first (never the new one)
                key = next(iter(self._copies))
                self._held -= self._copies.pop(key).nbytes
                self._n["evicted"] += 1
            return True

    def _read_path(self, path):
        """Read the file and record what it was (every full read does: a later light check compares with the last
        file read in full, not with the first)."""
        data, ident = _read(path)
        cv = check_value(data)
        with self._lock:
            if ident is not None:
                self._paths[path] = (ident, cv)
            else:                                              # (it changed while being read: no record)
                self._paths.pop(path, None)
        return data, cv

    def _decode(self, dec, src, colorspace, timings, check, dtype, layout, scale, bias, want_image=True):
        """decode() with this cache: (image or None, whether the file has a copy afterwards)."""
        if check not in CHECKS:
            raise ValueError(f"check must be one of {CHECKS}, got {check!r}")
        dev = dec.device.id
        data = cv = path = ident = None
        if isinstance(src, (str, os.PathLike)):
            path = os.path.normcase(os.path.abspath(os.fspath(src)))
            if check == "light":
                ident = _identity(path)
                with self._lock:
                    rec = self._paths.get(path)
                if rec is not None:
                    if rec[0] == ident:
                        cv = rec[1]
                    else:
                        with self._lock:
                            self._n["changed"] += 1
        elif isinstance(src, bytes):
            data = src                                         # (immutable: what is hashed is what is decoded)
        else:
            # a buffer the caller can change (bytearray, memoryview, NumPy array): one copy, hashed and decoded,
            # so the identity always belongs to the bytes decoded. (A buffer staged by upload_ahead() and changed
            # since would otherwise be decoded as staged and kept under the new bytes' identity.)
            data = bytes(memoryview(src).cast("B"))
        handle = self._take(dev, cv) if cv is not None else None
        if handle is None:                                     # the bytes themselves decide
            if data is None:
                data, cv = self._read_path(path)
            else:
                cv = check_value(data)
            handle = self._take(dev, cv)
        if handle is not None:
            if not want_image:
                return None, True
            img = dec.decode_from_handle(handle, colorspace, timings, dtype=dtype, layout=layout, scale=scale, bias=bias)
            if img is not None:
                with self._lock:
                    self._n["used"] += 1
                return img, True
            # (the GPU path is off or memory is short: decode() says so itself below)
            if data is None:                                   # a light hit: the file was not read; read it now
                data, cv = self._read_path(path)
        keep, dec.keep_coefs, dec._cache_busy = dec.keep_coefs, self.form, True
        try:
            img = dec.decode(data, colorspace, timings, dtype=dtype, layout=layout, scale=scale, bias=bias)
        finally:
            dec.keep_coefs, dec._cache_busy = keep, False
        new = dec.take_coefs()
        return img, (new is not None and self._keep(dev, cv, new))
