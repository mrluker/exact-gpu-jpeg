"""RepeatCache (copies in GPU memory): a file decoded before is decoded again from its copy to the same pixels; a
copy is only ever used for a file with the bytes it was made from; the full check always notices a changed file,
the light check notices what it is documented to notice; budgets, threads and the calls that do not use the cache."""
import io
import os
import threading

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _repeat  # noqa: E402
from conftest import reference, sized  # noqa: E402


def _jpeg(w, h, seed=0, q=90, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=q, **kw)
    return b.getvalue()


def _same_start_same_size():
    """Two JPEGs of the same length whose first bytes (far more than 64 KiB) are the same and whose pixels differ:
    the same noise in the upper rows, other noise below; the shorter file gets fill bytes before its EOI."""
    rng = np.random.default_rng(3)
    top = rng.integers(0, 256, (464, 800, 3), dtype=np.uint8)
    files = []
    for seed in (4, 5):
        low = np.random.default_rng(seed).integers(0, 256, (136, 800, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(np.concatenate([top, low])).save(buf, "JPEG", quality=92)
        files.append(buf.getvalue())
    n = max(len(f) for f in files)
    a, b = (f[:-2] + bytes([0xFF]) * (n - len(f)) + f[-2:] for f in files)
    assert len(a) == len(b) and a[:150_000] == b[:150_000] and a != b
    assert not np.array_equal(reference(a), reference(b))
    return a, b


def _cache(**kw):
    kw.setdefault("min_pixels", 0)
    return egj.RepeatCache(**kw)


def test_the_identity_of_a_files_bytes():
    """The identity's test vectors: an empty input, 200,000 bytes from a formula, its first chunk; and the same value
    for every bytes-like, also on the path that uses several threads."""
    i = np.arange(200_000, dtype=np.uint64)
    d = (((i * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(24)).astype(np.uint8).tobytes()
    assert d[:8].hex() == "009e3cda7817b553"
    assert _repeat.check_value(b"").hex() == "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
    assert _repeat.check_value(d).hex() == "aa3dc24402f0ee4cce6f6fe0ac80b0270df5f1ac3b3e9ff1e6a449770ab9ed55"
    assert _repeat.check_value(d[:65536]).hex() == "4c9bf2d8f654caec2a8c96764a96ec56ecdf33c05585f7869d93e8f7efe83e5b"
    big = d * 40
    import hashlib
    import struct
    import zlib
    leaves = b"".join(hashlib.sha256(big[k:k + 65536]).digest() for k in range(0, len(big), 65536))
    assert _repeat.check_value(big) == hashlib.sha256(struct.pack("<Q", len(big)) + leaves).digest()
    assert _repeat.check_value(bytearray(d)) == _repeat.check_value(memoryview(d)) == _repeat.check_value(np.frombuffer(d, np.uint8))


def test_the_second_decode_comes_from_the_copy_and_is_the_same(corpus):
    cache = _cache()
    dec = egj.Decoder(cache=cache)
    n = 0
    for name, data, expected in corpus[::5]:
        want = reference(data)
        first = dec.decode(data)
        if first is None:
            assert dec.decode(data) is None, name              # declined both times, and nothing is kept
            continue
        before = cache.stats["used"]
        again = dec.decode(data)
        assert cache.stats["used"] == before + 1 and dec.stats["read"] == "handle", name
        assert np.array_equal(first.get(), want) and np.array_equal(again.get(), want), name
        assert np.array_equal(dec.decode(data, "BGR").get(), reference(data, "BGR")), name
        assert dec.take_coefs() is None and dec.keep_coefs is None      # the decoder's own setting is as it was
        n += 1
    s = cache.stats
    assert n >= 50 and s["copies"] == s["made"] and s["bytes"] > 0
    assert s["made"] + (s["used"] - 2 * n) == n                # (two corpus files with the same bytes share a copy)


def test_the_full_check_notices_a_changed_file_whatever_its_size_and_time(tmp_path):
    a = _jpeg(640, 480, 1)
    p = tmp_path / "x.jpg"
    p.write_bytes(a)
    st = os.stat(p)
    cache = _cache()
    dec = egj.Decoder(cache=cache)
    assert np.array_equal(dec.decode(str(p)).get(), reference(a))
    assert np.array_equal(dec.decode(str(p)).get(), reference(a)) and cache.stats["used"] == 1
    b = _jpeg(640, 480, 2)
    p.write_bytes(b)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))            # the old modification time
    img = dec.decode(str(p))                                    # check="full" is the default
    assert np.array_equal(img.get(), reference(b)) and cache.stats["used"] == 1 and cache.stats["made"] == 2
    assert np.array_equal(dec.decode(a).get(), reference(a)) and cache.stats["used"] == 2      # the old copy: for the old bytes


def test_the_light_check_and_the_change_it_does_not_notice(tmp_path):
    a, b = _same_start_same_size()
    p = tmp_path / "x.jpg"
    p.write_bytes(a)
    st = os.stat(p)
    cache = _cache()
    dec = egj.Decoder(cache=cache)
    assert np.array_equal(dec.decode(str(p), check="light").get(), reference(a))
    assert np.array_equal(dec.decode(str(p), check="light").get(), reference(a)) and cache.stats["used"] == 1
    p.write_bytes(b)                                            # changed, same size, and the old time put back
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    stale = dec.decode(str(p), check="light")                   # the documented miss: the copy of the old bytes
    assert np.array_equal(stale.get(), reference(a)) and cache.stats["used"] == 2
    fresh = dec.decode(str(p), check="full")                    # the full check reads the file: the new pixels
    assert np.array_equal(fresh.get(), reference(b))
    assert np.array_equal(dec.decode(b).get(), reference(b))    # and no copy was ever filed under the wrong bytes
    assert np.array_equal(dec.decode(a).get(), reference(a))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))        # a new time: the light check sees it
    changed = cache.stats["changed"]
    assert np.array_equal(dec.decode(str(p), check="light").get(), reference(b)) and cache.stats["changed"] == changed + 1
    with pytest.raises(ValueError, match="check"):
        dec.decode(str(p), check="none")


def test_budget_least_recently_used_and_the_minimum_size():
    files = [_jpeg(640, 480, s) for s in range(6)]
    dec0 = egj.Decoder(keep_coefs="raw")
    dec0.decode(files[0])
    one = dec0.take_coefs().nbytes
    cache = _cache(memory_bytes=3 * one + 100, form="raw")      # (raw copies of one size: the sums are exact)
    dec = egj.Decoder(cache=cache)
    for f in files[:3]:
        dec.decode(f)
    assert cache.stats["copies"] == 3 and cache.stats["bytes"] == 3 * one and cache.stats["evicted"] == 0
    dec.decode(files[0])                                        # used: now the most recent
    dec.decode(files[3])                                        # one must go: the least recently used, files[1]
    s = cache.stats
    assert s["copies"] == 3 and s["evicted"] == 1 and s["bytes"] == 3 * one
    used = s["used"]
    for k in (0, 2, 3):
        dec.decode(files[k])
    assert cache.stats["used"] == used + 3
    dec.decode(files[1])                                        # was evicted: a normal decode again
    assert cache.stats["used"] == used + 3 and cache.stats["made"] == 5
    tiny = egj.RepeatCache(memory_bytes=one - 1, min_pixels=0, form="raw")
    egj.Decoder(cache=tiny).decode(files[0])
    assert tiny.stats["copies"] == 0 and tiny.stats["over_budget"] == 1
    small = egj.RepeatCache(min_pixels=1_000_000)               # a minimum size: smaller images are not kept
    d = egj.Decoder(cache=small)
    assert np.array_equal(d.decode(files[0]).get(), reference(files[0]))
    assert small.stats["copies"] == 0 and small.stats["too_small"] == 1 and small.min_pixels == 1_000_000
    assert egj.RepeatCache().min_pixels == _repeat.MIN_PIXELS == 0
    assert cache.drop(files[1]) and not cache.drop(files[1])        # (files[0] went when files[1] came back)
    cache.clear()
    assert cache.stats["copies"] == 0 and cache.stats["bytes"] == 0


def test_the_module_level_decode_and_what_does_not_use_the_cache(tmp_path):
    cache = _cache()
    big, small, prog = _jpeg(800, 600, 1), _jpeg(160, 120, 2), _jpeg(800, 600, 3, progressive=True)
    p = tmp_path / "big.jpg"
    p.write_bytes(big)
    info = {}
    assert np.array_equal(egj.decode(str(p), cache=cache, info=info).get(), reference(big)) and info["path"] == "gpu"
    assert np.array_equal(egj.decode(str(p), cache=cache, check="light").get(), reference(big))
    assert np.array_equal(egj.decode(big, "GRAY", cache=cache).get(), reference(big, "GRAY"))
    assert cache.stats["made"] == 1 and cache.stats["used"] == 2
    info = {}
    assert np.array_equal(egj.decode(small, cache=cache, info=info).get(), reference(small))       # the CPU route
    assert info["path"] == "cpu" and cache.stats["made"] == 1
    info = {}
    assert np.array_equal(egj.decode(prog, cache=cache, info=info).get(), reference(prog))         # declined: CPU fallback
    assert info["path"] == "cpu" and cache.stats["made"] == 1
    imgs = egj.decode([big, big], cache=None)                   # a list: no cache
    assert len(imgs) == 2
    dec = egj.Decoder(cache=cache)
    job = dec.decode_async(big)                                 # a queued decode: no cache
    assert np.array_equal(job.result().get(), reference(big))
    assert all(i is not None for i in dec.decode_batch([big, big]))
    assert cache.stats["made"] == 1 and cache.stats["used"] == 2
    assert cache.prepare([str(p), prog, _jpeg(800, 600, 9)], decoder=dec) == 2 and cache.stats["made"] == 2
    with pytest.raises(TypeError):                             # the library keeps no copies on disk: no such argument
        egj.RepeatCache(directory=str(tmp_path))


def test_one_cache_shared_by_decoders_on_several_threads():
    files = [_jpeg(480, 360, s) for s in range(8)]
    want = [reference(f) for f in files]
    cache = _cache(memory_bytes=1 << 26)
    errors = []

    def work(k):
        try:
            dec = egj.Decoder(cache=cache)
            for r in range(30):
                i = (k + r) % len(files)
                img = dec.decode(files[i], check="full")
                if img is None or not np.array_equal(img.get(), want[i]):
                    errors.append((k, r, i, dec.reason))
        except Exception as e:                                  # noqa: BLE001
            errors.append((k, repr(e)))

    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:5]
    s = cache.stats
    assert s["copies"] <= len(files) and s["used"] >= 4 * 30 - 4 * len(files) and s["used"] + s["made"] >= 100


def test_guard_bands_with_the_cache(corpus):
    cache = _cache()
    dec = egj.Decoder(canary=True, cache=cache)
    for name, data, expected in sized(corpus, 10):
        for _ in range(3):
            assert np.array_equal(dec.decode(data).get(), reference(data)), name
    assert cache.stats["used"] >= 20
