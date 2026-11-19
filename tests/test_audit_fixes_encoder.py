"""Faults the independent audit (2026-11-18) found on the encoder's host side. Each test fails on a54cd05 and
passes with the fix. (The kernels' bytes were found sound; these are about files and jobs.)"""
import io

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _encoder as E  # noqa: E402


def _image(w, h, seed=0, noise=False):
    r = np.random.default_rng(seed)
    if noise:
        return r.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a = r.integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(a).resize((w, h), Image.BICUBIC))


def _pillow(a, q=90, sub=2):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=sub)
    return b.getvalue()


class _Short:
    """A file whose writes take at most `most` bytes each, as POSIX allows of an unbuffered write."""
    def __init__(self, f, most):
        self.f, self.most, self.name = f, most, f.name

    def write(self, data):
        return self.f.write(bytes(memoryview(data)[:self.most]))

    def close(self):
        self.f.close()


def test_short_writes_still_write_the_whole_file(tmp_path, monkeypatch):
    a = _image(640, 480, 1, noise=True)
    want = _pillow(a, 95, 0)
    p = tmp_path / "out.jpg"
    monkeypatch.setattr(E, "open", lambda *x, **kw: _Short(open(*x, **kw), 1000), raising=False)
    assert egj.Encoder().encode(a, 95, "444", path=str(p)) == len(want)
    assert p.read_bytes() == want                      # was: a file cut off, and its full length returned


class _NoProgress(_Short):
    def write(self, data):
        return 0


def test_a_write_that_makes_no_progress_is_an_error_and_leaves_no_file(tmp_path, monkeypatch):
    p = tmp_path / "out.jpg"
    monkeypatch.setattr(E, "open", lambda *x, **kw: _NoProgress(open(*x, **kw), 0), raising=False)
    with pytest.raises(OSError):
        egj.Encoder().encode(_image(64, 48), path=str(p))
    assert not p.exists()


def test_a_failed_job_raises_again(tmp_path):
    enc = egj.Encoder()
    job = enc.encode_async(_image(64, 48), path=str(tmp_path / "no_such_dir" / "x.jpg"))
    with pytest.raises(OSError):
        job.result()
    with pytest.raises(OSError):                        # was: None, as if the file had been written
        job.result()


def test_the_next_encode_names_the_earlier_jobs_error(tmp_path):
    enc = egj.Encoder()
    enc.encode_async(_image(64, 48), path=str(tmp_path / "no_such_dir" / "x.jpg"))
    with pytest.raises(OSError) as e:
        enc.encode(_image(64, 48), path=str(tmp_path / "ok.jpg"))
    assert "previous queued encode" in "".join(getattr(e.value, "__notes__", []))


def test_an_encode_done_again_opens_the_file_once(tmp_path, monkeypatch):
    """An image over the first pass's size bound is encoded again; the file stays open in between (a second
    open could fail, and the first one already emptied the file)."""
    a = _image(320, 240, 2, noise=True)
    want = _pillow(a, 100, 0)
    p = tmp_path / "out.jpg"
    opens = []
    real_open = open
    monkeypatch.setattr(E, "open", lambda *x, **kw: opens.append(x[0]) or real_open(*x, **kw), raising=False)
    real = E._out_cap
    monkeypatch.setattr(E, "_out_cap", lambda cap, retry: real(cap, True) if retry else 64)     # force the retry
    assert egj.Encoder().encode(a, 100, "444", path=str(p)) == len(want)
    assert p.read_bytes() == want and len(opens) == 1


def test_an_image_on_another_gpu_than_the_encoders_is_refused():
    img = cp.asarray(_image(64, 48))
    enc = egj.Encoder()
    with pytest.raises(ValueError, match="device"):
        E._as_cupy_rgb(img, enc.device.id + 1)          # (one GPU here: the check itself)
    assert egj.encode(img) == _pillow(_image(64, 48), 75)


def _retry_fails(monkeypatch, where):
    """Force the second pass of an encode and make it fail: in _queue (where="queue") or in the second
    collect before it has started its download (where="collect")."""
    real = E._out_cap
    monkeypatch.setattr(E, "_out_cap", lambda cap, retry: real(cap, True) if retry else 64)     # force the retry
    if where == "queue":
        q = E.Encoder._queue

        def queue(self, *a, **k):
            if k.get("attempt", 0) == 1:
                raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)
            return q(self, *a, **k)
        monkeypatch.setattr(E.Encoder, "_queue", queue)
    else:
        pin, n = E.Encoder._pinned, []

        def pinned(self, size):
            n.append(size)
            if len(n) == 2:                                  # the second collect's first step
                raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)
            return pin(self, size)
        monkeypatch.setattr(E.Encoder, "_pinned", pinned)


@pytest.mark.parametrize("where", ["queue", "collect"])
def test_a_retry_that_fails_leaves_no_file(tmp_path, monkeypatch, where):
    """(second audit, part 4.1) The file opened by the first pass is closed and removed when the second pass
    fails before its own collect owns it: no empty file under the name (on Windows an open file cannot be
    removed, so a file left open would stay)."""
    a = _image(320, 240, 2, noise=True)
    p = tmp_path / "out.jpg"
    p.write_bytes(b"an older file")
    _retry_fails(monkeypatch, where)
    with pytest.raises(cp.cuda.memory.OutOfMemoryError):
        egj.Encoder().encode(a, 100, "444", path=str(p))
    assert not p.exists()


def test_a_failed_job_does_not_keep_the_encodes_gpu_memory(tmp_path):
    """(second audit, part 4.6) A job whose result() raised keeps its exception, not the encode's device
    buffers (they were held by the frames of the exception's traceback)."""
    import gc
    a = _image(1024, 768, 3)
    enc = egj.Encoder()
    enc.encode(a, 90, "444")                                 # tables and pinned buffers made once
    pool = cp.get_default_memory_pool()
    gc.collect()
    gc.disable()                                             # (only what is still referenced counts)
    try:
        before = pool.used_bytes()
        job = enc.encode_async(a, 90, "444", path=str(tmp_path))     # a folder: the open fails in result()
        with pytest.raises(OSError):
            job.result()
        with pytest.raises(OSError):
            job.result()                                     # the same exception again
        held = pool.used_bytes() - before
    finally:
        gc.enable()
    assert held < 1 << 20, held                              # (the encode's own buffers are about 10 MB)
