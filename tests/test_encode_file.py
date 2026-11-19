"""encode(path=...) writes the file itself (no temporary name, no rename). When the file cannot be opened the
caller gets Python's own error naming the file and nothing is changed; when the encode fails after the file was
opened, no empty or partial file is left under that name; the encoder works again afterwards."""
import io
import os
import stat
import sys

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _encoder as E  # noqa: E402

OLD = b"what was there before " * 50
windows = pytest.mark.skipif(sys.platform != "win32", reason="file sharing modes are a Windows matter")


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


def _hold(path, share):
    """Open `path` for reading as another program would, sharing it for: 0 nothing, 1 reading, 3 reading and
    writing. Returns a function that lets go of it."""
    import ctypes as C
    from ctypes import wintypes as W
    k = C.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = W.HANDLE
    k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
    k.CloseHandle.argtypes = [W.HANDLE]
    h = k.CreateFileW(str(path), 0x80000000, share, None, 3, 0, None)
    assert h not in (None, W.HANDLE(-1).value), C.get_last_error()
    return lambda: k.CloseHandle(h)


@windows
@pytest.mark.parametrize("share", [0, 1])
def test_a_file_held_by_another_program_gives_a_clear_error_and_is_left_as_it_was(tmp_path, share):
    a = _image(640, 480)
    want = _pillow(a)
    p = tmp_path / "out.jpg"
    p.write_bytes(OLD)
    enc = egj.Encoder()
    let_go = _hold(p, share)
    try:
        with pytest.raises(PermissionError) as e:
            enc.encode(a, 90, "420", path=str(p))
        assert e.value.filename == str(p)                      # the error names the file
        with pytest.raises(PermissionError):
            egj.encode(a, 90, "420", path=str(p))              # the module-level call the same
        job = enc.encode_async(cp.asarray(a), 90, "420", path=str(p))
        with pytest.raises(PermissionError):
            job.result()
        assert enc.encode(a, 90, "420") == want                # the encoder is fine: the same call, to bytes
    finally:
        let_go()
    assert p.read_bytes() == OLD                               # nothing was written, nothing was removed
    assert enc.encode(a, 90, "420", path=str(p)) == len(want) and p.read_bytes() == want     # once it is free
    assert egj.encode(a, 90, "420") == want


@windows
def test_a_reader_that_shares_the_file_does_not_stop_the_write(tmp_path):
    a = _image(640, 480, 1)
    want = _pillow(a)
    p = tmp_path / "out.jpg"
    p.write_bytes(OLD)
    let_go = _hold(p, 3)                                       # as a scanner or an indexer opens it
    try:
        assert egj.encode(a, 90, "420", path=str(p)) == len(want)
    finally:
        let_go()
    assert p.read_bytes() == want


def test_a_read_only_file_a_folder_and_a_missing_folder(tmp_path):
    a = _image(320, 240, 2)
    want = _pillow(a)
    p = tmp_path / "readonly.jpg"
    p.write_bytes(OLD)
    os.chmod(p, stat.S_IREAD)
    try:
        if not os.access(p, os.W_OK):                          # (root may write it anyway)
            with pytest.raises(PermissionError):
                egj.encode(a, 90, "420", path=str(p))
            assert p.read_bytes() == OLD
    finally:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    d = tmp_path / "a_folder.jpg"
    d.mkdir()
    with pytest.raises(OSError):                               # PermissionError on Windows, IsADirectoryError elsewhere
        egj.encode(a, 90, "420", path=str(d))
    assert d.is_dir()
    with pytest.raises(FileNotFoundError):
        egj.encode(a, 90, "420", path=str(tmp_path / "missing" / "x.jpg"))
    assert not (tmp_path / "missing").exists()
    assert egj.encode(a, 90, "420") == want


class _FullDisk:
    """A file whose n-th write fails (as a full disk does, or because the user pressed Ctrl-C)."""

    def __init__(self, f, fail_at, error):
        self.f, self.left, self.error = f, fail_at, error

    def write(self, data):
        self.left -= 1
        if self.left < 0:
            raise self.error
        return self.f.write(data)

    def close(self):
        self.f.close()


FULL = OSError(28, "No space left on device")


@pytest.mark.parametrize("fail_at, existed, error", [(0, False, FULL), (2, False, FULL), (2, True, FULL),
                                                     (1, False, KeyboardInterrupt())])
def test_a_write_that_fails_leaves_no_partial_file(tmp_path, monkeypatch, fail_at, existed, error):
    a = _image(1024, 768, 3, noise=True)                       # about 1 MB of data: several pieces
    want = _pillow(a, 95, 0)
    p = tmp_path / "out.jpg"
    if existed:
        p.write_bytes(OLD)
    enc = egj.Encoder()
    enc.PIECE = 1 << 16
    monkeypatch.setattr(E, "open", lambda *x, **kw: _FullDisk(open(*x, **kw), fail_at, error), raising=False)
    with pytest.raises(type(error)):
        enc.encode(a, 95, "444", path=str(p))
    assert not p.exists()                                      # neither an empty nor a partial file
    job = enc.encode_async(cp.asarray(a), 95, "444", path=str(p))
    with pytest.raises(type(error)):
        job.result()
    assert not p.exists()
    monkeypatch.undo()
    assert enc.encode(a, 95, "444", path=str(p)) == len(want) and p.read_bytes() == want
    assert enc.encode(a, 95, "444") == want
