"""No offset taken from a JPEG file reaches a kernel as the address of a 16-bit or wider load or store.

The kernels load the entropy-coded bytes 16 at a time, the unstuffed stream as 32-bit words, coefficients 16 bytes
at a time, and store pixels 8 and 16 bytes at a time. A GPU faults on such an access at an address that is not a
multiple of its size, and the CUDA context is unusable afterwards. The file decides where its entropy-coded data
starts (anything can precede the scan), how many bytes unstuffing keeps, and the image's width; the host and the
kernels must turn each of those into an aligned address:
  - the entropy-coded data is copied to the start of a device buffer, or read in place only where it starts on a
    64-byte boundary of the page-locked buffer (bytes input is placed so that it does);
  - a tile's unstuffed bytes are stored bytewise up to a 4-byte boundary, then as words;
  - plane offsets and pitches are multiples of 8, coefficient blocks start at multiples of 128 bytes;
  - an output row leaves as 16- or 4-byte words only where its address allows.
Here the same image is decoded with its scan data at every offset modulo 64 (a comment segment of every length
before it), by every route, with the buffer addresses checked at each kernel launch, and images of awkward sizes
are decoded in every layout."""
import io
import struct

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from exact_gpu_jpeg._header import parse  # noqa: E402
from conftest import reference  # noqa: E402


def _jpeg(w, h, seed=0, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 4, 1), max(w // 4, 1), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=85, **kw)
    return b.getvalue()


def _shifted(data, k):
    """The same JPEG with a comment segment of k bytes after SOI: everything after it moves by 4 + k bytes."""
    return data[:2] + bytes([0xFF, 0xFE]) + struct.pack(">H", 2 + k) + bytes(k) + data[2:]


def _ptr(a):
    return a.data.ptr if hasattr(a, "data") else int(a)


# kernel -> (argument index, the alignment its accesses need)
NEEDS = {"unstuff_count": [(0, 16)], "unstuff_scatter": [(0, 16), (6, 4)], "sync_pass": [(0, 4)],
         "decode_write": [(0, 4), (17, 16)], "idct_islow": [(0, 16), (6, 8)], "idct_ifast": [(0, 16), (6, 8)]}


@pytest.fixture
def checked(monkeypatch):
    """Every launch of the kernels above is checked: its wide-access buffers start at an aligned address."""
    seen = {}
    real = D._k

    def k(name):
        f = real(name)
        if name not in NEEDS:
            return f

        def launch(grid, block, args, **kw):
            for i, al in NEEDS[name]:
                assert _ptr(args[i]) % al == 0, (name, i, al, _ptr(args[i]) % al)
            seen[name] = seen.get(name, 0) + 1
            return f(grid, block, args, **kw)
        return launch
    monkeypatch.setattr(D, "_k", k)
    return seen


def test_scan_data_at_every_offset_modulo_64(checked, tmp_path):
    base = _jpeg(96, 64, 1, subsampling=2)
    want = reference(base)
    files = [_shifted(base, k) for k in range(64)]
    starts = {parse(f).scan_start % 64 for f in files}
    assert starts == set(range(64))                            # every residue, so every alignment of the scan data
    dec, stager = egj.Decoder(), egj.Decoder()
    mapped = 0
    for k, f in enumerate(files):
        assert np.array_equal(reference(f), want)
        for src in (f, bytearray(f), memoryview(f)):           # in memory
            img = dec.decode(src)
            assert img is not None and np.array_equal(img.get(), want), (k, type(src).__name__, dec.reason)
            mapped += dec.stats.get("upload") == "mapped"
        p = tmp_path / f"s{k}.jpg"
        p.write_bytes(f)
        img = dec.decode(str(p))                               # from a file
        assert img is not None and np.array_equal(img.get(), want), (k, "path", dec.reason)
        if dec.stats.get("upload") == "mapped":                # read in place: only from a 64-byte boundary
            assert parse(f).scan_start % 64 == 0
        assert stager.upload_ahead(f)                          # staged ahead
        img = stager.decode(f)
        assert img is not None and np.array_equal(img.get(), want), (k, "ahead", stager.reason)
    if D.MAPPED_BELOW and D._mapped_ok(dec.device.id):
        assert mapped > 0                                      # the in-place read was among the routes taken
    assert checked["unstuff_count"] >= 5 * 64 and checked["decode_write"] >= 5 * 64 and checked["idct_islow"] >= 5 * 64
    imgs = egj.decode(files)                                   # one batch: each image's data at a tile boundary
    assert len(imgs) == 64 and all(np.array_equal(i.get(), want) for i in imgs)
    cp.cuda.Device().synchronize()                             # (a fault would be reported here at the latest)
    assert np.array_equal(dec.decode(base).get(), want)


@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_awkward_sizes_in_every_layout(checked, subsampling):
    dec = egj.Decoder()
    for w, h in ((1, 1), (3, 5), (17, 9), (67, 35), (130, 3), (255, 129), (1025, 7)):
        f = _jpeg(w, h, w + h, subsampling=subsampling)
        L = D.layout(parse(f))
        pix = np.asarray(L["pix"])
        assert not (pix[38:44].view(np.int64) % 8).any() and not (pix[44:47] % 8).any()   # plane offsets, pitches
        for cs in ("RGB", "BGR", "GRAY"):                      # rows of 3 * w and of w bytes: every row alignment
            want = reference(f, cs)
            img = dec.decode(f, cs)
            assert img is not None and np.array_equal(img.get(), want), (w, h, cs, dec.reason)
        for dtype in ("float32", "float16"):
            for layout in ("HWC", "CHW"):
                img = dec.decode(f, dtype=dtype, layout=layout)
                x = reference(f).astype(np.float32)
                x = x.transpose(2, 0, 1) if layout == "CHW" else x
                assert img is not None and np.array_equal(img.get(), x.astype(dtype)), (w, h, dtype, layout)
    cp.cuda.Device().synchronize()


def test_restart_markers_put_the_unstuffed_words_at_every_byte_offset(checked):
    """Restart markers and stuffed 0xFF bytes are dropped, so a tile's kept bytes start at any byte offset of the
    unstuffed stream: the word stores must begin at the next 4-byte boundary."""
    dec = egj.Decoder()
    rng = np.random.default_rng(5)
    for n in range(12):
        a = rng.integers(0, 256, (96, 240, 3), dtype=np.uint8)                # noise: many 0xFF bytes to unstuff
        b = io.BytesIO()
        Image.fromarray(a).save(b, "JPEG", quality=95, subsampling=n % 3, restart_marker_blocks=1 + n)
        f = b.getvalue()
        assert parse(f).dri
        img = dec.decode(f)
        assert img is not None and np.array_equal(img.get(), reference(f)), (n, dec.reason)
    assert checked["unstuff_scatter"] >= 12
    cp.cuda.Device().synchronize()


@pytest.mark.parametrize("flat", [False, True])
def test_a_file_from_a_path_after_bytes_is_read_from_its_start(tmp_path, flat):
    """Bytes input is placed in the page-locked buffer so that its scan data starts on a 64-byte boundary; a file
    read from a path sits at the buffer's start. Once the decoder kept the first offset for the second read: a
    small file was then decoded from scan data taken 1 to 63 bytes late. With an end marker after the scan that
    was always declined; a file with nothing after its scan (which libjpeg-turbo accepts) could come back with
    other pixels. Here: files with and without the end marker, the same file from bytes first (strict=False: a file
    without its end marker is one libjpeg-turbo decodes with a warning, declined by default)."""
    dec = egj.Decoder(strict=False)
    rng = np.random.default_rng(9)
    n = 0
    for w, h in ((96, 64), (333, 517), (640, 480)):
        for sub in (0, 1, 2):
            if flat:
                a = np.empty((h, w, 3), np.uint8)
                a[:] = (200, 30, 90)
            else:
                g = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
                a = np.asarray(Image.fromarray(g).resize((w, h), Image.BICUBIC))
            b = io.BytesIO()
            Image.fromarray(a).save(b, "JPEG", quality=85, subsampling=sub)
            full = b.getvalue()
            assert full.endswith(bytes([0xFF, 0xD9]))
            for k, data in enumerate((full, full[:-2])):       # with the end marker, and without it
                want = reference(data)
                first = dec.decode(data)                       # from bytes
                assert first is not None and np.array_equal(first.get(), want), (w, h, sub, k, dec.reason)
                p = tmp_path / f"p{n}.jpg"
                p.write_bytes(data)
                n += 1
                img = dec.decode(str(p))                       # then from a path, on the same decoder
                assert img is not None and np.array_equal(img.get(), want), (w, h, sub, k, dec.reason)
                via = egj.decode(data, cpu_below=0, strict=False)  # the same through the module's own decoder
                info = {}
                img = egj.decode(str(p), cpu_below=0, info=info, strict=False)
                assert np.array_equal(via.get(), want) and np.array_equal(img.get(), want) and info["path"] == "gpu"
