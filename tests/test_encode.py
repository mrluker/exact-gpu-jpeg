"""GPU encoder: byte-identical to Pillow's JPEG writer (libjpeg-turbo), and strict about its inputs."""
import io
import random

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402

SS = {"444": 0, "422": 1, "420": 2}


def pil(a, q, ss, icc=None, exif=None):
    from PIL import Image
    kw = {}
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=SS[ss], **kw)
    return b.getvalue()


def content(rng, h, w, kind):
    y, x = np.mgrid[0:h, 0:w]
    if kind == "smooth":
        a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x * y) % 256], -1) + rng.normal(0, 15, (h, w, 3))
    elif kind == "noise":
        a = rng.integers(0, 256, (h, w, 3))
    elif kind == "edges":
        a = ((x // 3 + y // 5) % 2 * 255)[..., None].repeat(3, -1)
    elif kind == "black":
        a = np.zeros((h, w, 3))
    elif kind == "white":
        a = np.full((h, w, 3), 255)
    else:                                   # saturated primaries
        a = np.zeros((h, w, 3))
        a[..., 0] = (x % 3 == 0) * 255
        a[..., 1] = (x % 3 == 1) * 255
        a[..., 2] = (y % 2) * 255
    return np.clip(a, 0, 255).astype(np.uint8)


def test_every_quality():
    rng = np.random.default_rng(0)
    a = content(rng, 40, 56, "smooth")
    for q in range(1, 101):
        for ss in SS:
            assert egj.encode(a, q, ss) == pil(a, q, ss), (q, ss)


@pytest.mark.parametrize("ss", list(SS))
def test_sizes_and_content(ss):
    rng = np.random.default_rng(1)
    for h, w in [(1, 1), (1, 17), (17, 1), (2, 3), (7, 13), (8, 8), (9, 9), (15, 16), (16, 15), (17, 33), (31, 7),
                 (64, 48), (65, 47), (333, 501)]:
        for kind in ("smooth", "noise", "edges", "black", "white", "primaries"):
            a = content(rng, h, w, kind)
            assert egj.encode(cp.asarray(a), 90, ss) == pil(a, 90, ss), (h, w, kind)


def test_metadata():
    rng = np.random.default_rng(2)
    a = content(rng, 30, 40, "smooth")
    exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x00" + bytes(200)
    for icc in (b"x" * 100, bytes(rng.integers(0, 256, 140000, dtype=np.uint8))):   # 1 and 3 ICC chunks
        for ex in (None, exif):
            assert egj.encode(a, 87, "420", icc, ex) == pil(a, 87, "420", icc, ex)


def test_random_parameters():
    """Randomised sizes, qualities, subsamplings, content and metadata (a parameter fuzz)."""
    r = random.Random(5)
    rng = np.random.default_rng(5)
    for _ in range(150):
        h, w = r.randint(1, 130), r.randint(1, 130)
        a = content(rng, h, w, r.choice(["smooth", "noise", "edges", "primaries"]))
        q, ss = r.randint(1, 100), r.choice(list(SS))
        icc = bytes(rng.integers(0, 256, r.randint(1, 3000), dtype=np.uint8)) if r.random() < 0.3 else None
        ex = (b"Exif\x00\x00" + bytes(r.randint(0, 500))) if r.random() < 0.3 else None
        assert egj.encode(a, q, ss, icc, ex) == pil(a, q, ss, icc, ex), (h, w, q, ss)


def test_roundtrip_decodes_exactly():
    import simplejpeg
    rng = np.random.default_rng(3)
    a = content(rng, 120, 160, "smooth")
    for ss in SS:
        data = egj.encode(a, 90, ss)
        out = egj.decode(data, fallback=False)
        assert np.array_equal(out.get(), simplejpeg.decode_jpeg(data, "RGB", False, False))


def test_torch_input():
    torch = pytest.importorskip("torch")
    a = content(np.random.default_rng(4), 24, 40, "smooth")
    assert egj.encode(torch.from_numpy(a).cuda(), 95, "444") == pil(a, 95, "444")


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 3), np.float32),
    np.zeros((0, 4, 3), np.uint8), np.zeros((4, 0, 3), np.uint8)])
def test_bad_arrays(bad):
    with pytest.raises((ValueError, TypeError)):
        egj.encode(bad, 90, "444")


def test_bad_arguments():
    a = np.zeros((8, 8, 3), np.uint8)
    for kw in (dict(quality=0), dict(quality=101), dict(quality=90.5), dict(quality=True), dict(subsampling="411"),
               dict(subsampling=3), dict(exif="text"), dict(icc_profile=12), dict(exif=bytes(70000))):
        with pytest.raises((ValueError, TypeError)):
            egj.encode(a, **{"quality": 90, "subsampling": "444", **kw})
    with pytest.raises(ValueError):
        egj.encode(cp.zeros((8, 16, 3), cp.uint8)[:, ::2], 90, "444")          # not contiguous
    with pytest.raises(TypeError):
        egj.encode([[1, 2, 3]], 90, "444")


def test_larger_than_the_size_bound():
    """Noise at quality 100 needs more than the default buffer bound: the encoder redoes it with exact sizes."""
    a = np.random.default_rng(6).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    e = egj.Encoder()
    assert e.encode(a, 100, "444") == pil(a, 100, "444")
    assert e.stats["scan_bytes"] > max(1 << 20, (512 // 8) ** 2 * 3 * 40)


@pytest.mark.parametrize("piece", [1 << 20, 4096, 777])
def test_encode_to_path(tmp_path, piece):
    """path=...: the same bytes, written as they arrive from the GPU, whatever the piece size."""
    from exact_gpu_jpeg import _encoder as E
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (301, 517, 3), dtype=np.uint8)
    enc = E.Encoder()
    enc.PIECE = piece
    want = enc.encode(img, quality=93, subsampling="444", exif=b"Exif\x00\x00" + bytes(100))
    p = tmp_path / "o.jpg"
    n = enc.encode(img, quality=93, subsampling="444", exif=b"Exif\x00\x00" + bytes(100), path=str(p))
    assert n == len(want) and p.read_bytes() == want


def test_image_still_being_uploaded():
    """The caller's image may still be on its way to the GPU when encode() is called: cupy.asarray(numpy) copies
    asynchronously on the caller's stream, while the encoder works on a stream of its own, so encode() has to wait
    for the caller's stream first. Here that stream is kept busy, so the upload is still queued behind other work,
    and the memory it will fill holds other pixels until then."""
    rng = np.random.default_rng(5)
    a = content(rng, 64, 48, "noise")
    want = pil(a, 90, "420")
    for _ in range(4):
        stale = cp.full(a.shape, 200, cp.uint8)        # what the upload's memory block holds before the upload
        cp.cuda.get_current_stream().synchronize()
        del stale
        busy = cp.zeros(1 << 26, cp.float32)
        for _ in range(30):
            busy += 1                                  # work queued on the caller's stream
        x = cp.asarray(a)                              # the upload, queued behind it
        assert egj.encode(x, 90, "420") == want
        del x, busy
