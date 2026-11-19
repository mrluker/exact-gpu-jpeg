import threading

import numpy as np
import pytest

pytest.importorskip("cupy")

import exact_gpu_jpeg as egj
from conftest import reference


def _pick(corpus, prefix):
    return next(d for n, d, e in corpus if n.startswith(prefix))


def test_path_and_bytes(tmp_path, corpus):
    data = _pick(corpus, "sj_480x640_smooth_420")
    p = tmp_path / "a.jpg"
    p.write_bytes(data)
    a = egj.decode(str(p), fallback=False)
    for x in (data, bytearray(data), memoryview(data), np.frombuffer(data, np.uint8)):
        assert np.array_equal(a.get(), egj.decode(x, fallback=False).get())
    assert a.shape == (480, 640, 3) and a.dtype == np.uint8


def test_fallback(corpus):
    data = _pick(corpus, "pil_97x203_s2_prog")
    assert egj.decode(data, fallback=False) is None
    info = {}
    img = egj.decode(data, info=info)
    assert info["path"] == "cpu" and "progressive" in info["reason"]
    assert np.array_equal(img.get(), reference(data))
    egj.set_cpu_decoder(lambda d, cs: np.zeros((2, 2, 3), np.uint8))
    try:
        assert egj.decode(data).shape == (2, 2, 3)
    finally:
        egj.set_cpu_decoder(None)


def test_inspect(corpus):
    i = egj.inspect(_pick(corpus, "sj_480x640_smooth_422"))
    assert i["supported"] and i["subsampling"] == "4:2:2" and (i["width"], i["height"]) == (640, 480)
    i = egj.inspect(_pick(corpus, "pil_97x203_s2_prog"))
    assert not i["supported"] and "progressive" in i["reason"]
    assert not egj.inspect(b"not a jpeg")["supported"]


def test_bad_colorspace(corpus):
    with pytest.raises(ValueError):
        egj.Decoder().decode(_pick(corpus, "sj_8x8"), "CMYK")


def test_threads(corpus):
    files = [(d, reference(d)) for n, d, e in corpus if e and n.startswith(("sj_480", "sj_97", "pil_480"))]
    errors = []

    def run(k):
        for d, ref in files[k::4] * 3:
            out = egj.decode(d, fallback=False)
            if out is None or not np.array_equal(out.get(), ref):
                errors.append(k)
    ts = [threading.Thread(target=run, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def test_torch(corpus):
    torch = pytest.importorskip("torch")
    t = egj.decode(_pick(corpus, "sj_480x640_smooth_420"), out="torch")
    assert t.is_cuda and t.shape == (480, 640, 3) and t.dtype == torch.uint8


def test_max_pixels(corpus):
    d = egj.Decoder(max_pixels=1000)
    assert d.decode(_pick(corpus, "sj_480x640_smooth_420")) is None and "max_pixels" in d.reason


@pytest.mark.parametrize("app_bytes", [0, 300_000, 2_500_000])
@pytest.mark.parametrize("pieces", [(1 << 18, 1 << 21), (4096, 4096), (64, 1000)])
def test_streamed_read(tmp_path, corpus, app_bytes, pieces):
    """Files are read in pieces, each copied to the GPU as soon as it is read; the header is parsed from the
    first piece(s). Same pixels and same declines as decoding the bytes, whatever the piece sizes and however
    far into the file the scan starts (large APPn segments push it past the first pieces)."""
    from exact_gpu_jpeg import _decoder as D
    data = _pick(corpus, "sj_480x640_smooth_420")
    app = b""
    while len(app) < app_bytes:                     # APP2 segments of up to 65533 bytes
        k = min(65533, app_bytes - len(app))
        app += b"\xff\xe2" + (k + 2).to_bytes(2, "big") + bytes(k)
    data = data[:2] + app + data[2:]
    p = tmp_path / "a.jpg"
    p.write_bytes(data)
    dec = D.Decoder()
    dec.FIRST_PIECE, dec.PIECE = pieces
    assert D.STREAM_READ
    a = dec.decode(str(p))
    assert a is not None, dec.reason
    assert np.array_equal(a.get(), D.Decoder().decode(data).get())
    bad = data[:-5000]                              # truncated: declined the same way either way
    p.write_bytes(bad)
    assert dec.decode(str(p)) is None
    r = dec.reason
    d2 = D.Decoder()
    # the first reason (the Huffman stage's) is the same; later ones can depend on unwritten coefficients
    assert d2.decode(bad) is None and d2.reason.split(";")[0] == r.split(";")[0]
    p.write_bytes(_pick(corpus, "pil_97x203_s2_prog"))
    assert dec.decode(str(p)) is None and "progressive" in dec.reason
