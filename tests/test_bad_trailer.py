"""A complete, good scan followed by something libjpeg-turbo rejects (a bad table, a second scan, an unknown
marker): libjpeg-turbo gives no image, so the GPU path must decline, in every way of calling it, and the top
level must not return an image either. (The audit of 2026-11-18 found that the test oracle used to accept
Pillow's image for such files, so the fuzzers could not see a trailer accepted wrongly.)"""
import io

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import reference  # noqa: E402


def _good():
    a = np.random.default_rng(1).integers(0, 256, (31, 41, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((320, 240), Image.BICUBIC).save(b, "JPEG", quality=90)
    return b.getvalue()


def _seg(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _candidates():
    g = _good()
    body, eoi = g[:-2], g[-2:]
    sos = g[g.index(b"\xff\xda"):]
    sos = sos[:4 + int.from_bytes(sos[2:4], "big") - 2 + 2]
    return {
        "DQT with table index 11": body + _seg(0xDB, bytes([11]) + bytes(64)) + eoi,
        "DQT cut short": body + _seg(0xDB, bytes([0]) + bytes(10)) + eoi,
        "DHT with class 5": body + _seg(0xC4, bytes([0x50]) + bytes(16)) + eoi,
        "a second SOF": body + g[g.index(b"\xff\xc0"):g.index(b"\xff\xc0") + 19] + eoi,
        "a second SOS with no data": body + sos + eoi,
        "marker 0x01 (TEM)": body + b"\xff\x01" + eoi,
    }


def _rejected():
    out = {k: d for k, d in _candidates().items() if reference(d) is None}
    assert len(out) >= 3, sorted(_candidates())         # libjpeg-turbo really rejects most of these
    return out


def test_libjpeg_turbo_rejects_them_and_the_reference_says_so():
    from PIL import ImageFile
    was = ImageFile.LOAD_TRUNCATED_IMAGES
    _rejected()
    assert ImageFile.LOAD_TRUNCATED_IMAGES == was       # the oracle leaves Pillow's switch as it was


def test_the_gpu_path_declines_them_in_every_route(tmp_path):
    good = _good()
    want = reference(good)
    for name, bad in _rejected().items():
        dec = egj.Decoder()
        assert dec.decode(bad) is None, name
        assert "after the scan" in dec.reason or "marker" in dec.reason, (name, dec.reason)
        p = tmp_path / "bad.jpg"
        p.write_bytes(bad)
        for read in ("streamed", "whole"):
            assert egj.Decoder(read=read).decode(str(p)) is None, (name, read)
        job = dec.decode_async(bad)
        assert job is None or job.result() is None, name
        assert dec.upload_ahead(bad) in (True, False)
        assert dec.decode(bad) is None, name
        got = dec.decode_batch([good, bad, good])
        assert got[1] is None and dec.reasons[1], name
        assert np.array_equal(got[0].get(), want) and np.array_equal(got[2].get(), want), name
        with pytest.raises(Exception):                   # the CPU decoder rejects it too: no image at the top
            egj.decode(bad, cpu_below=0)
