"""The one timing switch: off by default and silent; on, each module-level decode of one image and each encode prints
one line and fills info=, and the bytes are the same either way."""
import io

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import reference  # noqa: E402


def _jpeg(w, h, seed=0, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


@pytest.fixture
def switched_on():
    old = egj.timing(True)
    yield
    egj.timing(old)


def test_off_by_default_and_silent(capsys):
    assert egj.TIMING is False and egj.timing(False) is False
    info = {}
    img = egj.decode(_jpeg(640, 480), cpu_below=0, info=info)
    data = egj.encode(img, 90)
    assert info == dict(path="gpu", reason=None, warning=None, warnings_checked=True) and len(data) > 100
    out = capsys.readouterr()
    assert "timing" not in out.err and "timing" not in out.out


def test_on_it_prints_one_line_per_call_and_fills_info(switched_on, capsys, tmp_path):
    big, small, prog = _jpeg(640, 480), _jpeg(160, 120, 1), _jpeg(320, 240, 2, progressive=True)
    egj._timing_reported.clear()
    info = {}
    img = egj.decode(big, cpu_below=0, info=info)
    assert np.array_equal(img.get(), reference(big))
    assert info["path"] == "gpu" and info["total_ms"] > 0 and len(info["ms"]) >= 3
    assert all(isinstance(v, float) and v >= 0 for v in info["ms"].values())
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and err[0].startswith("exact_gpu_jpeg timing: device {") and "compute_capability" in err[0]
    assert err[1].startswith("exact_gpu_jpeg timing: decode gpu ") and " ms | " in err[1]
    info = {}
    assert np.array_equal(egj.decode(small, cpu_below=200_000, info=info).get(), reference(small))        # the CPU route
    assert info["path"] == "cpu" and info["ms"] == {} and info["total_ms"] > 0
    info = {}
    assert np.array_equal(egj.decode(prog, info=info).get(), reference(prog))                             # declined
    assert info["path"] == "cpu" and info["reason"] == "progressive JPEG"
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and err[0].startswith("exact_gpu_jpeg timing: decode cpu (") and "progressive JPEG" in err[1]
    b = io.BytesIO()
    Image.fromarray(img.get()).save(b, "JPEG", quality=90, subsampling=2)
    info = {}
    assert egj.encode(img, 90, "420", info=info) == b.getvalue()
    assert info["path"] == "gpu" and set(info["ms"]) == {"dct", "huffman", "stuff_download"} and info["total_ms"] > 0
    p = tmp_path / "x.jpg"
    assert egj.encode(img, 90, "420", path=str(p)) == len(b.getvalue()) and p.read_bytes() == b.getvalue()
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and all(line.startswith("exact_gpu_jpeg timing: encode gpu ") for line in err)
    imgs = egj.decode([big, big])                                                                         # a batch: not timed
    assert len(imgs) == 2 and capsys.readouterr().err == ""
    assert egj._thread_decoder(None).profile is False                                                     # left as it was


def test_the_same_bytes_with_timing_on_for_the_whole_corpus(switched_on, corpus, capsys):
    n = 0
    for name, data, expected in corpus[::4]:
        want = reference(data)
        if want is None:
            continue
        got = egj.decode(data, cpu_below=0)
        assert np.array_equal(got.get(), want), name
        n += 1
    assert n >= 50
    capsys.readouterr()
