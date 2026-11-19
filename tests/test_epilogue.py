"""Float output written by the colour kernel (dtype / layout / scale / bias) equals the uint8 decode converted with
NumPy (img.astype(float32) * scale + bias, then float16 / CHW), bit for bit; uint8 output is unchanged."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import sized  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def numpy_epilogue(img, dtype, layout, scale, bias):
    nch = img.shape[2]
    s = np.broadcast_to(np.asarray(1.0 if scale is None else scale, np.float32), (nch,))
    b = np.broadcast_to(np.asarray(0.0 if bias is None else bias, np.float32), (nch,))
    v = img.astype(np.float32) * s + b
    if dtype == "float16":
        v = v.astype(np.float16)
    return np.ascontiguousarray(v.transpose(2, 0, 1)) if layout == "CHW" else v


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("layout", ["HWC", "CHW"])
@pytest.mark.parametrize("norm", [False, True])
def test_epilogue_matches_numpy(corpus, dtype, layout, norm):
    dec = egj.Decoder()
    scale, bias = (1 / (255 * STD), -MEAN / STD) if norm else (None, None)
    n = 0
    for name, data, expected in corpus[::9]:
        if not expected:
            continue
        ref = dec.decode(data).get()
        out = dec.decode(data, dtype=dtype, layout=layout, scale=scale, bias=bias)
        assert out is not None, (name, dec.reason)
        want = numpy_epilogue(ref, dtype, layout, scale, bias)
        assert out.dtype == want.dtype and out.shape == want.shape, name
        assert np.array_equal(out.get().view(np.uint8), want.view(np.uint8)), name     # bit for bit (NaN-free)
        n += 1
    assert n > 10


def test_gray_and_async(corpus):
    dec = egj.Decoder()
    name, data, _ = sized(corpus, 1)[0]
    g = dec.decode(data, "GRAY").get()
    f = dec.decode(data, "GRAY", dtype="float32", layout="CHW", scale=1 / 255).get()
    assert np.array_equal(f, numpy_epilogue(g, "float32", "CHW", 1 / 255, None)), name
    ref = dec.decode(data, dtype="float16").get()
    out = cp.empty(ref.shape, cp.float16)
    job = dec.decode_async(data, out=out, dtype="float16")
    assert job.result() is out and np.array_equal(out.get(), ref), name


def test_bad_options():
    dec = egj.Decoder()
    with pytest.raises(ValueError):
        dec.decode(b"\xff\xd8", layout="CHW")              # layout without a float dtype
    with pytest.raises(ValueError):
        dec.decode(b"\xff\xd8", dtype="int8")


def test_apply_epilogue_cpu(corpus):
    name, data, _ = sized(corpus, 1)[0]
    a = egj.decode_cpu(data)
    got = egj.apply_epilogue(a, "float16", "CHW", 1 / 255, 0.5)
    assert np.array_equal(got, numpy_epilogue(a, "float16", "CHW", 1 / 255, 0.5))
