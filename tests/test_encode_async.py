"""Encoder.encode_async and ready=: the same bytes as encode() (and so Pillow's), queued without a host wait."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from test_encode import content, pil  # noqa: E402

SETTINGS = [(90, "444"), (90, "422"), (90, "420"), (95, "444"), (87, "420")]


def _images(rng):
    return [content(rng, h, w, kind) for (h, w), kind in zip([(1, 1), (9, 9), (64, 48), (333, 501), (720, 1080)],
                                                              ["noise", "edges", "smooth", "noise", "smooth"])]


def test_async_equals_encode_and_pillow():
    rng = np.random.default_rng(2)
    enc = egj.Encoder()
    for a in _images(rng):
        x = cp.asarray(a)
        for q, ss in SETTINGS:
            want = pil(a, q, ss)
            assert enc.encode(x, q, ss) == want
            job = enc.encode_async(x, q, ss)
            assert job.result() == want
            assert job.result() == want                    # asking again gives the same bytes
            ev = cp.cuda.Event()
            ev.record(cp.cuda.get_current_stream())
            assert enc.encode_async(x, q, ss, ready=ev).result() == want
            assert enc.encode(x, q, ss, ready=ev) == want
            assert enc.encode_async(a, q, ss).result() == want          # a NumPy image
            assert enc.encode_async(a, q, ss, ready=ev).result() == want


def test_two_encoders_queued_back_to_back(tmp_path):
    """The two exports of one photo: both scans queued before any host wait, then collected."""
    rng = np.random.default_rng(3)
    big = content(rng, 2560, 3840, "smooth")
    small = np.ascontiguousarray(big[::2, ::2][:1365, :2048])
    xb, xs = cp.asarray(big), cp.asarray(small)
    ev = cp.cuda.Event()
    ev.record(cp.cuda.get_current_stream())
    e1, e2 = egj.Encoder(), egj.Encoder()
    j1 = e1.encode_async(xb, 95, "444", ready=ev)
    j2 = e2.encode_async(xs, 90, "444", ready=ev)
    assert j2.result() == pil(small, 90, "444")              # collected in any order
    assert j1.result() == pil(big, 95, "444")
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    j1 = e1.encode_async(xb, 95, "444", path=str(p1), ready=ev)
    j2 = e2.encode_async(xs, 90, "444", path=str(p2), ready=ev)
    n1, n2 = j1.result(), j2.result()
    assert p1.read_bytes() == pil(big, 95, "444") and n1 == p1.stat().st_size
    assert p2.read_bytes() == pil(small, 90, "444") and n2 == p2.stat().st_size


def test_ready_event_orders_the_encode_after_the_producer():
    """The image is still being uploaded behind other work on the producer stream; with ready= the encoder stream
    waits for it on the GPU (no host wait in encode_async)."""
    rng = np.random.default_rng(5)
    a = content(rng, 64, 48, "noise")
    want = pil(a, 90, "420")
    enc = egj.Encoder()
    for _ in range(4):
        stale = cp.full(a.shape, 200, cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        del stale
        busy = cp.zeros(1 << 26, cp.float32)
        for _ in range(30):
            busy += 1
        x = cp.asarray(a)
        ev = cp.cuda.Event()
        ev.record(cp.cuda.get_current_stream())
        job = enc.encode_async(x, 90, "420", ready=ev)
        assert job.result() == want
        del x, busy


def test_next_encode_collects_the_previous_job():
    rng = np.random.default_rng(6)
    a, b = content(rng, 40, 56, "noise"), content(rng, 24, 24, "edges")
    enc = egj.Encoder()
    j1 = enc.encode_async(cp.asarray(a), 90, "444")
    j2 = enc.encode_async(cp.asarray(b), 80, "420")            # collects j1 first
    assert enc.encode(a, 75, "422") == pil(a, 75, "422")       # collects j2 first
    assert j1.result() == pil(a, 90, "444") and j2.result() == pil(b, 80, "420")


def test_errors():
    enc = egj.Encoder()
    a = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError):
        enc.encode_async(a, 0, "420")
    with pytest.raises(ValueError):
        enc.encode_async(a, 90, "411")
    with pytest.raises(ValueError):
        enc.encode_async(np.zeros((8, 8), np.uint8), 90, "420")
    with pytest.raises(TypeError):
        enc.encode_async(a, 90, "420", exif="text")
    with pytest.raises(TypeError):
        enc.encode_async(cp.asarray(a), 90, "420", ready="not an event")
    assert enc.encode(a, 90, "420") == pil(a, 90, "420")       # the encoder is fine afterwards
    assert egj.encode(cp.asarray(a), 90, "420", ready=cp.cuda.Event()) == pil(a, 90, "420")


def test_torch_event():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA torch")
    rng = np.random.default_rng(7)
    a = content(rng, 64, 48, "noise")
    t = torch.from_numpy(a).cuda()
    ev = torch.cuda.Event()
    ev.record()
    assert egj.Encoder().encode_async(t, 90, "420", ready=ev).result() == pil(a, 90, "420")
