"""How decode(path) reads the file (Decoder(read=...), EGJ_STREAM_READ): streamed for a lone decoder, the whole file
when another Decoder is decoding at the same moment. The image is the same either way."""
import threading

import numpy as np
import pytest

pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from conftest import reference  # noqa: E402


@pytest.fixture()
def files(corpus, tmp_path):
    out = []
    for k, (name, data, expected) in enumerate(corpus[::9]):
        p = tmp_path / f"f{k}.jpg"
        p.write_bytes(data)
        out.append((str(p), data, expected))
    return out


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and bool((a == b).all())


def test_lone_decoder_streams_and_modes_give_the_same_image(files):
    plain = egj.Decoder()
    for mode, how in (("auto", "streamed"), ("streamed", "streamed"), ("whole", "whole")):
        d = egj.Decoder(read=mode)
        for path, data, expected in files:
            img = d.decode(path)
            assert d.stats["read"] == how
            assert (img is not None) == expected
            assert _same(img, plain.decode(data))
            if img is not None:
                assert np.array_equal(img.get(), reference(data))
        assert D._busy == 0
    d = egj.Decoder()
    d.decode(files[0][1])
    assert d.stats["read"] == "memory"
    assert d.upload_ahead(files[0][0]) and d.decode(files[0][0]) is not None and d.stats["read"] == "staged"


def test_another_decoder_at_work_switches_auto_to_whole(files, monkeypatch):
    d, s = egj.Decoder(), egj.Decoder(read="streamed")
    path, data, _ = files[0]
    monkeypatch.setattr(D, "_busy", 1)                 # as if another Decoder were inside decode()
    assert _same(d.decode(path), egj.Decoder(read="whole").decode(path)) and d.stats["read"] == "whole"
    assert s.decode(path) is not None and s.stats["read"] == "streamed"      # forced modes do not switch
    monkeypatch.setattr(D, "_busy", 0)
    assert d.decode(path) is not None and d.stats["read"] == "streamed"


def test_threads(files):
    """Several decoders at once: every image right, some reads whole, the counter back at 0 (also after declined
    files and after an exception)."""
    plain = egj.Decoder()
    want = {p: plain.decode(d) for p, d, _ in files}
    want = {p: None if v is None else v.get() for p, v in want.items()}
    seen, bad = set(), []

    def work():
        d = egj.Decoder()
        for _ in range(6):
            for p, _, _ in files:
                img = d.decode(p)
                seen.add(d.stats.get("read"))
                w = want[p]
                if (img is None) != (w is None) or (img is not None and not np.array_equal(img.get(), w)):
                    bad.append(p)
    ts = [threading.Thread(target=work) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not bad
    assert "whole" in seen
    with pytest.raises(ValueError):
        egj.Decoder().decode(files[0][0], "XYZ")
    assert D._busy == 0


def test_option_and_environment():
    with pytest.raises(ValueError):
        egj.Decoder(read="fast")
    assert D.READ_MODE in D.READ_MODES and D.STREAM_READ == (D.READ_MODE != "whole")
    assert egj.Decoder().read == D.READ_MODE


def test_a_batch_counts_as_a_decoder_at_work(files, monkeypatch):
    from exact_gpu_jpeg import _batch
    seen = []
    real = _batch.decode_batch

    def spy(dec, *a):
        seen.append(D._busy)
        return real(dec, *a)
    monkeypatch.setattr(_batch, "decode_batch", spy)
    d = egj.Decoder()
    paths = [p for p, _, e in files if e][:6]
    imgs = d.decode_batch(paths)
    assert seen == [1] and D._busy == 0
    assert all(_same(a, d.decode(p)) for a, p in zip(imgs, paths))
    with pytest.raises(TypeError):
        d.decode_batch([object()])
    assert D._busy == 0                                 # also when the batch raises
