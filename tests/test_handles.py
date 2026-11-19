"""Decoder.keep_coefs / take_coefs / decode_from_handle, form "raw": a decode's coefficients kept on the GPU give
the image again, the same bytes, from the pixel stage alone; nothing is kept unless it is asked for."""
import io

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _decoder as D  # noqa: E402
from exact_gpu_jpeg._header import parse  # noqa: E402
from conftest import reference, sized  # noqa: E402


def _jpeg(w, h, seed=0, **kw):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


def test_nothing_is_kept_by_default(corpus):
    dec = egj.Decoder()
    assert dec.keep_coefs is None
    name, data, expected = sized(corpus, 1)[0]
    assert dec.decode(data) is not None
    assert dec.take_coefs() is None and dec._coefs is None and dec._coefs_pending is None


def test_the_handle_gives_the_same_image_for_the_whole_corpus(corpus):
    dec, other = egj.Decoder(keep_coefs="raw"), egj.Decoder()
    n = declined = 0
    for name, data, expected in corpus[::3]:
        img = dec.decode(data)
        h = dec.take_coefs()
        if img is None:
            assert h is None, name                             # a declined file leaves no handle
            declined += 1
            continue
        assert isinstance(h, egj.CoefHandle) and h.form == "raw" and not h.freed, name
        assert (h.height, h.width) == img.shape[:2] and h.device == dec.device.id
        assert h.nbytes == D.layout(parse(data))["nblk"] * 133, name
        assert dec.take_coefs() is None                        # handed out once
        want = img.get()
        for d in (dec, other):                                 # the decoder it came from, and another one
            again = d.decode_from_handle(h)
            assert again is not None and np.array_equal(again.get(), want), (name, d.reason)
            assert d.stats["read"] == "handle"
        for cs in ("BGR", "GRAY"):
            assert np.array_equal(other.decode_from_handle(h, cs).get(), reference(data, cs)), (name, cs)
        n += 1
    assert n >= 80 and declined >= 1


def test_every_output_option_and_both_fast_modes():
    data = _jpeg(331, 187, 3, subsampling=2)
    dec = egj.Decoder(keep_coefs="raw")
    assert dec.decode(data) is not None
    h = dec.take_coefs()
    plain = egj.Decoder()
    for kw in (dict(dtype="float32"), dict(dtype="float16", layout="CHW"),
               dict(dtype="float32", layout="CHW", scale=[1 / 58.4, 1 / 57.1, 1 / 57.4], bias=[-2.1, -2.0, -1.8])):
        got, want = plain.decode_from_handle(h, **kw), plain.decode(data, **kw)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert np.array_equal(got.get().view(np.uint8), want.get().view(np.uint8)), kw
    for kw in (dict(fast_dct=True), dict(fast_upsample=True), dict(fast_dct=True, fast_upsample=True)):
        d = egj.Decoder(**kw)                                  # the decoder's own modes apply to a handle too
        assert np.array_equal(d.decode_from_handle(h).get(), d.decode(data).get()), kw
    tm = {}
    plain.profile = True
    assert plain.decode_from_handle(h, timings=tm) is not None
    assert tm["pixels"] > 0 and {"idct", "colour"} <= set(plain.stats["profile_ms"])


def test_a_queued_decode_hands_out_its_handle_after_it_is_concluded():
    data = _jpeg(640, 480, 4)
    dec = egj.Decoder(keep_coefs="raw")
    job = dec.decode_async(data)
    h = dec.take_coefs()                                       # concludes the job first
    assert h is not None and np.array_equal(dec.decode_from_handle(h).get(), job.result().get())
    assert np.array_equal(job.result().get(), reference(data))


def test_small_chunks_and_the_route_of_large_files():
    """Whatever way the entropy decode took (many passes, a decode redone with host checks, the large tables),
    the handle is that of the decode that concluded."""
    data = _jpeg(800, 600, 5, subsampling=1, restart_marker_blocks=3)
    want = reference(data)
    for kw, large in ((dict(chunk_bits=128), False), (dict(chunk_bits=256), False), ({}, True)):
        dec = egj.Decoder(keep_coefs="raw", **kw)
        dec._large_route = large
        img = dec.decode(data)
        assert img is not None and np.array_equal(img.get(), want), (kw, large, dec.reason)
        h = dec.take_coefs()
        assert np.array_equal(egj.Decoder().decode_from_handle(h).get(), want), (kw, large)


def test_guard_bands_and_a_handle_that_outlives_later_decodes(corpus):
    dec = egj.Decoder(canary=True, keep_coefs="raw")
    good = sized(corpus, 12)
    handles = []
    for name, data, expected in good:
        assert dec.decode(data) is not None, name
        handles.append((name, data, dec.take_coefs()))
    for name, data, h in handles:                              # all still valid after the decodes that followed
        assert np.array_equal(dec.decode_from_handle(h).get(), reference(data)), name


def test_free_wrong_arguments_and_the_memory_budget():
    data = _jpeg(320, 240, 6)
    dec = egj.Decoder(keep_coefs="raw")
    img = dec.decode(data)
    h = dec.take_coefs()
    small = egj.Decoder(max_gpu_bytes=1000)
    assert small.decode_from_handle(h) is None and "GPU memory" in small.reason
    with pytest.raises(TypeError):
        dec.decode_from_handle(img)
    with pytest.raises(ValueError):
        dec.decode_from_handle(h, "CMYK")
    n = h.nbytes
    h.free()
    assert h.freed and h.nbytes == 0 and n > 0
    with pytest.raises(ValueError, match="freed"):
        dec.decode_from_handle(h)
    dec.keep_coefs = "zip"
    with pytest.raises(ValueError, match="keep_coefs"):
        dec.decode(data)
    dec.keep_coefs = None
    assert dec.decode(data) is not None and dec.take_coefs() is None


def test_a_batch_keeps_nothing_and_leaves_the_setting_as_it_was():
    files = [_jpeg(256, 192, s) for s in range(4)]
    dec = egj.Decoder(keep_coefs="raw")
    assert dec.decode(files[0]) is not None
    imgs = dec.decode_batch(files)
    assert all(i is not None for i in imgs)
    assert dec.take_coefs() is None and dec.keep_coefs == "raw"
    assert dec.decode(files[1]) is not None and dec.take_coefs() is not None


@pytest.mark.parametrize("canary", [False, True])
def test_keep_coefs_converts_a_handle_between_packed_and_raw(canary):
    """keep_coefs applies to decode_from_handle(): a handle of the other form comes back from take_coefs() in the
    decoder's form, decoding to the same image; a handle of the same form gives nothing new."""
    data = _photo_for_conversion()
    want = reference(data)
    for first, second in (("packed", "raw"), ("raw", "packed")):
        dec = egj.Decoder(keep_coefs=first, canary=canary)
        dec.decode(data)
        h = dec.take_coefs()
        assert h.form == first
        conv = egj.Decoder(keep_coefs=second, canary=canary)
        assert np.array_equal(conv.decode_from_handle(h).get(), want)
        h2 = conv.take_coefs()
        assert h2 is not None and h2.form == second, (first, second)
        assert np.array_equal(egj.Decoder().decode_from_handle(h2).get(), want)
        same = egj.Decoder(keep_coefs=first, canary=canary)
        assert np.array_equal(same.decode_from_handle(h).get(), want)
        assert same.take_coefs() is None                       # the same form: nothing new is made


def test_a_handle_is_made_by_a_decode_only_and_cannot_be_changed():
    """(the 0.1.0 opacity rule) No handle from outside: the constructor refuses; what a handle says about itself
    cannot be set (a changed form would hand one layout's buffers to the other's kernels)."""
    data = _photo_for_conversion()
    dec = egj.Decoder(keep_coefs="packed")
    dec.decode(data)
    h = dec.take_coefs()
    with pytest.raises(TypeError):
        egj.CoefHandle("packed", parse(data), h.device, (), 1)
    for name, value in (("form", "raw"), ("device", 1), ("width", 1), ("height", 1), ("nbytes", 0)):
        with pytest.raises(AttributeError):
            setattr(h, name, value)
    assert h.form == "packed" and np.array_equal(dec.decode_from_handle(h).get(), reference(data))


def _photo_for_conversion():
    a = np.random.default_rng(9).integers(0, 256, (61, 81, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((640, 480), Image.BICUBIC).save(b, "JPEG", quality=90)
    return b.getvalue()
