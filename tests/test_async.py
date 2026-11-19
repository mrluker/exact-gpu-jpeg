"""Decoder.decode_async: the same pixels as decode(); out= decodes into the caller's array; work on another stream
that waits for job.event sees the finished pixels; a new decode concludes the previous job."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj


def test_async_matches_decode(corpus):
    a, b = egj.Decoder(), egj.Decoder()
    n = 0
    for name, data, expected in corpus[::3]:
        ref = a.decode(data)
        job = b.decode_async(data)
        if ref is None:
            assert job is None or job.result() is None, name
            continue
        assert job is not None, (name, b.reason)
        img = job.result()
        assert img is not None and np.array_equal(img.get(), ref.get()), name
        n += 1
    assert n > 50


def test_out_buffer_and_event(corpus):
    """On a real-size image (the corpus's first files are 1x1): work on another stream that waits for job.event
    sees the finished pixels."""
    dec = egj.Decoder()
    name, data, _ = next(c for c in corpus if c[2] and c[0].startswith("sj_1080x1920"))
    ref = dec.decode(data).get()
    out = cp.empty(ref.shape, cp.uint8)
    job = dec.decode_async(data, out=out)
    other = cp.cuda.Stream(non_blocking=True)
    other.wait_event(job.event)
    with other:
        copy = out.copy()                       # queued behind the decode on another stream
    other.synchronize()
    assert job.result() is out
    assert np.array_equal(copy.get(), ref), name
    assert not job.redone


def test_wrong_out_shape_raises(corpus):
    dec = egj.Decoder()
    name, data, _ = next(c for c in corpus if c[2])
    h, w, c = dec.decode(data).shape
    with pytest.raises(ValueError):
        dec.decode_async(data, out=cp.empty((h + 1, w, c), cp.uint8))


def test_next_decode_concludes_job(corpus):
    dec = egj.Decoder()
    goods = [c for c in corpus if c[2] and len(c[1]) > 20000][:2]
    job = dec.decode_async(goods[0][1])
    second = dec.decode(goods[1][1])            # concludes the first job before starting
    assert job._done and job.result() is not None and second is not None
    from conftest import reference
    assert np.array_equal(job.result().get(), reference(goods[0][1]))      # neither damaged the other
    assert np.array_equal(second.get(), reference(goods[1][1]))


def _photo(w, h, seed=1, **kw):
    import io
    from PIL import Image
    a = np.random.default_rng(seed).integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


@pytest.mark.parametrize("which", [None, 0, 1, 2], ids=["eager group of one pass", "corpus redo file 1",
                                                        "corpus redo file 2", "corpus redo file 3"])
def test_a_redone_job_has_the_right_pixels_in_the_buffer_it_handed_out(monkeypatch, redo_files, redos, which):
    """When the eager passes have not settled, result() decodes again. The pixels must end up in the array the job
    handed out (job.out, or the caller's out=), which is the array queued work reads, and result() must return
    that same array: with the image's own output options, also when another decode's options came in between."""
    from exact_gpu_jpeg import _decoder as D
    from conftest import reference
    if which is None:
        monkeypatch.setattr(D, "SYNC_GROUP", 1)         # one eager pass: an image of several blocks of chunks is redone
        data = _photo(1024, 768)
    else:
        data = redo_files[which][1]                     # does not settle in the eager passes, with no setting changed
    want = reference(data)
    dec = egj.Decoder()
    job = dec.decode_async(data)
    handed = job.out
    img = job.result()
    assert job.redone and len(redos) == 1 and img is not None, dec.reason
    assert img is handed and job.out is handed          # the same array, not a new one
    assert np.array_equal(handed.get(), want)

    mine = cp.zeros(want.shape, cp.uint8)               # the caller's buffer
    job = dec.decode_async(data, out=mine)
    assert job.result() is mine and job.redone
    assert np.array_equal(mine.get(), want)

    plain = egj.Decoder()                               # output options: the job's own, at the redo too
    opts = dict(dtype="float32", layout="CHW", scale=1 / 255)
    ref = plain.decode(data, **opts).get()
    job = dec.decode_async(data, **opts)
    handed = job.out
    other = dec.decode(_photo(64, 48, seed=3))          # another decode, with other options, concludes the job
    assert other is not None and job.redone
    r = job.result()
    assert r is handed and r.dtype == cp.float32 and r.shape == ref.shape
    assert np.array_equal(r.get(), ref)


def test_work_queued_behind_a_redone_job_is_right_when_run_again(redo_files):
    """What the docs tell a pipeline to do: work queued behind job.event read the array before the second decode;
    after result() says redone, the same work on the same array gives the right answer."""
    from conftest import reference
    name, data = redo_files[0]
    want = reference(data)
    dec = egj.Decoder()
    mine = cp.cuda.Stream(non_blocking=True)
    job = dec.decode_async(data)
    mine.wait_event(job.event)
    with mine:
        early = job.out.copy()                          # queued behind the first attempt
    img = job.result()
    assert img is job.out and job.redone
    with mine:
        again = job.out.copy()                          # the same work, run again
    mine.synchronize()
    assert np.array_equal(again.get(), want), name
    assert early.shape == again.shape                   # (what `early` holds is not defined: the first attempt's pixels)
