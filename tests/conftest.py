import io
import os
import sys

import numpy as np
import pytest

# The suite runs with the small-image CPU route pinned at its fixed value: no measurement, no background thread
# and no settings file (tests/test_route.py lifts the pin for its own tests, with a temporary settings file).
os.environ.setdefault("EGJ_SMALL_CPU_PIXELS", "200000")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def reference(data, colorspace="RGB"):
    """libjpeg-turbo's output (simplejpeg with the accurate IDCT and fancy upsampling; Pillow when simplejpeg
    refuses a harmless header warning). None if libjpeg-turbo cannot decode it (see _pillow_reference)."""
    import simplejpeg
    try:
        return simplejpeg.decode_jpeg(data, colorspace=colorspace, fastdct=False, fastupsample=False, strict=False)
    except Exception:
        pass
    a = _pillow_reference(data, "L" if colorspace == "GRAY" else "RGB")
    if a is None:
        return None
    if colorspace == "GRAY":
        return a[..., None]
    return a[..., ::-1] if colorspace == "BGR" else a


def strict_refuses(data):
    """True if libjpeg-turbo's strict decode (simplejpeg, strict=True) refuses the file for what libjpeg reports
    (a warning or an error), not for a limit of TurboJPEG's API (a sampling layout it has no name for). A file
    the GPU path decodes under strict=True must not be one of these."""
    import simplejpeg
    try:
        simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=False, fastupsample=False)
        return False
    except Exception as e:                                       # noqa: BLE001
        return "subsampling level" not in str(e)


def sized(corpus, n, min_pixels=64 * 64):
    """The first n corpus entries the GPU path decodes (expected True) of at least min_pixels pixels: not the 1x1
    files the corpus starts with, which leave most of a decode's work untested."""
    import exact_gpu_jpeg as egj
    out = []
    for c in corpus:
        if c[2]:
            i = egj.inspect(c[1])
            if i["supported"] and i["width"] * i["height"] >= min_pixels:
                out.append(c)
                if len(out) == n:
                    return out
    raise AssertionError(f"the corpus has fewer than {n} such files")


def _pillow_reference(data, mode):
    """Pillow's decode (libjpeg-turbo inside) with Pillow's own default for damaged files, so that a file
    libjpeg-turbo rejects (a bad table after the scan, a second scan) is None here. Only when Pillow's complaint
    is that the data ends early is it decoded again with LOAD_TRUNCATED_IMAGES: libjpeg-turbo treats a missing
    end (EOI) as a warning and supplies it. The switch is put back as it was."""
    import io
    import numpy as np
    from PIL import Image, ImageFile
    was = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        for truncated in (False, True):
            ImageFile.LOAD_TRUNCATED_IMAGES = truncated
            try:
                with Image.open(io.BytesIO(data), formats=("JPEG", "MPO")) as im:
                    if mode == "L":
                        im.draft("L", im.size)
                    return np.asarray(im.convert(mode))
            except OSError as e:
                if truncated or "truncated" not in str(e):
                    return None
            except Exception:                                    # noqa: BLE001
                return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = was
    return None


@pytest.fixture(autouse=True, scope="module")
def _gpu_memory_given_back_after_each_module():
    """The suite is one long process with hundreds of Decoders, each with a CUDA stream of its own, and CuPy's pool
    keeps freed blocks per stream: without this the process ends up holding gigabytes it will never use again
    (more in guard-band mode), on a card other programs share. After each test module: collect what tests left
    in reference cycles (a Decoder in a caught exception's frames keeps its last buffers), then give the pool's
    free blocks back."""
    yield
    import gc
    gc.collect()
    try:
        import cupy as cp
    except ImportError:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(scope="session")
def corpus():
    import corpus as c
    return c.generate()


@pytest.fixture(scope="session")
def dec():
    pytest.importorskip("cupy")
    import exact_gpu_jpeg as egj
    return egj.Decoder()


@pytest.fixture
def redos(monkeypatch):
    """A list that grows by one each time the single-image path decodes a file a second time because its quick
    synchronisation passes had not settled. Tests of the "redo_" corpus files assert on it, so that those files
    keep exercising that path (a change that lets them settle makes the tests fail, and the files must grow)."""
    from exact_gpu_jpeg import _decoder as D
    seen, real = [], D.Decoder._decode

    def _decode(self, buf, h, colorspace, timings, eager=True):
        if not eager:
            seen.append(1)
        return real(self, buf, h, colorspace, timings, eager)
    monkeypatch.setattr(D.Decoder, "_decode", _decode)
    return seen


@pytest.fixture(scope="session")
def redo_files(corpus):
    """(name, bytes) of the corpus files whose decode is done a second time (see corpus.py)."""
    files = [(n, d) for n, d, e in corpus if n.startswith("redo_")]
    assert len(files) == 3 and all(e for n, d, e in corpus if n.startswith("redo_"))
    return files
