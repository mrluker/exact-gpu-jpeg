"""exact-gpu-jpeg: GPU JPEG decoding that is byte-identical to libjpeg-turbo.

    import exact_gpu_jpeg as egj

    img = egj.decode("photo.jpg")                 # CuPy (H, W, 3) uint8 on the GPU; CPU fallback if needed
    img = egj.decode(data, fallback=False)        # None if the GPU path cannot decode this JPEG exactly
    dec = egj.Decoder(); img = dec.decode(path)   # one Decoder per thread; dec.reason says why it returned None
    egj.inspect(path)                             # header summary and whether the GPU path supports it

The GPU path decodes baseline (and extended-sequential Huffman) 8-bit JPEGs with one interleaved scan:
grayscale, 4:4:4, 4:2:2, 4:2:0 and 4:4:0, with or without restart markers. Its output equals libjpeg-turbo's
with the accurate integer IDCT (JDCT_ISLOW) and fancy upsampling (TurboJPEG/simplejpeg defaults), byte for
byte. Opt-in: fast_dct=True matches libjpeg-turbo's fast integer IDCT (JDCT_IFAST, simplejpeg fastdct=True)
instead. Everything else (progressive, arithmetic, CMYK, 12-bit, corrupt data, ...) returns None from the GPU
path, so callers can decode those on the CPU.
"""
from __future__ import annotations

import os
import threading

from ._header import Unsupported, parse
from ._header import env_int as _env_int
from ._header import header_warning as _header_warning

__version__ = "0.1.0"
__all__ = ["Decoder", "Encoder", "decode", "decode_batch", "encode", "decode_cpu", "inspect", "set_cpu_decoder", "to_torch", "apply_epilogue",
           "self_check", "device_report", "clear_settings", "FastModeUnavailable", "Unsupported", "__version__", "timing",
           "CoefHandle", "RepeatCache"]


def __getattr__(name):
    if name == "Decoder":
        from ._decoder import Decoder
        return Decoder
    if name == "apply_epilogue":
        from ._decoder import apply_epilogue
        return apply_epilogue
    if name == "CoefHandle":
        from ._decoder import CoefHandle
        return CoefHandle
    if name == "RepeatCache":
        from ._repeat import RepeatCache
        return RepeatCache
    if name == "Encoder":
        from ._encoder import Encoder
        return Encoder
    raise AttributeError(name)


def _read(src):
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as f:
            return f.read()
    return src


def inspect(src):
    """Header summary without touching the GPU: dict(supported, reason, width, height, subsampling,
    restart_interval). `supported` means the header is one the GPU path handles; the entropy-coded data can
    still turn out to be corrupt."""
    data = _read(src)
    try:
        h = parse(data)
    except Unsupported as e:
        return dict(supported=False, reason=str(e))
    return dict(supported=True, reason=None, width=h.width, height=h.height, components=h.ncomp,
                subsampling=h.subsampling, restart_interval=h.dri)


# ---- CPU fallback ----

class FastModeUnavailable(RuntimeError):
    """fast_dct / fast_upsample was asked for and the CPU decoder at hand cannot decode in that mode, so it must not
    stand in for the GPU: it would return other pixels than the caller asked for."""


_pil_lock = threading.Lock()


_cpu_tl = threading.local()             # .warning: the default CPU decoder's last libjpeg-turbo warning (strict=False)


def _cpu_warning():
    return getattr(_cpu_tl, "warning", None)


def _cpu_checked():
    """Whether the CPU decoder's last decode in this thread saw every libjpeg-turbo warning: True for simplejpeg,
    False for Pillow (it decodes past them; the library still applies the header rules and what the GPU path
    found), None for a decoder set with set_cpu_decoder()."""
    return getattr(_cpu_tl, "checked", None)


def _cpu_reason(reason):
    """A route reason, with a note when the pixels came from Pillow."""
    if _cpu_checked() is False:
        return (reason or "decoded by the CPU decoder") + "; decoded by Pillow, which does not report libjpeg-turbo's warnings inside the scan"
    return reason


def _api_limit(e):
    """True if simplejpeg's error is a limit of TurboJPEG's API, not something libjpeg reports about the file."""
    return "subsampling level" in str(e)


def _default_cpu(data, colorspace, fast_dct=False, fast_upsample=False, strict=True):
    """simplejpeg (libjpeg-turbo: accurate IDCT unless fast_dct, fancy upsampling unless fast_upsample) if
    installed, else Pillow, which decodes in the accurate mode only. The pixels are always those of the mode asked
    for: in a fast mode without simplejpeg this raises FastModeUnavailable instead of decoding."""
    fast = fast_dct or fast_upsample
    _cpu_tl.warning = None
    _cpu_tl.checked = True
    pil_warning = None
    # strict (the default): a file libjpeg-turbo decodes only with a warning raises, naming the warning, as the GPU
    # path declines it. strict=False: libjpeg-turbo's image, and the warning is kept for the caller (info).
    hw = _header_warning(data)
    if hw and strict:
        raise ValueError(f"libjpeg-turbo warns: {hw} (strict=True)")
    try:
        import simplejpeg
    except ImportError:
        simplejpeg = None
    if simplejpeg is not None:
        try:
            a = simplejpeg.decode_jpeg(data, colorspace=colorspace, fastdct=fast_dct, fastupsample=fast_upsample)
            _cpu_tl.warning = hw
            return a
        except ValueError as e1:
            try:                  # libjpeg-turbo's non-strict decode: an image means e1 was a warning
                a = simplejpeg.decode_jpeg(data, colorspace=colorspace, fastdct=fast_dct, fastupsample=fast_upsample,
                                           strict=False)
            except ValueError:
                a = None
            if a is not None:
                if strict:
                    raise ValueError(f"libjpeg-turbo warns: {e1} (strict=True)") from None
                _cpu_tl.warning = str(e1)
                return a
            if fast:              # Pillow could not give the fast mode: simplejpeg's own error stands
                raise
            if not _api_limit(e1):
                # TurboJPEG refused the file in both modes for what libjpeg itself reports as a warning or an
                # error (a JFIF version, a bad ICC segment, extraneous bytes in the header): strict raises with
                # its text; strict=False takes Pillow's decode (the libjpeg API, which goes on past warnings) and
                # names the text as the warning. (Pillow could not tell: it decodes past every warning silently.)
                if strict:
                    raise ValueError(f"libjpeg-turbo: {e1} (strict=True)") from None
                pil_warning = str(e1)
            # else: a file TurboJPEG's API cannot take (a sampling layout it has no name for): Pillow decodes it
    elif fast:
        raise FastModeUnavailable(
            "the CPU decoder cannot decode with fast_dct / fast_upsample: only simplejpeg has libjpeg-turbo's fast "
            "modes and it is not installed (pip install simplejpeg)")
    _cpu_tl.checked = False
    # Pillow from here on: it decodes past libjpeg's warnings without a word. What the GPU path found, when it
    # declined this file, still counts: a warning it saw raises under strict (and is named otherwise); unused
    # bytes in the scan, which libjpeg-turbo may or may not report, cannot be settled here.
    gpu = getattr(_cpu_tl, "gpu_reason", None) or ""
    if gpu.startswith("libjpeg-turbo warns: "):
        seen = gpu[len("libjpeg-turbo warns: "):].split(" (declined because strict=True")[0]
        if strict:
            raise ValueError(f"libjpeg-turbo warns: {seen} (strict=True; found by the GPU path)")
        pil_warning = "; ".join(x for x in (pil_warning, seen) if x)
    elif gpu.startswith("unused bytes after the last block"):
        if strict:
            raise ValueError("unused bytes in the scan, which libjpeg-turbo may report as extraneous bytes; without "
                             "TurboJPEG (simplejpeg) for this file that cannot be settled (strict=True; strict=False "
                             "decodes it)")
        pil_warning = "; ".join(x for x in (pil_warning, "possibly extraneous bytes in the scan (not checked)") if x)
    import io

    import numpy as np
    from PIL import Image, ImageFile
    # JPEG only (Pillow opens many formats; libjpeg-turbo answers anything else with an error), and with Pillow's
    # own default for damaged files: a program that sets ImageFile.LOAD_TRUNCATED_IMAGES (common in training code)
    # would otherwise get an image for a file libjpeg-turbo rejects. The switch is module-wide in Pillow, so it is
    # cleared only for this decode, under a lock, and put back.
    # (Pillow's switch, cleared first: a file libjpeg-turbo rejects must raise here. Only when Pillow's complaint
    # is that the data ends early, a warning for libjpeg-turbo, is the file decoded again with the switch set.)
    a = None
    with _pil_lock:
        was = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            for truncated in ((False,) if strict else (False, True)):
                ImageFile.LOAD_TRUNCATED_IMAGES = truncated
                try:
                    with Image.open(io.BytesIO(data), formats=("JPEG",)) as im:   # (an MPO opens as JPEG too)
                        mode = "L" if colorspace == "GRAY" else "RGB"
                        if mode == "L":
                            im.draft("L", im.size)   # libjpeg's own grayscale output (the Y channel), not from RGB
                        a = np.asarray(im.convert(mode))
                    if truncated:
                        _cpu_tl.warning = "Premature end of JPEG file"
                    break
                except OSError as e:
                    if truncated or "truncated" not in str(e) or strict:
                        raise
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = was
    if hw or pil_warning:
        _cpu_tl.warning = "; ".join(dict.fromkeys(x for x in (hw, pil_warning, _cpu_tl.warning) if x))
    if colorspace == "GRAY":
        return a[..., None]
    return a[..., ::-1].copy() if colorspace == "BGR" else a


_cpu_decoder = _default_cpu


def set_cpu_decoder(fn):
    """fn(data: bytes, colorspace: str) -> numpy (H, W, C) uint8. None restores the default (simplejpeg, else
    Pillow). The library cannot check a decoder of your own: with one set, the pixels of every image it decodes
    (files the GPU path declines) are your responsibility, and device_report() says so. If fn also takes the
    keywords fast_dct and fast_upsample it is called with them; if it does not, a file the GPU path declines while
    one of those modes is asked for raises FastModeUnavailable instead of being decoded in another mode."""
    global _cpu_decoder
    _cpu_decoder = fn or _default_cpu


def decode_cpu(src, colorspace="RGB", fast_dct=False, fast_upsample=False, strict=True):
    """Decode with the CPU decoder (numpy array). fast_dct / fast_upsample reach a decoder set with set_cpu_decoder
    only when it takes those keywords; otherwise a fast mode raises FastModeUnavailable."""
    if colorspace not in ("RGB", "BGR", "GRAY"):
        raise ValueError("colorspace must be one of ('RGB', 'BGR', 'GRAY')")
    _cpu_tl.warning = None                  # (a decoder of the program's own reports none: no earlier call's warning)
    _cpu_tl.checked = None                  # (nor says whether it checked: unknown)
    if _cpu_decoder is _default_cpu:
        return _default_cpu(bytes(_read(src)), colorspace, fast_dct, fast_upsample, strict)
    if _takes_fast_modes(_cpu_decoder):
        return _cpu_decoder(bytes(_read(src)), colorspace, fast_dct=fast_dct, fast_upsample=fast_upsample)
    if fast_dct or fast_upsample:
        raise FastModeUnavailable("the CPU decoder set with set_cpu_decoder() takes no fast_dct / fast_upsample "
                                  "keywords, so it cannot decode in the mode asked for")
    return _cpu_decoder(bytes(_read(src)), colorspace)


def _takes_fast_modes(fn):
    import inspect
    try:
        p = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return ("fast_dct" in p and "fast_upsample" in p) or any(v.kind is v.VAR_KEYWORD for v in p.values())


def _cpu_fallback(src, colorspace, fast_dct, fast_upsample, reason, strict=True):
    """The CPU decoder standing in for the GPU path on a file it declined. Never in another mode than asked for."""
    _cpu_tl.gpu_reason = reason                        # (what the GPU path found: used where Pillow decodes)
    try:
        return decode_cpu(src, colorspace, fast_dct, fast_upsample, strict)
    except FastModeUnavailable as e:
        raise FastModeUnavailable(f"exact_gpu_jpeg: the GPU path declined this file ({reason}) and {e}. Pass "
                                  "fallback=False to get None for such files instead.") from None
    finally:
        _cpu_tl.gpu_reason = None


def _cpu_decoder_report():
    """Which CPU decoder stands in for the GPU path, and for which modes."""
    if _cpu_decoder is not _default_cpu:
        return ("custom (set_cpu_decoder): the pixels of images it decodes are the caller's responsibility; fast modes "
                + ("are passed to it" if _takes_fast_modes(_cpu_decoder) else "raise FastModeUnavailable"))
    try:
        import simplejpeg
        return f"simplejpeg {getattr(simplejpeg, '__version__', '?')} (accurate and fast modes)"
    except ImportError:
        pass
    try:
        import PIL
        return (f"Pillow {PIL.__version__} (accurate mode only: with fast_dct / fast_upsample the CPU decoder never "
                "stands in; a file the GPU path declines then raises FastModeUnavailable)")
    except ImportError:
        return "none installed: a file the GPU path declines raises ImportError with fallback=True"


# ---- convenience ----

_tl = threading.local()


def _thread_decoder(device, fast_dct=False):
    from ._decoder import Decoder
    import cupy as cp
    dev = device if device is not None else cp.cuda.Device().id
    decs = getattr(_tl, "decs", None)
    if decs is None:
        decs = _tl.decs = {}
    key = (dev, True) if fast_dct else dev
    d = decs.get(key)
    if d is None:
        d = decs[key] = Decoder(dev, fast_dct=bool(fast_dct))
    return d


# One image of this many pixels or fewer is decoded on the CPU by decode() and uploaded: a GPU decode costs about
# 1.0-1.2 ms up to 0.25 MP, libjpeg-turbo needs 0.4 ms at 0.05 MP and 1.2 ms at 0.2 MP for a 4:2:0 file (GTX 1070,
# i7-6700: bench/results/20261117/call_floor.md). The pixels are the same. EGJ_SMALL_CPU_PIXELS=0 turns it off.
SMALL_CPU_PIXELS = _env_int("EGJ_SMALL_CPU_PIXELS", 200000, 0, 1 << 40)
_SMALL_FIXED = SMALL_CPU_PIXELS      # (a program that assigns SMALL_CPU_PIXELS pins the value, as the variable does)
SMALL_REASON = "small image: one image this small decodes faster on the CPU"


def _frame_pixels(src):
    """Width x height from the frame header, found by stepping over the marker segments (nothing is parsed or
    checked); None when the walk does not reach a frame header."""
    try:
        b = src if isinstance(src, (bytes, bytearray)) else memoryview(src).cast("B")
    except (TypeError, ValueError):
        return None
    n, i = len(b), 2
    if n < 4 or b[0] != 0xFF or b[1] != 0xD8:
        return None
    while i + 9 <= n and b[i] == 0xFF:
        m = b[i + 1]
        if m == 0xFF:                                       # a fill byte
            i += 1
        elif 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            return ((b[i + 5] << 8) | b[i + 6]) * ((b[i + 7] << 8) | b[i + 8])
        elif m == 0xDA or m == 0xD9 or m < 0xC0:
            return None
        else:
            i += 2 + ((b[i + 2] << 8) | b[i + 3])
    return None


def _small_on_cpu(src, limit, colorspace, device, dtype, layout, scale, bias, fast_dct, fast_upsample, strict=True):
    """(image on the GPU, None) if `src` is a JPEG of at most `limit` pixels that the default CPU decoder decodes;
    else (None, src or its bytes) and the caller carries on with the GPU path."""
    if isinstance(src, (str, os.PathLike)):
        try:
            if os.path.getsize(src) > 2 * limit + 65536:   # far too large a file for so few pixels: leave it be
                return None, src
        except OSError:
            return None, src
        src = _read(src)
    else:
        try:
            if len(src) > 2 * limit + 65536:                # the same test for bytes: a larger image costs nothing here
                return None, src
        except TypeError:
            pass
    px = _frame_pixels(src)
    if px is not None and px > limit:                       # the usual exit for a larger image: nothing parsed
        return None, src
    try:
        h = parse(src)
    except Unsupported:
        return None, src                                    # the GPU path declines it and says why
    if h.width * h.height > limit:
        return None, src
    # only a file simplejpeg (libjpeg-turbo) takes: it reports libjpeg-turbo's warnings for strict, and has the
    # fast modes. Without it, or for a sampling layout TurboJPEG's API refuses, the GPU path decodes the image (it
    # finds the warnings itself); Pillow would decode past them silently.
    try:
        import simplejpeg
        simplejpeg.decode_jpeg_header(bytes(src))
    except Exception:                                       # noqa: BLE001 - ImportError, or TurboJPEG refuses it
        return None, src
    try:
        a = _default_cpu(bytes(src), colorspace, fast_dct, fast_upsample, strict)
    except Exception:                                       # noqa: BLE001 - whatever it is, the usual path reports it
        return None, src
    import cupy as cp
    with cp.cuda.Device(device if device is not None else cp.cuda.Device().id):
        return _to_gpu(a, dtype, layout, scale, bias), None


def _to_gpu(a, dtype, layout, scale, bias):
    """A CPU-decoded image on the current GPU, with the output options applied, and final when this returns: the
    options are applied on the host (NumPy, the same float32 steps and rounding as the GPU kernel), then the
    copy is waited for, so work the caller queues on any stream sees the pixels, as with a GPU decode."""
    import cupy as cp
    from ._decoder import apply_epilogue
    return cp.asarray(apply_epilogue(a, dtype, layout, scale, bias), blocking=True)


def decode(src, colorspace="RGB", *, fallback=True, device=None, out="cupy", info=None, dtype="uint8", layout="HWC",
           scale=None, bias=None, fast_dct=False, fast_upsample=False, cpu_below=None, cache=None, check="full",
           strict=True):
    """Decode a JPEG (path or bytes-like) to the GPU.

    colorspace: 'RGB', 'BGR' or 'GRAY'. fallback: if True, JPEGs the GPU path does not decode go through the CPU
    decoder (see set_cpu_decoder) and are uploaded; if False, those return None. out: 'cupy' or 'torch'.
    info: optional dict, filled with 'path' ('gpu' or 'cpu') and 'reason' (why the GPU path declined).
    dtype / layout / scale / bias: float output written by the decoder itself (see Decoder.decode), e.g.
    decode(f, dtype="float16", layout="CHW", scale=1 / (255 * std), bias=-mean / std) for a normalised PyTorch
    input in one pass; the CPU fallback gets the same values (apply_epilogue).
    fast_dct: libjpeg-turbo's fast integer IDCT (JDCT_IFAST, simplejpeg fastdct=True) instead of the accurate one,
    on the GPU and in the default CPU fallback; less accurate, so off by default.
    fast_upsample: opt-in, libjpeg-turbo's fast upsampling (simplejpeg fastupsample=True), byte-identical to it:
    chroma repeated instead of interpolated (not the default's bytes; a compatibility option, not faster here).
    Thread-safe: each thread gets its own Decoder (and CUDA stream).
    Errors: the GPU path never raises for bad JPEG data (it declines); with fallback=True, a file the CPU decoder
    cannot read either raises that decoder's exception (ValueError or OSError for simplejpeg / Pillow).
    cpu_below: one image of at most this many pixels is decoded by the CPU decoder and uploaded, which is faster
    than a GPU decode for a single small image and gives the same pixels (info: path 'cpu'). Default
    SMALL_CPU_PIXELS (200,000; the environment variable EGJ_SMALL_CPU_PIXELS); 0 always uses the GPU. It applies
    with fallback=True and the default CPU decoder only, and not to lists (a batch is faster on the GPU).
    cache: a RepeatCache (opt-in, default none): a file decoded before is decoded again from a copy kept on the
    GPU, the same pixels without the entropy decode; a first decode keeps a copy. check: how a copy is known to be
    current, "full" (default: the file is read and its bytes compared by check value) or "light" (see
    RepeatCache). Not for lists, and not for images the CPU route takes.
    strict (default True): a file libjpeg-turbo decodes only with a warning (no end marker, data that ends early,
    extraneous bytes before a marker, an unknown JFIF version or Adobe transform) is not decoded: the GPU path
    declines it and the CPU decoder raises, naming the warning. strict=False: libjpeg-turbo's image (as djpeg gives
    it, a file that ends early with its missing part grey), and info['warning'] names the warning.
    """
    if isinstance(src, (list, tuple)):         # several JPEGs: decoded together (decode_batch)
        return decode_batch(src, colorspace, fallback=fallback, device=device, out=out, info=info, dtype=dtype,
                            layout=layout, scale=scale, bias=bias, fast_dct=fast_dct, fast_upsample=fast_upsample,
                            strict=strict)
    if colorspace not in ("RGB", "BGR", "GRAY"):
        raise ValueError("colorspace must be one of ('RGB', 'BGR', 'GRAY')")
    from . import _route
    entered = _route.enter()
    try:
        return (_decode1_timed if TIMING else _decode1)(src, colorspace, fallback, device, out, info, dtype, layout,
                                                        scale, bias, fast_dct, fast_upsample, cpu_below, cache, check,
                                                        strict)
    finally:
        _route.leave(entered)


TIMING = os.environ.get("EGJ_TIMING") == "1"       # see timing()
_timing_reported = set()


def timing(on=True):
    """Per-call timing for decode() of one image and for encode(): off by default; the environment variable
    EGJ_TIMING=1 switches it on from outside. While it is on, each such call prints one line to stderr: the route
    taken ('gpu' or 'cpu') and why, the call's wall time, and the GPU time of each stage; the first line of a
    process (per GPU) is preceded by device_report(). A dict passed as info= gets the same values ('ms': stage ->
    milliseconds, 'total_ms'). Timing adds GPU events and one stream wait to the call, so take end-to-end numbers
    with it off. Lists (batches) are not timed. Returns the previous setting."""
    global TIMING
    old, TIMING = TIMING, bool(on)
    return old


def _timing_line(kind, dev, d):
    import sys
    if dev not in _timing_reported:
        _timing_reported.add(dev)
        print(f"exact_gpu_jpeg timing: device {device_report(dev)}", file=sys.stderr)
    why = f" ({d['reason']})" if d.get("reason") else ""
    stages = ", ".join(f"{k} {v:.3f}" for k, v in d["ms"].items())
    print(f"exact_gpu_jpeg timing: {kind} {d.get('path')}{why} {d['total_ms']:.3f} ms" + (f" | {stages}" if stages else ""),
          file=sys.stderr)


def _decode1_timed(src, colorspace, fallback, device, out, info, dtype, layout, scale, bias, fast_dct, fast_upsample,
                   cpu_below, cache=None, check="full", strict=True):
    from time import perf_counter
    d = info if info is not None else {}
    dec = _thread_decoder(device, fast_dct)
    old, dec.profile = dec.profile, True
    t = perf_counter()
    try:
        img = _decode1(src, colorspace, fallback, device, out, d, dtype, layout, scale, bias, fast_dct, fast_upsample,
                       cpu_below, cache, check, strict)
        dec.stream.synchronize()
    finally:
        dec.profile = old
    d["total_ms"] = round((perf_counter() - t) * 1e3, 3)
    d["ms"] = dict(dec.stats.get("profile_ms", {})) if d.get("path") == "gpu" else {}
    _timing_line("decode", dec.device.id, d)
    return img


def _decode1(src, colorspace, fallback, device, out, info, dtype, layout, scale, bias, fast_dct, fast_upsample, cpu_below,
             cache=None, check="full", strict=True):
    if cpu_below is not None:
        limit = int(cpu_below)                             # pinned for this call
    elif SMALL_CPU_PIXELS != _SMALL_FIXED or not (fallback and _cpu_decoder is _default_cpu):
        limit = SMALL_CPU_PIXELS                           # pinned by the program, or the CPU route is not in play
    else:                                                  # the measured or saved value, else the fixed default
        import cupy as cp
        from . import _route
        limit = _route.limit(device if device is not None else cp.cuda.Device().id, SMALL_CPU_PIXELS)
    if limit > 0 and fallback and _cpu_decoder is _default_cpu:
        img, src = _small_on_cpu(src, limit, colorspace, device, dtype, layout, scale, bias, fast_dct, fast_upsample,
                                 strict)
        if img is not None:
            if info is not None:
                info["path"], info["reason"], info["warning"] = "cpu", SMALL_REASON, _cpu_warning()
                info["warnings_checked"] = _cpu_checked()
            return to_torch(img) if out == "torch" else img
    dec = _thread_decoder(device, fast_dct)
    dec.fast_upsample, dec.strict = fast_upsample, strict
    if cache is None:
        img = dec.decode(src, colorspace, dtype=dtype, layout=layout, scale=scale, bias=bias)
    else:
        img = cache._decode(dec, src, colorspace, None, check, dtype, layout, scale, bias)[0]
    if info is not None:
        info["path"] = "gpu" if img is not None else None
        info["reason"], info["warning"] = dec.reason, dec.warning
        info["warnings_checked"] = True if img is not None else None
    if img is None and fallback:
        img = _upload(dec, _cpu_fallback(src, colorspace, fast_dct, fast_upsample, dec.reason, strict), dtype, layout,
                      scale, bias)
        if info is not None:
            info["path"], info["warning"], info["warnings_checked"] = "cpu", _cpu_warning(), _cpu_checked()
            info["reason"] = _cpu_reason(info["reason"])
    if img is not None and out == "torch":
        return to_torch(img)
    return img


def decode_batch(srcs, colorspace="RGB", *, fallback=True, device=None, out="cupy", info=None, dtype="uint8",
                 layout="HWC", scale=None, bias=None, fast_dct=False, fast_upsample=False, strict=True):
    """Decode several JPEGs (paths or bytes-like) to the GPU: a list of images, in the order given. `decode()` does
    this when it is given a list or tuple.

    JPEGs that share their Huffman tables and sampling factors (files from one camera, phone or encoder usually
    do) are decoded together by one set of kernel launches, which is several times faster than one call per file
    for small images; every image equals what decode() returns for it, byte for byte. The options are decode()'s.
    info: optional list, filled with one dict per image ('path': 'gpu' or 'cpu', or None for a declined image
    with fallback=False; 'reason': why the GPU path declined it). With fallback=False a declined image is None.
    Images decoded together are views of one GPU buffer: keeping one alive keeps that buffer (copy it to release
    the others). If the CPU decoder cannot read a file the GPU path declined, the call raises an error that names
    the file's index (and path), after `info` is filled for every file."""
    srcs = list(srcs)
    if info is not None and not isinstance(info, list):
        raise TypeError("info must be a list when decoding a list of files (one dict per file is put in it)")
    from . import _route
    entered = _route.enter()
    try:
        dec = _thread_decoder(device, fast_dct)
        dec.fast_upsample, dec.strict = fast_upsample, strict
        imgs = dec.decode_batch(srcs, colorspace, dtype=dtype, layout=layout, scale=scale, bias=bias)
    finally:
        _route.leave(entered)
    reasons = list(dec.reasons)
    warnings = list(dec.warnings) if len(dec.warnings) == len(imgs) else [None] * len(imgs)
    paths = ["gpu" if im is not None else None for im in imgs]
    checked = [True if im is not None else None for im in imgs]
    failure = None
    if fallback:
        for i, im in enumerate(imgs):
            if im is None:
                try:
                    imgs[i] = _upload(dec, _cpu_fallback(srcs[i], colorspace, fast_dct, fast_upsample, reasons[i],
                                                         strict), dtype, layout, scale, bias)
                    paths[i], warnings[i], checked[i] = "cpu", _cpu_warning(), _cpu_checked()
                    reasons[i] = _cpu_reason(reasons[i])
                except Exception as e:                       # noqa: BLE001 - raised below, with the file named
                    failure = failure or (i, e)
    if info is not None:
        info[:] = [dict(path=p, reason=r, warning=w, warnings_checked=c)
                   for p, r, w, c in zip(paths, reasons, warnings, checked)]
    if failure is not None:
        i, e = failure
        name = f" ({os.fspath(srcs[i])})" if isinstance(srcs[i], (str, os.PathLike)) else ""
        msg = f"file {i}{name}: the GPU path declined it ({reasons[i]}) and the CPU decoder raised: {e}"
        try:
            err = type(e)(msg)                               # the same kind of error, with the file named
        except Exception:                                    # noqa: BLE001 - (a type that takes other arguments)
            err = RuntimeError(msg)
        raise err from e
    if out == "torch":
        imgs = [to_torch(im) if im is not None else None for im in imgs]
    return imgs


def _upload(dec, a, dtype, layout, scale, bias):
    """A CPU-decoded image to the GPU (and the output options applied). If the GPU cannot even hold an array, say
    why in one message instead of a bare CUDA error."""
    import cupy as cp
    from . import _selfcheck
    from ._decoder import apply_epilogue
    try:
        with dec.device:
            return _to_gpu(a, dtype, layout, scale, bias)
    except cp.cuda.memory.OutOfMemoryError:
        raise
    except (RuntimeError, ImportError) as e:
        why = _selfcheck.failed(dec.device.id) or _selfcheck.gpu_error(dec.device.id, e)
        raise RuntimeError(f"exact_gpu_jpeg: {why}; the image was decoded on the CPU but could not be copied to "
                           "the GPU") from e


def self_check(device=None):
    """Run the self-check now, all of it (decode, batch decode, encode, and the first-use checks of the fast IDCT,
    the fast upsampling and the output options), on one device: a few small
    built-in cases whose results must have the SHA-256 hashes stored in the package (pixels as libjpeg-turbo
    decodes them, files as Pillow writes them). It runs by itself, part by part, before the first decode, the
    first batch and the first encode of a process; call this to pay for it (and for compiling the kernels) at a
    time of your choosing. Having run every part, the process has also used the one CuPy routine that a first
    encode of more than 65,536 blocks would otherwise prepare (about 45 ms). Returns a dict: 'ok', 'reason' (None, or why the GPU path is now off for this process:
    "self-check failed on <GPU>, driver <version>: <case>"), 'gpu', 'device' and 'ms' per part.
    When it fails, decode() uses the CPU decoder (one RuntimeWarning says why), Decoder.decode() declines every
    file with the reason, and encode() raises it. EGJ_SELFCHECK_FAIL=1 forces a failure, for testing."""
    import cupy as cp
    from . import _route, _selfcheck
    dev = device if device is not None else cp.cuda.Device().id
    _route.wait_background(dev)                # a background measurement running now ends first (they disturb each other)
    r = _selfcheck.run_all(dev)
    if (r["ok"] and SMALL_CPU_PIXELS == _SMALL_FIXED and _cpu_decoder is _default_cpu and not _route.PINNED_BY_ENV
            and _route.describe(dev, SMALL_CPU_PIXELS)["source"].startswith("default")):
        _route.measure_now(dev)                # the small-image CPU route, measured now instead of in the background
    r["cpu_below"] = _cpu_below_report(dev)
    return r


def _cpu_below_report(dev):
    """The small-image CPU route's threshold in use on the device, and where it came from (as device_report)."""
    from . import _route
    if SMALL_CPU_PIXELS != _SMALL_FIXED:
        return dict(pixels=SMALL_CPU_PIXELS, source="pinned by the program (SMALL_CPU_PIXELS)")
    return _route.describe(dev, SMALL_CPU_PIXELS)


def device_report(device=None):
    """What the library found on this GPU and what it chose, as a dict to print or to put in a bug report: the GPU
    name and driver, compute capability, memory, CUDA runtime and kernel compiler versions, the minimum it needs,
    the settings derived from those values (the memory budget share and the data one batch may hold), which parts
    of the self-check have passed, and whether the GPU path is on (or why it is off). It runs no kernel; call
    self_check() first to have the self-check's result in it. copy_check: the check of the kernels behind the
    packed form (it runs at the first use of that form on the GPU, not before)."""
    import cupy as cp
    from . import _copycheck, _route, _selfcheck
    dev = device if device is not None else cp.cuda.Device().id
    r = _selfcheck.report(dev)
    r["copy_check"] = _copycheck.report(dev)
    r["cpu_below"] = _cpu_below_report(dev)
    r["cpu_decoder"] = _cpu_decoder_report()
    return r


def clear_settings():
    """Delete the one settings file this library writes (the measured size below which one image is decoded on
    the CPU, see cpu_below) and forget what this process measured: the fixed default applies again until the next
    measurement. Returns True if a file was removed. Harmless at any time."""
    from . import _route
    return _route.clear()


def to_torch(arr):
    """CuPy array -> torch tensor on the same GPU, sharing memory (DLPack)."""
    import torch
    return torch.from_dlpack(arr)


def _thread_encoder(device, img=None):
    from ._encoder import Encoder
    import cupy as cp
    if device is None and img is not None:          # a GPU image: the encoder of the GPU it is on
        if isinstance(img, cp.ndarray):
            device = img.device.id
        elif getattr(img, "is_cuda", False) and getattr(getattr(img, "device", None), "index", None) is not None:
            device = img.device.index
    dev = device if device is not None else cp.cuda.Device().id
    encs = getattr(_tl, "encs", None)
    if encs is None:
        encs = _tl.encs = {}
    e = encs.get(dev)
    if e is None:
        e = encs[dev] = Encoder(dev)
    return e


def encode(img, quality=75, subsampling="420", icc_profile=None, exif=None, *, device=None, path=None, ready=None,
           info=None):
    """JPEG bytes for an (H, W, 3) uint8 RGB image on the GPU (CuPy array or CUDA torch tensor; a numpy array is
    uploaded). Byte-identical to Pillow 13's `Image.fromarray(img).save(f, "JPEG", quality=quality,
    subsampling=subsampling, icc_profile=icc_profile, exif=exif)` (libjpeg-turbo 3.2.0): baseline, standard
    Huffman tables. subsampling: "444", "422", "420" or Pillow's 0, 1, 2. Thread-safe (an Encoder per thread).
    path: write the file there instead (download and file write overlap) and return its length.
    ready: a GPU event recorded after the image was written (see Encoder.encode_async, which also queues encodes
    without waiting, on Encoder objects of your own).
    info: optional dict; with timing() on it gets 'ms' (the GPU time of each stage) and 'total_ms'."""
    from . import _route
    entered = _route.enter()
    try:
        enc = _thread_encoder(device, img)
        if not TIMING:
            return enc.encode(img, quality, subsampling, icc_profile, exif, path=path, ready=ready)
        from time import perf_counter
        tm, t = {}, perf_counter()
        r = enc.encode(img, quality, subsampling, icc_profile, exif, timings=tm, path=path, ready=ready)
        d = info if info is not None else {}
        d.update(path="gpu", reason=None, total_ms=round((perf_counter() - t) * 1e3, 3),
                 ms={k: round(v * 1e3, 3) for k, v in tm.items()})
        _timing_line("encode", enc.device.id, d)
        return r
    finally:
        _route.leave(entered)
