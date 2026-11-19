"""Which decoder gets one small image: the CPU decoder up to a pixel count, the GPU above it.

The count is where the two routes take the same time, and that depends on the machine: a fast GPU beside a slow
CPU wants a low count, the reverse a high one. So it is the one setting this library measures. Until a measurement
exists the fixed default applies (exact_gpu_jpeg.SMALL_CPU_PIXELS: 200,000 pixels, the GTX 1070 + i7-6700's value).
The measurement decodes three small built-in JPEGs (_route_data.py: generated images, not photographs) a few times
through both routes, about a quarter of a second in all, fits a line to each route's times and takes the crossing,
kept between LOW and HIGH so that no odd timing can send large images to the CPU. It runs when
exact_gpu_jpeg.self_check() is called, or by itself in one short background thread after the caller's first
decode has returned, and only while the caller is not decoding or encoding: if the caller's work overlaps it, that
measurement is thrown away and tried again later, a few times at most.

A machine does not always run at one speed: after some seconds of idleness a GPU can be several times slower for
a moment. Two rules keep such a moment out of the saved value. A measurement whose first and second half give
crossings more than a quarter apart is thrown away (the speed changed while it ran). And a crossing at either end
of the range is not saved as a value at once: a disturbed moment can push a measurement there, and one that lasts
longer than a measurement passes the first rule and gives the same end again a second later. Such an end is used
by the process that measured it and kept in the settings file as unconfirmed, with its time and without a value,
so the next process measures again; the process that got an end measures once more at once, and a value inside
the range then replaces it. An end becomes the saved value only when a measurement in a later process, at least
CONFIRM_S (10 minutes) after its first sighting, lands on the same end: a machine that really belongs there gets
there again. A value inside the range replaces an unconfirmed end; the other end starts the 10 minutes again, and
so does a first sighting dated in the future (the clock was set back). A confirmed end stays, like any saved value,
until the machine, the version or clear_settings() changes it. Accepted: a disturbance that lasts longer than
10 minutes and spans two processes (a long compile, another GPU job, a laptop on battery) can confirm a wrong end;
clear_settings() resets it.

The range is what the measurement can see, not what a machine can be: the built-in files are 0.06 to 0.25 MP,
LOW is just under the smallest and HIGH twice the largest. A machine whose true crossing lies outside gets the
nearest end, and device_report() says so ("at the end of the range (clamped)", with the crossing the lines gave):
the value in use is then a bound of the measurement, not the machine's own (an end not yet confirmed: "an
unconfirmed end of the range"). When measuring is given up (TRIES
measurements thrown away), the default stays in use and device_report() says that and why.

The result is kept in one small settings file (see settings_path()), under a key naming the GPU, its driver, the
CPU, the CPU decoder and this library's version, so a changed machine or version measures again; the crossing as
the lines gave it and the end it was clamped to are kept with it. A missing, unreadable, stale or unwritable file
means the fixed default, silently; deleting the file is harmless.
EGJ_SMALL_CPU_PIXELS, or cpu_below= on a call, pins the value: nothing is measured and no thread is started.

Either route gives the same pixels, so the value only moves time. (The CPU route is not taken in the fast modes
without simplejpeg, where the CPU decoder could not give the mode asked for.)
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import platform
import statistics
import threading
import time

from ._header import env_int

LOW, HIGH = 50_000, 500_000        # the measured value is kept in this range
CPU_ROUTE_ANY = 1 << 40            # the limit handed to the CPU route while timing it: every timing image takes it
TIMING = "timing 3"                # the way the value is measured and kept, in the settings key: a value saved by an
#                                    earlier way is not used, and is measured again (timing 1 timed the CPU decode and
#                                    upload alone; timing 2 confirmed an end of the range within one process)
NO_CPU_ROUTE = "simplejpeg is not installed: the CPU route is not taken, so there is nothing to measure"
ROUNDS = 12                        # times each built-in file goes through each route
TRIES = 3                          # measurements thrown away (the caller was busy, or see AGREE), before giving up
AGREE = 0.25                       # the two halves of a measurement must give crossings this close (a share of the larger)
IDLE_S = 0.05                      # the caller must have been idle this long before a background measurement starts
GIVE_UP_S = 3.0                    # a background attempt waits at most this long for the caller to be idle
CONFIRM_S = 600                    # an end of the range is confirmed by a later process this long after its first sighting
_PROCESS = os.urandom(8).hex()     # this process, in an unconfirmed end's record (a later process has another)
_now = time.time                   # (the clock of that record; tests replace it)
UNCONFIRMED = ", an unconfirmed end of the range (used by this process only; the next process measures again)"

_lock = threading.Lock()
_state = {}                        # device id -> dict(pixels, source)
_info = {}                         # device id -> dict(crossing, clamped, unconfirmed) of the measured value in use
_why = {}                          # device id -> why the last measurement was thrown away
_gave_up = {}                      # device id -> why measuring was given up (the default stays in use)
_threads = {}                      # device id -> the background thread, once started
_tl = threading.local()
activity = 0                       # counts the caller's decodes and encodes (see enter())
_inflight = 0                      # how many of them are running now
_last_activity = 0.0
_stop = False                      # set at interpreter exit
PINNED_BY_ENV = env_int("EGJ_SMALL_CPU_PIXELS", None, 0, 1 << 40) is not None     # (a value that is not a number:
                                                                                  # not pinned, the default measured)


def enter():
    """Called at the start of every decode and encode that is not the measurement's own."""
    global activity, _inflight, _last_activity
    if getattr(_tl, "measuring", False):
        return False
    activity += 1
    _inflight += 1
    _last_activity = time.perf_counter()
    return True


def leave(entered):
    """Called at its end, with what enter() returned."""
    global _inflight, _last_activity
    if entered:
        _inflight -= 1
        _last_activity = time.perf_counter()


def _at_exit():
    global _stop
    _stop = True
    for t in list(_threads.values()):
        t.join(0.5)


atexit.register(_at_exit)


def settings_path():
    """The one settings file: <local application data>/exact-gpu-jpeg/settings.json on Windows, else under
    $XDG_CACHE_HOME or ~/.cache."""
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "exact-gpu-jpeg", "settings.json")


def _cpu_decoder_name():
    try:
        import simplejpeg
        return f"simplejpeg {getattr(simplejpeg, '__version__', '?')}"
    except ImportError:
        pass
    try:
        import PIL
        return f"Pillow {PIL.__version__}"
    except ImportError:
        return "none"


def key(dev):
    from . import __version__, _selfcheck
    return " | ".join([_selfcheck.gpu_name(dev), platform.processor() or platform.machine(), f"{os.cpu_count()} threads",
                       _cpu_decoder_name(), f"exact-gpu-jpeg {__version__}", TIMING])


def _load(dev):
    """The saved value for this machine and version, or None (whatever is wrong with the file; an unconfirmed end
    has no value)."""
    try:
        with open(settings_path(), encoding="utf-8") as f:
            v = json.load(f)["cpu_below"][key(dev)]["pixels"]
        return int(v) if isinstance(v, int) and LOW <= v <= HIGH else None
    except Exception:
        return None


def _load_unconfirmed(dev):
    """The unconfirmed end kept for this machine and version: dict(end, time, process), or None."""
    try:
        with open(settings_path(), encoding="utf-8") as f:
            u = json.load(f)["cpu_below"][key(dev)]["unconfirmed"]
        if u["end"] in (LOW, HIGH) and isinstance(u["time"], (int, float)) and isinstance(u["process"], str):
            return dict(end=u["end"], time=u["time"], process=u["process"])
    except Exception:
        pass
    return None


def _load_info(dev):
    """dict(crossing, clamped) kept with the saved value, or None."""
    try:
        with open(settings_path(), encoding="utf-8") as f:
            e = json.load(f)["cpu_below"][key(dev)]
        c, end = e.get("crossing"), e.get("clamped")
        return dict(crossing=c if isinstance(c, int) else None, clamped=end if end in ("low", "high") else None)
    except Exception:
        return None


def _save(dev, pixels, detail):
    """The value for this machine and version (pixels), or, with pixels None, an unconfirmed end (in detail). On
    Windows the rename fails while another process has the file open: it is tried again for about 1.6 s."""
    tmp = None
    try:
        p = settings_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict) or not isinstance(d.get("cpu_below"), dict):
                d = {}
        except Exception:
            d = {}
        e = dict(pixels=int(pixels)) if pixels is not None else {}
        d.setdefault("cpu_below", {})[key(dev)] = dict(e, measured=time.strftime("%Y-%m-%d %H:%M"), **detail)
        tmp = p + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        for k in range(8):
            try:
                os.replace(tmp, p)
                return True
            except PermissionError:
                if k == 7:
                    raise
                time.sleep(min(0.4, 0.025 * 2 ** k))
    except Exception:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


_generation = 0


def clear():
    """Forget every saved value (delete the settings file) and what this process measured. True if a file was
    removed."""
    global _generation
    with _lock:
        _generation += 1                       # a measurement running now keeps nothing and writes no file
        _state.clear()
        _info.clear()
        _why.clear()
        _gave_up.clear()
        _threads.clear()                       # the next limit() may measure again (a thread still running sees
        try:                                   # the new generation and keeps nothing)
            os.remove(settings_path())
            return True
        except OSError:
            return False


def time_routes(dev, data, rounds):
    """Median ms of one decode of `data` by the CPU route and by the GPU route, taken in turn, and the same two
    medians for the first and for the second half of the rounds: (cpu, gpu, ((cpu, gpu), (cpu, gpu))). The CPU
    route is timed as a decode takes it (_small_on_cpu: the header parse and check, the CPU decode, the output step
    and the upload, waited for). The GPU route is timed as a call that goes to the GPU pays it: the frame-size check
    that sends it there, then Decoder.decode, which returns once the GPU has finished (it reads the result back).
    Tests replace this to pretend to be
    another machine (a replacement may return the first two alone). Returns a reason (a str) instead when no timing
    could be taken: the process is ending, the caller's work overlapped, or a route declined the timing image."""
    import cupy as cp

    from . import _frame_pixels, _small_on_cpu
    from ._decoder import Decoder
    dec = getattr(_tl, "dec", None)
    if dec is None or dec.device.id != dev:
        dec = _tl.dec = Decoder(dev)
    pc = time.perf_counter
    tc, tg = [], []
    with cp.cuda.Device(dev):
        for r in range(rounds + 2):
            if _stop:
                return "the process is ending"
            if _inflight:
                return "the caller's own decodes overlapped every measurement"
            before = activity
            t0 = pc()
            a, _ = _small_on_cpu(data, CPU_ROUTE_ANY, "RGB", dev, "uint8", "HWC", None, None, False, False, True)
            t1 = pc()
            if a is None:
                return "the CPU route declined a timing image"
            _frame_pixels(data)
            g = dec.decode(data)
            t2 = pc()
            if g is None:
                return f"the GPU decode of a timing image was declined ({dec.reason})"
            del a
            if activity != before:
                return "the caller's own decodes overlapped every measurement"
            if r >= 2:                                 # the first two rounds warm both routes up
                tc.append((t1 - t0) * 1e3)
                tg.append((t2 - t1) * 1e3)
    half = len(tc) // 2
    med = statistics.median
    return med(tc), med(tg), ((med(tc[:half]), med(tg[:half])), (med(tc[half:]), med(tg[half:])))


def _fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def crossing(points):
    """points: [(pixels, cpu ms, gpu ms)]. (the pixel count to use, in LOW..HIGH: the CPU route below it, the GPU
    above; the pixel count where the two fitted lines cross, or None when they do not cross that way; the end the
    value was clamped to, "low" or "high", or None when the crossing is inside the range)."""
    xs = [p[0] for p in points]
    ac, bc = _fit(xs, [p[1] for p in points])
    ag, bg = _fit(xs, [p[2] for p in points])
    if bc <= bg:                                       # the CPU does not fall behind as images grow: no crossing
        cpu = sum(ac + bc * x for x in xs)             # of that kind, so the route that is faster over the measured
        gpu = sum(ag + bg * x for x in xs)             # sizes gets the whole range
        raw, x = None, (HIGH if cpu <= gpu else LOW)
    else:
        raw = x = (ag - ac) / (bc - bg)
    px = int(min(HIGH, max(LOW, round(x))))
    end = None
    if raw is None or raw < LOW or raw > HIGH:
        end = "low" if px == LOW else "high"
    return px, (None if raw is None else int(round(raw))), end


def crossover(points):
    """The pixel count to use (crossing()'s first value)."""
    return crossing(points)[0]


def measure(dev, only_inside=False):
    """Measure now, in this thread. Returns the pixel count, or None (nothing is kept then): the caller's own work
    overlapped, the measurement could not be made, or its two halves disagreed. An end of the range is returned and
    used, but saved as unconfirmed until a later process confirms it (see the module text); with only_inside, an end
    is returned and nothing of it is used or kept (the retry after an unconfirmed end)."""
    from . import _route_data, _selfcheck, parse
    if _selfcheck.ensure(dev, "decode") is not None:
        _why[dev] = "the GPU path is off"
        return None
    try:
        import simplejpeg  # noqa: F401
    except ImportError:
        _why[dev] = NO_CPU_ROUTE
        return None
    except Exception as e:                     # a broken install: whatever its load raises, the reason names it
        _why[dev] = f"simplejpeg could not be loaded ({type(e).__name__}): the CPU route is not taken"
        return None
    _tl.measuring = True
    gen = _generation
    try:
        points, halves, detail = [], ([], []), {}
        for name, b64 in _route_data.FILES:
            data = base64.b64decode(b64)
            h = parse(data)
            t = time_routes(dev, data, ROUNDS)
            if t is None or isinstance(t, str):
                _why[dev] = t or "the caller's own decodes overlapped every measurement"
                return None
            n = h.width * h.height
            points.append((n, t[0], t[1]))
            for k, part in enumerate(t[2] if len(t) > 2 else ((t[0], t[1]),) * 2):
                halves[k].append((n, part[0], part[1]))
            detail[name] = [round(t[0], 3), round(t[1], 3)]
        a, b = crossover(halves[0]), crossover(halves[1])
        if abs(a - b) > AGREE * max(a, b):             # the machine changed speed while this ran
            _why[dev] = "the two halves of every measurement disagreed (the machine's speed kept changing)"
            return None
        px, raw, end = crossing(points)
    except Exception as e:
        _why[dev] = f"the measurement failed ({type(e).__name__})"
        return None
    finally:
        _tl.measuring = False
    if only_inside and end is not None:
        return px
    with _lock:
        if gen != _generation:                 # clear_settings() was called meanwhile: this result is not kept
            return None
        rec = None
        if end is not None:                    # an end of the range: confirmed by a later process, CONFIRM_S on
            prev, now = _load_unconfirmed(dev), _now()
            if prev and (prev["end"] != px or prev["time"] > now):
                prev = None                    # the other end, or a first sighting dated in the future: a fresh one
            if not (prev and prev["process"] != _PROCESS and now - prev["time"] >= CONFIRM_S):
                rec = prev or dict(end=px, time=now, process=_PROCESS)
        _state[dev] = dict(pixels=px, source="measured")
        _info[dev] = dict(crossing=raw, clamped=end, unconfirmed=rec is not None)
        _gave_up.pop(dev, None)
        detail = dict(cpu_gpu_ms=detail, crossing=raw, clamped=end)
        if rec is None:                        # (under the lock: a clear_settings() cannot fall between the check
            _save(dev, px, detail)             # and the write)
        else:                                  # kept without a value (the first sighting's time and process)
            _save(dev, None, dict(detail, unconfirmed=rec))
    return px


def _unconfirmed_in_use(dev):
    return bool(_info.get(dev, {}).get("unconfirmed"))


def measure_now(dev):
    """measure(), again while it returns nothing, TRIES times at most: for self_check(), in the caller's thread. After
    an unconfirmed end, once more at once: a value inside the range replaces it (an end again changes nothing)."""
    for _ in range(TRIES):
        px = measure(dev)
        if px is not None or _stop:
            break
    else:
        _gave_up[dev] = _why.get(dev, "no measurement could be made")
        return None
    if px is not None and _unconfirmed_in_use(dev) and not _stop:
        again = measure(dev, only_inside=True)
        if again is not None and again not in (LOW, HIGH):
            px = again
    return px


def wait_background(dev, timeout=10.0):
    """For self_check(): a background measurement running now is waited for, so that the two do not disturb each
    other (the self-check's own decodes would throw it away, and it would slow theirs)."""
    t = _threads.get(dev)
    if t is not None and t.is_alive() and t is not threading.current_thread():
        t.join(timeout)


def _background(dev):
    gen = _generation
    why = _background_tries(dev)
    with _lock:
        if why is not None and gen == _generation:     # (after a clear_settings(), a give-up of before is not kept)
            _gave_up[dev] = why


def _background_tries(dev):
    """The background measurement: None when it ended (measured, or the process is ending), else why it gave up.
    After an unconfirmed end, once more when the caller is next idle: a value inside the range replaces it."""
    retry, failed = False, 0                   # retry: an unconfirmed end is in use, and this is the one retry
    while True:
        t_end = time.perf_counter() + GIVE_UP_S
        while _inflight or time.perf_counter() - _last_activity < IDLE_S:    # wait until the caller is idle
            if _stop:
                return None
            if time.perf_counter() > t_end:
                return None if retry else "the caller was never idle long enough for a measurement"
            time.sleep(0.01)
        if _stop:
            return None
        px = measure(dev, only_inside=retry)
        if retry or (px is not None and not _unconfirmed_in_use(dev)):
            return None                        # (the retry: a value inside the range is kept, else the end stays)
        if px is not None:
            retry = True
            continue
        failed += 1
        if failed >= TRIES:
            return _why.get(dev, "no measurement could be made")
        time.sleep(0.2)                                                # the caller was busy: later


def limit(dev, default, start=True):
    """The pixel count in use on this device: the pinned value, else a measured or saved one, else `default`
    (and, the first time, a background measurement is started unless start is False)."""
    if PINNED_BY_ENV:
        return default
    s = _state.get(dev)
    if s is None:
        with _lock:
            s = _state.get(dev)
            if s is None:
                saved = _load(dev)
                s = _state[dev] = dict(pixels=saved, source="saved") if saved else dict(pixels=None, source="default")
    if s["pixels"] is None:
        if start and dev not in _threads:
            with _lock:
                if dev not in _threads:
                    t = _threads[dev] = threading.Thread(target=_background, args=(dev,), name="exact-gpu-jpeg cpu-route",
                                                         daemon=True)
                    t.start()
        return default
    return s["pixels"]


def describe(dev, default):
    """For device_report(): the value in use and where it came from."""
    if PINNED_BY_ENV:
        return dict(pixels=default, source="pinned by EGJ_SMALL_CPU_PIXELS")
    s = _state.get(dev)
    if s is not None and s.get("pixels"):
        return _described(s["pixels"], s["source"], _info.get(dev) if s["source"] == "measured" else _load_info(dev))
    if s is None:                              # not used yet on this device: the file's value is what will be used
        saved = _load(dev)
        if saved:
            return _described(saved, "saved", _load_info(dev))
    why = _gave_up.get(dev)                    # in use: the default. (A value another process saved since is not
                                               # read again by this one, so it is not reported; the file is read here
                                               # only to say that an unconfirmed end waits in it.)
    if why:
        source = f"default: measuring given up, {why}"
    elif _load_unconfirmed(dev):               # (the default in use, and an end waiting to be confirmed)
        source = "default (not measured yet; the settings file holds an unconfirmed end of the range, measured again)"
    else:
        source = "default (not measured yet)"
    return dict(pixels=default, source=source, settings_file=settings_path())


def _described(pixels, source, info):
    """source: "measured" or "saved"; with ", at the end of the range (clamped)" when the lines crossed outside
    LOW..HIGH (or not at all), or UNCONFIRMED for such an end not yet confirmed by a later process: then `crossing`
    is where they crossed (None: not at all) and `range` the range."""
    d = dict(pixels=pixels, source=source, settings_file=settings_path())
    if info:
        d["crossing"] = info.get("crossing")
        if info.get("unconfirmed"):
            d["source"] = source + UNCONFIRMED
            d["range"] = [LOW, HIGH]
        elif info.get("clamped"):
            d["source"] = source + ", at the end of the range (clamped)"
            d["range"] = [LOW, HIGH]
    return d
