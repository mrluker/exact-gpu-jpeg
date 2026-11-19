"""The size below which one image is decoded on the CPU (cpu_below): measured once per machine, saved, clamped,
pinnable, and never changing a byte. The timing source and the settings file are replaced here, so the suite can
pretend to be a machine whose GPU or CPU is ten times faster."""
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pytest
from PIL import Image

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from conftest import reference  # noqa: E402
from exact_gpu_jpeg import _route as R  # noqa: E402

DEV = cp.cuda.Device().id


def _jpeg(w, h, seed=0):
    a = np.random.default_rng(seed).integers(0, 256, (max(h // 8, 2), max(w // 8, 2), 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90)
    return b.getvalue()


@pytest.fixture
def unpinned(monkeypatch, tmp_path):
    """The library as a user has it: no pin, its own (temporary) settings file, nothing measured yet."""
    monkeypatch.setattr(R, "PINNED_BY_ENV", False)
    monkeypatch.setattr(R, "settings_path", lambda: str(tmp_path / "exact-gpu-jpeg" / "settings.json"))
    monkeypatch.setattr(R, "IDLE_S", 0.01)
    monkeypatch.setattr(R, "GIVE_UP_S", 1.0)
    saved = dict(R._state), dict(R._threads)
    for d in (R._state, R._threads, R._gave_up, R._why, R._info):
        d.clear()
    yield tmp_path / "exact-gpu-jpeg" / "settings.json"
    for t in list(R._threads.values()):
        t.join(5)
    for d in (R._state, R._threads, R._gave_up, R._why, R._info):
        d.clear()
    R._state.update(saved[0])
    R._threads.update(saved[1])


def _machine(monkeypatch, cpu, gpu):
    """A machine whose two routes take cpu(pixels) and gpu(pixels) ms."""
    px = {}

    def fake(dev, data, rounds):
        h = egj.inspect(data)
        n = h["width"] * h["height"]
        px[n] = True
        return cpu(n), gpu(n)
    monkeypatch.setattr(R, "time_routes", fake)


def _wait():
    for t in list(R._threads.values()):
        t.join(5)
        assert not t.is_alive()


def test_the_crossing_and_its_clamp():
    line = lambda a, b: (lambda n: a + b * n)                                      # noqa: E731
    pts = lambda c, g: [(n, c(n), g(n)) for n in (60_800, 118_720, 251_328)]       # noqa: E731
    assert R.crossover(pts(line(0.2, 5e-6), line(1.0, 1e-6))) == 200_000           # 0.8 ms / 4e-6 ms per pixel
    assert R.crossover(pts(line(0.2, 5e-6), line(0.1, 1e-7))) == R.LOW             # the GPU wins everywhere
    assert R.crossover(pts(line(0.02, 5e-7), line(1.0, 1e-6))) == R.HIGH           # the CPU wins everywhere
    assert R.crossover(pts(line(0.2, 5e-5), line(1.0, 1e-5))) == R.LOW             # crossing below the range
    assert R.crossover(pts(line(0.2, 1e-6), line(3.0, 1e-7))) == R.HIGH            # crossing above the range
    assert R.crossover(pts(line(0.2, 1e-7), line(0.1, 5e-6))) == R.HIGH            # no such crossing; the CPU wins at
    assert R.crossover(pts(line(5.0, 1e-7), line(0.1, 5e-6))) == R.LOW             # the measured sizes / the GPU does
    assert R.crossing(pts(line(0.2, 5e-6), line(1.0, 1e-6))) == (200_000, 200_000, None)
    assert R.crossing(pts(line(0.2, 1e-6), line(3.0, 1e-7))) == (R.HIGH, 3_111_111, "high")
    assert R.LOW == 50_000 and R.HIGH == 500_000


@pytest.mark.parametrize("cpu, gpu, want", [
    (lambda n: 0.2 + 5e-6 * n, lambda n: 1.0 + 1e-6 * n, 200_000),                 # this machine, roughly
    (lambda n: 0.2 + 5e-6 * n, lambda n: 0.10 + 1e-7 * n, 50_000),                 # a GPU ten times faster
    (lambda n: 0.02 + 5e-7 * n, lambda n: 1.0 + 1e-6 * n, 500_000),                # a CPU ten times faster
], ids=["like this machine", "GPU ten times faster", "CPU ten times faster"])
def test_measured_saved_and_used(unpinned, monkeypatch, cpu, gpu, want):
    clock = [1_000_000.0]
    monkeypatch.setattr(R, "_now", lambda: clock[0])
    _machine(monkeypatch, cpu, gpu)
    small, mid = _jpeg(240, 160), _jpeg(480, 300, 1)                               # 38,400 and 144,000 pixels
    info = {}
    egj.decode(small, info=info)                                                   # the first decode: the default
    assert info["path"] == "cpu"
    _wait()                                                                        # the background measurement
    assert R._state[DEV] == dict(pixels=want, source="measured")
    d = json.loads(unpinned.read_text())
    (k, v), = d["cpu_below"].items()
    assert "exact-gpu-jpeg" in k and str(os.cpu_count()) in k
    r = egj.device_report()["cpu_below"]
    end = "" if want == 200_000 else ", at the end of the range (clamped)"
    if end:                                        # an end: used here, kept once a later process 10 minutes on agrees
        assert "pixels" not in v and v["unconfirmed"]["end"] == want
        assert r["pixels"] == want and r["source"] == "measured" + R.UNCONFIRMED
    else:
        assert v["pixels"] == want and r["pixels"] == want and r["source"] == "measured"
    assert r["settings_file"] == str(unpinned)
    info = {}
    img = egj.decode(mid, info=info)
    assert info["path"] == ("cpu" if 144_000 <= want else "gpu")
    assert np.array_equal(img.get(), egj.decode_cpu(mid))
    if end:
        _new_process(monkeypatch)
        clock[0] += R.CONFIRM_S
        assert R.measure_now(DEV) == want and _entry(unpinned)["pixels"] == want
    _new_process(monkeypatch)                                                      # a new process: the saved value
    assert R.limit(DEV, egj.SMALL_CPU_PIXELS) == want and not R._threads
    assert egj.device_report()["cpu_below"]["source"] == "saved" + end
    assert egj.clear_settings() is True and not os.path.exists(unpinned)
    assert egj.clear_settings() is False
    assert R.limit(DEV, egj.SMALL_CPU_PIXELS, start=False) == egj.SMALL_CPU_PIXELS


def test_the_report_says_measured_clamped_or_given_up(unpinned, monkeypatch):
    """device_report()["cpu_below"]["source"] never says more than is known: "measured" for a crossing inside the
    range; "..., at the end of the range (clamped)" with the crossing the lines gave when it lies outside (the
    value in use is then the measurement's bound, not the machine's own); "default: measuring given up, <why>" when
    every measurement was thrown away; "..., an unconfirmed end of the range (...)" for an end until a later process
    confirms it. The crossing and the clamp are kept in the settings file."""
    clock = [1_000_000.0]
    monkeypatch.setattr(R, "_now", lambda: clock[0])
    line = lambda a, b: (lambda n: a + b * n)                                      # noqa: E731
    # 1. inside the range
    _machine(monkeypatch, line(0.2, 5e-6), line(1.0, 1e-6))
    assert R.measure_now(DEV) == 200_000
    r = egj.device_report()["cpu_below"]
    assert r["source"] == "measured" and r["crossing"] == 200_000 and "range" not in r
    e, = json.loads(unpinned.read_text())["cpu_below"].values()
    assert e["pixels"] == 200_000 and e["crossing"] == 200_000 and e["clamped"] is None
    assert egj.clear_settings() is True
    # 2. the lines cross above the range: a slow GPU beside a fast CPU (the true crossing would be 1.4 MP)
    _machine(monkeypatch, line(0.2, 1e-6), line(1.46, 1e-7))
    assert R.measure_now(DEV) == R.HIGH                                            # used here, not yet saved
    r = egj.device_report()["cpu_below"]
    assert r["pixels"] == R.HIGH and r["source"] == "measured" + R.UNCONFIRMED
    assert r["crossing"] == 1_400_000 and r["range"] == [R.LOW, R.HIGH]
    assert "pixels" not in _entry(unpinned)
    _new_process(monkeypatch)                                                      # a later process, 10 minutes on
    clock[0] += R.CONFIRM_S
    assert R.measure_now(DEV) == R.HIGH
    r = egj.device_report()["cpu_below"]
    assert r["pixels"] == R.HIGH and r["source"] == "measured, at the end of the range (clamped)"
    assert r["crossing"] == 1_400_000 and r["range"] == [R.LOW, R.HIGH]
    e, = json.loads(unpinned.read_text())["cpu_below"].values()
    assert e["pixels"] == R.HIGH and e["crossing"] == 1_400_000 and e["clamped"] == "high"
    _new_process(monkeypatch)                                                      # a new process reads it back
    assert R.limit(DEV, egj.SMALL_CPU_PIXELS, start=False) == R.HIGH
    r = egj.device_report()["cpu_below"]
    assert r["source"] == "saved, at the end of the range (clamped)" and r["crossing"] == 1_400_000
    assert egj.clear_settings() is True
    # ... and below it, and lines that do not cross at all (the GPU wins everywhere): the lower end, no crossing
    _machine(monkeypatch, line(0.2, 5e-5), line(1.0, 1e-5))
    assert R.measure_now(DEV) == R.LOW
    r = egj.device_report()["cpu_below"]
    assert r["source"].endswith(R.UNCONFIRMED) and r["crossing"] == 20_000
    assert egj.clear_settings() is True
    _machine(monkeypatch, line(5.0, 1e-7), line(0.1, 5e-6))                        # no crossing: the GPU wins
    assert R.measure_now(DEV) == R.LOW
    r = egj.device_report()["cpu_below"]
    assert r["source"].endswith(R.UNCONFIRMED) and r["crossing"] is None
    assert egj.clear_settings() is True
    _machine(monkeypatch, line(0.2, 1e-7), line(0.1, 5e-6))                        # no crossing of that kind, and the
    assert R.measure_now(DEV) == R.HIGH                                            # CPU wins at every measured size
    r = egj.device_report()["cpu_below"]                                           # (the lines meet at 20,000 pixels,
    assert r["source"].endswith(R.UNCONFIRMED) and r["crossing"] is None           # the GPU faster below: not above)
    assert egj.clear_settings() is True
    # 3. given up: the two halves of every measurement disagree
    seen = _recorded(monkeypatch, [CHANGED])
    assert R.measure_now(DEV) is None and len(seen) == R.TRIES
    r = egj.device_report()["cpu_below"]
    assert r["pixels"] == egj.SMALL_CPU_PIXELS and r["source"].startswith("default: measuring given up, the two halves")
    assert "crossing" not in r and not os.path.exists(unpinned)
    seen = _recorded(monkeypatch, [NORMAL])                                        # a later steady one: measured
    assert R.measure_now(DEV) == pytest.approx(225_000, rel=0.03)
    assert egj.device_report()["cpu_below"]["source"] == "measured"
    assert egj.clear_settings() is True
    assert egj.device_report()["cpu_below"]["source"] == "default (not measured yet)"


def test_a_bad_settings_file_means_the_default_silently(unpinned, monkeypatch):
    _machine(monkeypatch, lambda n: 0.2 + 5e-6 * n, lambda n: 1.0 + 1e-6 * n)
    os.makedirs(os.path.dirname(unpinned))
    good = {"cpu_below": {R.key(DEV): {"pixels": 123_456}}}
    cases = ["not json {", json.dumps([1, 2]), json.dumps({"cpu_below": 5}),
             json.dumps({"cpu_below": {R.key(DEV): {"pixels": 10}}}),              # below the range
             json.dumps({"cpu_below": {R.key(DEV): {"pixels": "200000"}}}),        # not a number
             json.dumps({"cpu_below": {"another GPU | another CPU": {"pixels": 123_456}}})]     # stale: another machine
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for text in cases:
            unpinned.write_text(text)
            R._state.clear()
            assert R.limit(DEV, 200_000, start=False) == 200_000, text
        unpinned.write_text(json.dumps(good))
        R._state.clear()
        assert R.limit(DEV, 200_000, start=False) == 123_456
        monkeypatch.setattr(R, "key", lambda dev: "this GPU | this CPU | a newer library version")
        R._state.clear()
        assert R.limit(DEV, 200_000, start=False) == 200_000                       # stale: another version
        # a settings file that cannot be written: measured and used in this process, nothing raised
        monkeypatch.setattr(R, "settings_path", lambda: str(unpinned) + os.sep + "under-a-file" + os.sep + "settings.json")
        R._state.clear()
        assert R.measure(DEV) == 200_000 and R.limit(DEV, 1, start=False) == 200_000


def test_pins_measure_nothing_and_start_no_thread(unpinned, monkeypatch):
    called = []
    monkeypatch.setattr(R, "time_routes", lambda *a: called.append(1) or (1.0, 1.0))
    data = _jpeg(240, 160)
    info = {}
    egj.decode(data, cpu_below=10_000, info=info)                                  # the keyword
    assert info["path"] == "gpu" and not R._threads
    egj.decode(data, cpu_below=0)
    egj.decode([data, data])                                                       # a batch never takes the CPU route
    assert not R._threads
    monkeypatch.setattr(egj, "SMALL_CPU_PIXELS", 30_000)                           # the program's own value
    egj.decode(data, info=info)
    assert info["path"] == "gpu" and not R._threads
    assert egj.device_report()["cpu_below"] == dict(pixels=30_000, source="pinned by the program (SMALL_CPU_PIXELS)")
    monkeypatch.setattr(egj, "SMALL_CPU_PIXELS", egj._SMALL_FIXED)
    monkeypatch.setattr(R, "PINNED_BY_ENV", True)                                  # EGJ_SMALL_CPU_PIXELS
    egj.decode(data, info=info)
    assert info["path"] == "cpu" and not R._threads and not called
    assert egj.device_report()["cpu_below"]["source"] == "pinned by EGJ_SMALL_CPU_PIXELS"
    assert egj.self_check()["ok"] and not called                                   # the warm-up call does not measure either


def test_a_measurement_the_caller_overlaps_is_thrown_away(unpinned, monkeypatch):
    tries = []
    monkeypatch.setattr(R, "time_routes", lambda *a: tries.append(1))              # None: the caller's work overlapped
    egj.decode(_jpeg(240, 160))
    _wait()
    assert len(tries) == R.TRIES and R._state[DEV] == dict(pixels=None, source="default")
    assert not os.path.exists(unpinned)
    info = {}
    egj.decode(_jpeg(480, 300), info=info)                                         # the default stays, no new thread
    assert info["path"] == "cpu" and len(tries) == R.TRIES


# Two measurements recorded on the GTX 1070 + i7-6700 after five seconds of idleness (ms, CPU route and GPU route,
# for the three built-in sizes). In the first the GPU slowed down halfway; in the second it was slow throughout.
SIZES = (60_800, 118_720, 251_328)
NORMAL = ((0.397, 0.913), (0.579, 0.912), (1.110, 1.030))                          # crossing: about 225,000 pixels
CHANGED = dict(whole=((0.440, 0.936), (0.692, 0.936), (1.373, 1.339)),             # looks fine: about 232,000
               first=((0.513, 0.975), (0.790, 0.942), (1.470, 1.117)),             # but its first half says 164,000
               second=((0.438, 0.927), (0.620, 0.936), (1.313, 1.650)))            # and its second half 500,000
SLOW = ((0.513, 2.033), (0.792, 2.175), (1.669, 2.820))                            # 500,000: the upper end


def _recorded(monkeypatch, measurements):
    """time_routes replaced by recorded measurements, used in turn (the last one again and again)."""
    seen = []

    def fake(dev, data, rounds):
        h = egj.inspect(data)
        k = SIZES.index(h["width"] * h["height"])
        if k == 0:
            seen.append(measurements[min(len(seen), len(measurements) - 1)])
        m = seen[-1]
        if isinstance(m, dict):
            return m["whole"][k] + ((m["first"][k], m["second"][k]),)
        return m[k] + ((m[k], m[k]),)
    monkeypatch.setattr(R, "time_routes", fake)
    return seen


def test_a_measurement_whose_two_halves_disagree_is_thrown_away(unpinned, monkeypatch):
    seen = _recorded(monkeypatch, [CHANGED])
    assert R.crossover([(n,) + m for n, m in zip(SIZES, CHANGED["whole"])]) == pytest.approx(232_000, rel=0.02)
    assert R.measure(DEV) is None                                                  # the medians alone look fine
    assert DEV not in R._state and not os.path.exists(unpinned)
    egj.decode(_jpeg(240, 160))                                                    # in the background: tried, given up
    _wait()
    assert len(seen) == 1 + R.TRIES and R._state[DEV] == dict(pixels=None, source="default")
    assert not os.path.exists(unpinned)
    assert R.limit(DEV, egj.SMALL_CPU_PIXELS, start=False) == egj.SMALL_CPU_PIXELS
    seen = _recorded(monkeypatch, [CHANGED, NORMAL])                               # the next one is steady: kept
    assert R.measure_now(DEV) == pytest.approx(225_000, rel=0.03) and len(seen) == 2
    assert json.loads(unpinned.read_text())["cpu_below"]


def test_an_end_of_the_range_is_used_at_once_and_saved_only_when_a_later_process_confirms_it(unpinned, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(R, "_now", lambda: clock[0])
    seen = _recorded(monkeypatch, [SLOW, NORMAL])                                  # one slow moment, then as usual
    assert R.measure(DEV) == R.HIGH and R._state[DEV] == dict(pixels=R.HIGH, source="measured")
    assert "pixels" not in _entry(unpinned)                                        # (used here, not saved)
    px = R.measure(DEV)
    assert px == pytest.approx(225_000, rel=0.03) and R._state[DEV] == dict(pixels=px, source="measured")
    e = _entry(unpinned)
    assert e["pixels"] == px and "unconfirmed" not in e and len(seen) == 2
    assert egj.clear_settings() is True
    seen = _recorded(monkeypatch, [SLOW])                                          # a machine that is always there
    assert R.measure_now(DEV) == R.HIGH and len(seen) == 2                         # (the end, and one retry)
    assert "pixels" not in _entry(unpinned)
    _new_process(monkeypatch)
    clock[0] += R.CONFIRM_S
    assert R.measure_now(DEV) == R.HIGH and _entry(unpinned)["pixels"] == R.HIGH
    assert egj.clear_settings() is True
    seen = _recorded(monkeypatch, [SLOW, NORMAL])                                  # the warm-up call: one slow moment,
    c = egj.self_check()["cpu_below"]                                              # then as usual: the retry's value
    assert c["source"] == "measured" and c["pixels"] == px and len(seen) == 2
    e = _entry(unpinned)
    assert e["pixels"] == px and "unconfirmed" not in e
    assert egj.clear_settings() is True
    seen = _recorded(monkeypatch, [SLOW, FINAL9])                                  # the retry gives the other end:
    c = egj.self_check()["cpu_below"]                                              # nothing changes
    assert c["pixels"] == R.HIGH and c["source"] == "measured" + R.UNCONFIRMED and len(seen) == 2
    assert _entry(unpinned)["unconfirmed"]["end"] == R.HIGH


def test_the_real_measurement_stops_when_the_caller_decodes(unpinned):
    """The real timing loop: a decode on another thread during it makes it return None, and nothing is kept."""
    import threading
    stop = threading.Event()
    data = _jpeg(240, 160)
    dec = egj.Decoder()
    dec.decode(data)

    def caller():
        while not stop.is_set():
            dec.decode(data)
    t = threading.Thread(target=caller)
    t.start()
    try:
        time.sleep(0.05)
        assert R.measure(DEV) is None
    finally:
        stop.set()
        t.join()
    assert DEV not in R._state or R._state[DEV]["pixels"] is None
    assert not os.path.exists(unpinned)


def test_the_warm_up_call_measures_on_this_machine(unpinned):
    r = egj.self_check()
    assert r["ok"]
    for _ in range(5):                             # (a loaded machine can spoil a measurement: it is then repeated)
        if r["cpu_below"]["source"].startswith("measured"):
            break
        R.measure_now(DEV)
        r = egj.self_check()
    c = r["cpu_below"]
    assert c["source"].startswith("measured") and R.LOW <= c["pixels"] <= R.HIGH, c
    if c["source"] != "measured":                  # a slow moment (guard bands, a loaded machine): an end of the range,
        assert c["source"].endswith(R.UNCONFIRMED) and c["pixels"] in (R.LOW, R.HIGH) and c["range"] == [R.LOW, R.HIGH], c
    print("measured on this machine:", c, json.loads(unpinned.read_text()))
    assert not R._threads                                                          # measured in the calling thread


def test_every_corpus_file_at_the_lowest_and_the_highest_value(corpus):
    """Whatever a measurement gives, the bytes are the reference's: every file under both ends of the range."""
    n = routes = 0
    for name, data, expected in corpus:
        ref = reference(data)
        if ref is None:
            continue
        try:
            egj.decode_cpu(data)
        except Exception:                                  # libjpeg-turbo warns about it (strict, the default):
            for cb in (0, 10 ** 9):                        # both routes raise (tests/test_warnings_rule.py)
                with pytest.raises(Exception):
                    egj.decode(data, cpu_below=cb)
            continue
        seen = set()
        for lim in (R.LOW, R.HIGH):
            info = {}
            img = egj.decode(data, cpu_below=lim, info=info)
            assert np.array_equal(img.get(), ref), (name, lim)
            seen.add(info["path"])
        n += 1
        routes += len(seen) == 2
    assert n > 300 and routes >= 1               # some files change route between the two ends; none changes a byte


@pytest.mark.parametrize("mode", ["fast_dct", "fast_upsample"])
def test_fast_modes_without_simplejpeg_stay_on_the_gpu(mode, monkeypatch):
    """The CPU route could not give the fast mode without simplejpeg (Pillow has no such switch), so it is not
    taken: images on both sides of the limit come from the GPU, in the mode asked for."""
    simplejpeg = pytest.importorskip("simplejpeg")
    kw = {mode: True}
    files = [_jpeg(w, h, k) for k, (w, h) in enumerate([(160, 120), (320, 240), (480, 320), (640, 480), (800, 600)])]
    refs = [simplejpeg.decode_jpeg(d, colorspace="RGB", fastdct=mode == "fast_dct", fastupsample=mode == "fast_upsample")
            for d in files]
    for d, r in zip(files, refs):                                                  # with simplejpeg: either route, same bytes
        info = {}
        assert np.array_equal(egj.decode(d, info=info, **kw).get(), r)
    monkeypatch.setitem(sys.modules, "simplejpeg", None)                           # "import simplejpeg" now fails
    for lim in (None, R.LOW, R.HIGH):
        for d, r in zip(files, refs):
            info = {}
            img = egj.decode(d, cpu_below=lim, info=info, **kw)
            assert info["path"] == "gpu", (mode, lim, info)
            assert np.array_equal(img.get(), r), (mode, lim)


def test_the_measurement_times_the_routes_a_decode_takes(unpinned, monkeypatch):
    """The CPU half of the measurement goes through the CPU route a decode takes (_small_on_cpu: the header parse,
    the libjpeg-turbo header check, the decode, the output step and the upload): a measurement that timed less than
    the call (the decode and upload alone) set the limit 22 % too high on the GTX 1070 + i7-6700 (2026-11-18)."""
    import exact_gpu_jpeg as top
    calls = []
    real = top._small_on_cpu

    def spy(src, limit, *a, **k):
        calls.append(limit)
        return real(src, limit, *a, **k)
    monkeypatch.setattr(top, "_small_on_cpu", spy)
    data = _jpeg(240, 160)
    R._tl.measuring = True
    try:
        t = R.time_routes(DEV, data, 3)
    finally:
        R._tl.measuring = False
    assert not isinstance(t, str) and t is not None, t
    assert len(calls) == 3 + 2, calls                    # every round, the two warm-up rounds included
    assert all(c >= 240 * 160 for c in calls)            # a limit that sends this image to the CPU route


def test_the_measurement_says_why_when_the_cpu_route_declines(unpinned, monkeypatch):
    """A timing image the CPU route would not take (here: it declines every file) gives a reason, not a timing."""
    import exact_gpu_jpeg as top
    monkeypatch.setattr(top, "_small_on_cpu", lambda src, *a, **k: (None, src))
    R._tl.measuring = True
    try:
        t = R.time_routes(DEV, _jpeg(240, 160), 3)
    finally:
        R._tl.measuring = False
    assert isinstance(t, str) and "CPU route" in t, t


def test_a_value_saved_by_the_earlier_timing_is_not_used(unpinned, monkeypatch):
    """A value saved under the key without the timing tag (measured before the routes were timed as a decode takes
    them, and too high), or under "timing 2" (an end confirmed within one process: the clean install of 2026-11-19
    saved 50,000 so), is not used: the default applies until this machine is measured again."""
    os.makedirs(os.path.dirname(unpinned))
    old = R.key(DEV).rsplit(" | ", 1)[0]
    assert old + " | " + R.TIMING == R.key(DEV) and R.TIMING == "timing 3"
    unpinned.write_text(json.dumps({"cpu_below": {old: {"pixels": 295_434}}}))
    R._state.clear()
    assert R.limit(DEV, 200_000, start=False) == 200_000
    unpinned.write_text(json.dumps({"cpu_below": {old + " | timing 2": {"pixels": 50_000, "crossing": 2_462,
                                                                         "clamped": "low"}}}))
    R._state.clear()
    assert R.limit(DEV, 200_000, start=False) == 200_000
    unpinned.write_text(json.dumps({"cpu_below": {R.key(DEV): {"pixels": 158_041}}}))
    R._state.clear()
    assert R.limit(DEV, 200_000, start=False) == 158_041


def test_without_simplejpeg_nothing_is_measured_and_the_report_says_why(unpinned, monkeypatch):
    """Without simplejpeg the CPU route is never taken: no timing is made and the reason names it."""
    monkeypatch.setitem(sys.modules, "simplejpeg", None)          # import simplejpeg raises ImportError
    assert R.measure_now(DEV) is None
    src = egj.device_report()["cpu_below"]["source"]
    assert "simplejpeg is not installed" in src, src


def test_a_simplejpeg_that_fails_to_load_is_a_reason_not_an_error(unpinned, monkeypatch):
    """A broken install whose import raises something other than ImportError: measuring gives up with a reason that
    names it, and nothing reaches the caller (self_check, or the background thread)."""
    import builtins
    real = builtins.__import__

    def broken(name, *a, **k):
        if name == "simplejpeg":
            raise OSError("a DLL failed to load")
        return real(name, *a, **k)
    monkeypatch.delitem(sys.modules, "simplejpeg", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken)
    assert R.measure_now(DEV) is None
    monkeypatch.setattr(builtins, "__import__", real)
    src = egj.device_report()["cpu_below"]["source"]
    assert "simplejpeg could not be loaded (OSError)" in src, src


# The clean install of 2026-11-19 on fba64922, second route (sdist): a new user's first self_check() measured while
# something disturbed the machine (the CPU route about 1 ms slower at every size, the GPU route hardly): the lines
# cross below the range, so the lower end, in two measurements a second apart. That value was saved for good.
FINAL9 = ((1.721, 1.326), (1.816, 1.239), (2.873, 1.423))


def _new_process(monkeypatch):
    """What a new process starts with: nothing measured or read yet, and its own process token."""
    for d in (R._state, R._threads, R._gave_up, R._why, R._info):
        d.clear()
    getattr(R, "_unconfirmed", {}).clear()
    monkeypatch.setattr(R, "_PROCESS", os.urandom(8).hex(), raising=False)


def _entry(path):
    e, = json.loads(path.read_text())["cpu_below"].values()
    return e


def test_an_end_measured_twice_within_seconds_is_not_saved_as_confirmed(unpinned, monkeypatch):
    """An end of the range measured by one process (however many times) is used by that process only: the settings
    file keeps it as unconfirmed, without a value, and the next process measures again."""
    _recorded(monkeypatch, [FINAL9])
    c = egj.self_check()["cpu_below"]
    assert c["pixels"] == R.LOW and "unconfirmed end" in c["source"], c
    e = _entry(unpinned)
    assert "pixels" not in e and e["unconfirmed"]["end"] == R.LOW, e
    _new_process(monkeypatch)
    assert "unconfirmed end" in egj.device_report()["cpu_below"]["source"]
    assert R.limit(DEV, egj.SMALL_CPU_PIXELS, start=False) == egj.SMALL_CPU_PIXELS


def test_an_end_is_confirmed_only_by_a_later_process_ten_minutes_on(unpinned, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(R, "_now", lambda: clock[0], raising=False)
    _recorded(monkeypatch, [FINAL9])
    assert R.measure_now(DEV) == R.LOW                                  # process A: used here, unconfirmed
    first = _entry(unpinned)["unconfirmed"]
    clock[0] += 11 * 60                                                 # 11 minutes on, but the same process A
    assert R.measure_now(DEV) == R.LOW and _entry(unpinned)["unconfirmed"] == first
    clock[0] -= 2 * 60
    _new_process(monkeypatch)                                           # process B, 9 minutes on: not yet
    assert R.measure_now(DEV) == R.LOW
    e = _entry(unpinned)
    assert "pixels" not in e and e["unconfirmed"] == first, e          # (the first sighting's time and process kept)
    clock[0] += 60                                                      # process B again, 10 minutes on: B is a
    assert R.measure_now(DEV) == R.LOW                                  # later process than A, so it confirms,
    e = _entry(unpinned)                                                # though it measured before
    assert e["pixels"] == R.LOW and "unconfirmed" not in e, e
    assert egj.clear_settings() is True
    clock[0] = 1_000_000.0
    _recorded(monkeypatch, [FINAL9])
    assert R.measure_now(DEV) == R.LOW                                  # A again, then C 10 minutes on: confirmed
    _new_process(monkeypatch)
    clock[0] += R.CONFIRM_S
    assert R.measure_now(DEV) == R.LOW
    e = _entry(unpinned)
    assert e["pixels"] == R.LOW and e["clamped"] == "low" and "unconfirmed" not in e, e
    _new_process(monkeypatch)                                           # process D reads it and measures nothing
    assert R.limit(DEV, egj.SMALL_CPU_PIXELS, start=False) == R.LOW and not R._threads
    assert egj.device_report()["cpu_below"]["source"] == "saved, at the end of the range (clamped)"


def test_after_an_unconfirmed_end_a_steady_value_is_saved_and_the_other_end_starts_again(unpinned, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(R, "_now", lambda: clock[0], raising=False)
    _recorded(monkeypatch, [FINAL9])
    assert R.measure_now(DEV) == R.LOW
    _new_process(monkeypatch)
    _recorded(monkeypatch, [NORMAL])                                    # a steady value: saved as before
    px = R.measure_now(DEV)
    assert px == pytest.approx(225_000, rel=0.03)
    e = _entry(unpinned)
    assert e["pixels"] == px and "unconfirmed" not in e
    _new_process(monkeypatch)
    _recorded(monkeypatch, [SLOW])                                      # the upper end: unconfirmed
    assert R.measure_now(DEV) == R.HIGH and _entry(unpinned)["unconfirmed"]["end"] == R.HIGH
    _new_process(monkeypatch)
    clock[0] += 600
    _recorded(monkeypatch, [FINAL9])                                    # the other end: it starts again, from now
    assert R.measure_now(DEV) == R.LOW
    e = _entry(unpinned)
    assert e["unconfirmed"] == dict(end=R.LOW, time=clock[0], process=R._PROCESS) and "pixels" not in e, e
    _new_process(monkeypatch)
    clock[0] += 600
    assert R.measure_now(DEV) == R.LOW and _entry(unpinned)["pixels"] == R.LOW


def test_a_first_sighting_dated_in_the_future_is_a_fresh_one(unpinned, monkeypatch):
    """The clock set back after an end was first seen: that record would block confirmation until the clock caught
    up, so it counts as a new first sighting, dated now."""
    clock = [1_000_000.0]
    monkeypatch.setattr(R, "_now", lambda: clock[0])
    _recorded(monkeypatch, [FINAL9])
    assert R.measure_now(DEV) == R.LOW
    _new_process(monkeypatch)
    clock[0] -= 3600                                                    # an hour back
    assert R.measure_now(DEV) == R.LOW
    e = _entry(unpinned)
    assert "pixels" not in e and e["unconfirmed"] == dict(end=R.LOW, time=clock[0], process=R._PROCESS), e
    _new_process(monkeypatch)
    clock[0] += R.CONFIRM_S                                             # 10 minutes after the new first sighting
    assert R.measure_now(DEV) == R.LOW and _entry(unpinned)["pixels"] == R.LOW


def test_the_background_measurement_retries_once_after_an_end(unpinned, monkeypatch):
    """The first decode's background measurement: one slow moment gives an end, the retry when the caller is next
    idle gives the steady value, which is used and saved."""
    seen = _recorded(monkeypatch, [SLOW, NORMAL])
    egj.decode(_jpeg(240, 160))
    _wait()
    assert len(seen) == 2 and R._state[DEV]["pixels"] == pytest.approx(225_000, rel=0.03)
    e = _entry(unpinned)
    assert e["pixels"] == R._state[DEV]["pixels"] and "unconfirmed" not in e


def test_a_save_waits_for_a_reader_and_leaves_no_temporary_file(unpinned, monkeypatch):
    """On Windows os.replace fails while another process has settings.json open: the save is tried again for a
    moment; if it never succeeds, it is given up and no .tmp file is left beside the settings file."""
    real, fails = os.replace, [2]

    def busy(a, b):
        if fails[0]:
            fails[0] -= 1
            raise PermissionError(13, "the file is open in another process")
        return real(a, b)
    monkeypatch.setattr(R.os, "replace", busy)
    assert R._save(DEV, 180_000, {}) is True and _entry(unpinned)["pixels"] == 180_000
    fails[0] = 99
    assert R._save(DEV, 190_000, {}) is False and _entry(unpinned)["pixels"] == 180_000
    assert os.listdir(os.path.dirname(unpinned)) == ["settings.json"]


def test_self_check_waits_for_a_background_measurement(unpinned, monkeypatch):
    """self_check() and a background measurement running at the same time would disturb each other: self_check()
    lets it finish, and then does not measure again."""
    seen = _recorded(monkeypatch, [NORMAL])
    timed = R.time_routes

    def slow(dev, data, rounds):
        time.sleep(0.05)
        return timed(dev, data, rounds)
    monkeypatch.setattr(R, "time_routes", slow)
    egj.decode(_jpeg(240, 160))                                         # starts the background measurement
    c = egj.self_check()["cpu_below"]
    assert c["source"] == "measured" and len(seen) == 1 and not any(t.is_alive() for t in R._threads.values())
