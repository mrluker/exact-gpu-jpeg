"""The small-image CPU route's report must name the value in use (audit of 2026-11-18, findings 7.6 and 7.11).
Each test fails on a54cd05 and passes with the fix."""
import json
import os

import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _route as R  # noqa: E402
from test_route import unpinned, _wait, _jpeg  # noqa: E402,F401

DEV = cp.cuda.Device().id


def test_a_value_another_process_saves_later_is_not_reported_as_in_use(unpinned, monkeypatch):
    monkeypatch.setattr(R, "time_routes", lambda *a: "the caller's own decodes overlapped every measurement")
    egj.decode(_jpeg(240, 160))                          # this process: the default, then measuring is given up
    _wait()
    assert R.limit(DEV, egj._SMALL_FIXED, start=False) == egj._SMALL_FIXED
    os.makedirs(os.path.dirname(unpinned), exist_ok=True)
    R._save(DEV, 137_000, dict(crossing=137_000, clamped=None))      # another process measures and saves
    d = egj.device_report()["cpu_below"]
    assert d["pixels"] == egj._SMALL_FIXED                # what this process uses (it does not read the file again)
    assert d["source"].startswith("default: measuring given up")


def test_self_check_reports_a_value_the_program_pinned(unpinned, monkeypatch):
    monkeypatch.setattr(R, "time_routes", lambda *a: (1.0, 1.0))
    monkeypatch.setattr(egj, "SMALL_CPU_PIXELS", 30_000)
    assert egj.self_check()["cpu_below"] == dict(pixels=30_000, source="pinned by the program (SMALL_CPU_PIXELS)")


def test_the_reasons_for_giving_up_are_kept_apart(unpinned, monkeypatch):
    """The real time_routes, with a timing decoder that declines: the report names the decline (it said the
    caller's decodes overlapped)."""
    class Declines:
        device = cp.cuda.Device(DEV)
        reason = "declined in this test"

        def decode(self, data, *a, **k):
            return None
    monkeypatch.setattr(R._tl, "dec", Declines(), raising=False)
    assert R.measure_now(DEV) is None
    src = egj.device_report()["cpu_below"]["source"]
    assert "timing image was declined" in src and "declined in this test" in src, src


def test_clear_settings_lets_the_next_decode_measure_again(unpinned, monkeypatch):
    tries = []
    monkeypatch.setattr(R, "time_routes", lambda *a: tries.append(1))               # nothing measured
    egj.decode(_jpeg(240, 160))
    _wait()
    n = len(tries)
    assert n == R.TRIES
    egj.clear_settings()
    egj.decode(_jpeg(240, 160))                          # was: no new measurement in this process
    _wait()
    assert len(tries) == 2 * n


def test_a_measurement_finished_after_clear_settings_is_not_kept(unpinned, monkeypatch):
    """Timings that cross inside the range (a value that would be kept at once): only the generation check can
    throw this measurement away. A second one, with no clear in between, is kept."""
    calls = []

    def timing(dev, data, rounds):
        h = egj.inspect(data)
        n = h["width"] * h["height"]
        if not calls:
            egj.clear_settings()                         # cleared while the measurement runs
        calls.append(n)
        return 0.2 + 5e-6 * n, 1.0 + 1e-6 * n           # the routes cross at 200,000 pixels
    monkeypatch.setattr(R, "time_routes", timing)
    assert R.measure(DEV) is None
    assert DEV not in R._state and not os.path.exists(unpinned)
    px = R.measure(DEV)
    assert px is not None and R.LOW < px < R.HIGH and R._state[DEV]["pixels"] == px


def test_a_bad_environment_value_is_ignored_with_a_warning(monkeypatch):
    """(audit 7.12) A setting that is not a number: the import works, the default is used, one warning names it;
    a bad tuning value does not make decodes raise."""
    import subprocess
    import sys
    env = dict(os.environ, EGJ_SMALL_CPU_PIXELS="200k", EGJ_MAPPED_BELOW="x", PYTHONPATH=os.pathsep.join(sys.path))
    r = subprocess.run([sys.executable, "-c", "import exact_gpu_jpeg as e, exact_gpu_jpeg._route as r, exact_gpu_jpeg._decoder; "
                        "print(e.SMALL_CPU_PIXELS, r.PINNED_BY_ENV)"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["200000", "False"]
    assert "EGJ_SMALL_CPU_PIXELS='200k'" in r.stderr and "EGJ_MAPPED_BELOW='x'" in r.stderr
    monkeypatch.setenv("EGJ_NCP", "-1")
    monkeypatch.setenv("EGJ_CHUNK_BITS", "seven")
    import numpy as np
    from conftest import reference
    from exact_gpu_jpeg import _header
    data = _jpeg(640, 480)
    _header._ENV_WARNED.discard("EGJ_NCP")
    _header._ENV_WARNED.discard("EGJ_CHUNK_BITS")
    with pytest.warns(RuntimeWarning, match="EGJ_"):
        img = egj.Decoder().decode(data)
    assert img is not None and np.array_equal(img.get(), reference(data))
