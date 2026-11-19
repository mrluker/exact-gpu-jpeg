"""The first-use checks of the decode modes the start-up check does not run (the fast IDCT, the fast upsampling,
float output / CHW / BGR): each runs once per process and device, when the mode is first asked for, decodes its
cases alone and in a batch, and compares with libjpeg-turbo's pixels; a failure switches the GPU path off, as a
failed start-up check does (audit of 2026-11-18, item B4)."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _selfcheck as S  # noqa: E402
from exact_gpu_jpeg import _selfcheck_data as SD  # noqa: E402

DEV = cp.cuda.Device().id


def test_every_mode_check_passes_and_reports_its_time():
    r = egj.self_check()
    assert r["ok"], r["reason"]
    for p in S.MODE_PARTS:
        assert r["ms"][p] is not None and r["ms"][p] < 10_000, (p, r["ms"])
        assert egj.device_report()["self_check"][p] == "passed"
    print("one-time cost per mode (ms, first use in this process, compilation included):",
          {p: r["ms"][p] for p in S.MODE_PARTS})


@pytest.mark.parametrize("kw, part", [(dict(fast_dct=True), "fast_dct"), (dict(fast_upsample=True), "fast_upsample"),
                                      (dict(dtype="float32"), "output"), (dict(colorspace="BGR"), "output")])
def test_a_mode_is_checked_at_its_first_use_and_once(monkeypatch, kw, part):
    monkeypatch.setattr(S, "FAILED", {})
    monkeypatch.setattr(S, "_done", {(DEV, p) for p in S.PARTS})       # the start-up parts have passed
    runs = []
    real = S._run_mode
    monkeypatch.setattr(S, "_run_mode", lambda d, p, D: runs.append(p) or real(d, p, D))
    data = SD.DECODE[1][3]
    import base64
    jpeg = base64.b64decode(data)
    cs = kw.pop("colorspace", "RGB")
    fast = {k: kw.pop(k) for k in ("fast_dct", "fast_upsample") if k in kw}
    dec = egj.Decoder(**fast)
    assert dec.decode(jpeg, cs, **kw) is not None, dec.reason
    assert dec.decode(jpeg, cs, **kw) is not None, dec.reason
    assert runs.count(part) == 1, runs
    assert S.failed(DEV) is None


def test_a_failed_mode_check_switches_the_gpu_path_off(monkeypatch):
    monkeypatch.setattr(S, "FAILED", {})
    monkeypatch.setattr(S, "_done", {(DEV, p) for p in S.PARTS})
    bad = [(p, n, o, "0" * 64, b) if p == "fast_dct" else (p, n, o, h, b) for p, n, o, h, b in SD.MODES]
    monkeypatch.setattr(SD, "MODES", bad)
    import base64
    jpeg = base64.b64decode(SD.DECODE[1][3])
    dec = egj.Decoder(fast_dct=True)
    assert dec.decode(jpeg) is None and "fast IDCT" in dec.reason
    assert S.failed(DEV) is not None
    info = {}
    img = egj.decode(jpeg, info=info, cpu_below=0)                   # every decode now uses the CPU decoder
    assert info["path"] == "cpu" and img is not None


@pytest.mark.parametrize("where", ["single", "batch"])
def test_memory_short_inside_a_mode_check_does_not_switch_the_gpu_path_off(monkeypatch, where):
    """(second audit, parts 3.1 and 3.6) A mode check whose single decode, or whose batch, is short of GPU memory
    has checked nothing: this call is declined, the GPU path stays on, and the check runs again at the next use."""
    monkeypatch.setattr(S, "FAILED", {})
    monkeypatch.setattr(S, "_done", {(DEV, p) for p in S.PARTS})
    if where == "single":
        monkeypatch.setattr(S, "budget", lambda dev: 1)                  # every decode needs more than allowed
    else:
        from exact_gpu_jpeg import _batch
        monkeypatch.setattr(_batch, "MIN_GROUP", 3)      # the batch of two goes one by one: images right, batch
                                                         # kernels never run (was: passed as checked)
    import base64
    jpeg = base64.b64decode(SD.DECODE[1][3])
    dec = egj.Decoder(fast_upsample=True)
    assert dec.decode(jpeg) is None and "memory" in dec.reason, dec.reason
    assert S.failed(DEV) is None and (DEV, "fast_upsample") not in S._done
    monkeypatch.undo()
    monkeypatch.setattr(S, "FAILED", {})
    monkeypatch.setattr(S, "_done", {(DEV, p) for p in S.PARTS})
    assert egj.Decoder(fast_upsample=True).decode(jpeg) is not None      # the check ran again, and passed
    assert (DEV, "fast_upsample") in S._done
