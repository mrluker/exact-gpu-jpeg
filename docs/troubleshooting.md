**Tasks**

# When something is wrong

Each entry gives the symptom, its cause and the fix, with the command that shows it.

Every number and test result in these docs is from one PC, a GTX 1070 on Windows 10. The library has not been run
on Linux or on newer cards: on another machine, what you see may be new; please report it with
`egj.device_report()` ([new GPU](new-gpu.md#what-to-send-in-a-bug-report)).

## Install on a card older than Turing

```bash
pip install "cupy-cuda12x[ctk]"   # CuPy with the CUDA 12 runtime and NVRTC from pip
```

**Symptom (expected, not tested):** on a GTX 900 or 10 series card or a Titan V with a CUDA 13 CuPy, the kernels
do not build and the GPU path is off. **Cause:** CUDA 13 can no longer build code for compute capability below
7.5, and the library compiles its kernels on your machine. **Fix:** install `cupy-cuda12x`, as above. Only the
CUDA 12 package was tested.

## "The GPU path is off"

```python
import exact_gpu_jpeg as egj
report = egj.device_report()
report["gpu_path"], report["self_check"]
```

**Symptom:** one `RuntimeWarning` "self-check failed" or "device check failed"; decodes come from the CPU decoder,
`encode()` raises. **Cause:** a wrong byte in the self-check (at start-up, or in the check of `fast_dct`,
`fast_upsample` or float / CHW / BGR output the first time a process asks for it), a kernel that does not compile,
missing NVRTC, a CUDA error or an unmet requirement. A shortage of GPU memory during a check declines that call only. **Fix:** send `egj.device_report()` with the warning: [new GPU](new-gpu.md).

## `FastModeUnavailable`

```python
try:
    img = egj.decode("progressive.jpg", fast_dct=True)
except egj.FastModeUnavailable as e:
    print(e)
```

**Symptom:** `egj.FastModeUnavailable` (a `RuntimeError`) with `fast_dct` or `fast_upsample`. **Cause:** the GPU
path declined the file and no CPU decoder at hand decodes in that mode (simplejpeg not installed, or your own
decoder takes no fast-mode keywords). **Fix:** `pip install simplejpeg` (the package installs it; a platform without its wheels does not have it), or
`fallback=False` to get None.

## A file is declined

```python
info = {}
img = egj.decode("progressive.jpg", info=info)     # the GPU declines it -> CPU decoder -> uploaded
info                                               # {'path': 'cpu', 'reason': 'progressive JPEG'}
```

**Symptom:** `info["path"] == "cpu"`, or `None` from a `Decoder` or with `fallback=False`. **Cause:** a kind of
JPEG the GPU path does not decode exactly (`reason` names it); `egj.inspect(path)` tells from the header alone.
**Fix:** none needed with `fallback=True`: the CPU decoder gives the reference pixels. List: [exactness](exactness.md).
Limits that decline a file before any GPU work: a file over 1 GB (before any buffer is made), a header longer than
16 MB or with more than 4,096 marker segments before the scan, and a file for which page-locked memory cannot be
had. More than 4,096 marker segments after the scan are declined too, once the scan is decoded. At the top level a declined file goes to the CPU decoder. A file libjpeg-turbo decodes only with a
warning (no end marker, data that ends early, extraneous bytes before a marker, and the like) is declined by
default and the CPU decoder raises for it too; `strict=False` gives libjpeg-turbo's image and names the warning.
See [exactness](exactness.md) for where the warning is reported, the reason texts, and what changes where the CPU
decoder is Pillow (simplejpeg not installed).
One reason is about the data, not the kind of file: `Huffman chunks did not synchronise`. Ordinary JPEGs synchronise within a few symbols (every real photo measured: 2 passes). What gets this reason: flat images coded with optimised Huffman tables, pure white images at 4:2:0, very large blank pages, and files made never to synchronise. The GPU path gives up on them after at most 8 launches of bounded work, instead of working on them for as long as the file likes, and `egj.decode` goes to the CPU decoder, which gives the reference's pixels; the files made never to synchronise also carry extraneous bytes, so with the default `strict=True` the call raises for them ([the limit](exactness.md#declined-for-the-work-it-would-take-huffman-data-that-does-not-synchronise)).

## The first call is slow

```python
egj.self_check()                                   # at start-up
```

**Symptom:** the first decode or encode of a process takes 63 to 110 ms, or 1.6 to 8.4 s on the first run on a
machine. **Cause:** the self-check and loading the kernels, or compiling them once on a new machine. **Fix:** call
`egj.self_check()` at start-up; after it the first calls take 3 to 15 ms (GTX 1070 + i7-6700):
[The first call](first-call.md).
<!-- numbers: bench/results/20261118/first_calls.json -->

## A call after a pause is slow

```python
img = egj.decode("photo.jpg")
```

**Symptom:** a small decode takes about 0.4 ms longer than in a loop, or a 24 MP decode takes 16 to 38 ms instead
of about 6. **Cause:** the card was idle and may have dropped to a low power state (measured on one GTX 1070).
**Fix:** time a warm loop, not one cold call; see [an idle GPU](first-call.md#longer-pauses-the-cards-low-power-states).
<!-- numbers: bench/results/20261118/idle_gpu.json -->

A small GPU call during the pause did not reliably prevent it.

## Out of GPU memory

```python
dec = egj.Decoder()
img = dec.decode("photo.jpg")
dec.reason                                         # None, or why the file was declined
```

**Symptom:** a large file is declined (`reason`) and, with `egj.decode`, decoded on the CPU. **Cause:** its decode
would need more than `max_gpu_bytes` (default 90% of the free memory). A batch that would not fit is split
automatically. **Fix:** free GPU memory, for example batch images kept alive (each holds the whole batch buffer;
`img.copy()` the ones you keep).

## `PermissionError` when encoding to a file

```python
import time
for k in range(20):                                # up to 2 s
    try:
        n = egj.encode(img, 90, path="out.jpg")
        break
    except PermissionError:                        # the file is open in another program: nothing was written
        if k == 19:
            raise
        time.sleep(0.1)
```

**Symptom:** `PermissionError: [Errno 13] Permission denied: 'out.jpg'` from `encode(..., path=)`, mostly on
Windows. **Cause:** the file is open in another program (a viewer, a scanner, a sync client), or it is read-only or
a folder; `encode` writes the file itself, with no temporary name and no rename. **Fix:** the file is as it was, so
call again, as above. An encode that fails after the file was opened (a full disk) removes it: no partial file is
left under that name.

## Pixels differ from another decoder's

```python
import numpy as np
import simplejpeg
ref = simplejpeg.decode_jpeg(open("photo.jpg", "rb").read(), fastdct=False, fastupsample=False)
np.array_equal(egj.decode("photo.jpg").get(), ref)
```

**Cause:** the reference is libjpeg-turbo with the accurate IDCT and fancy upsampling; other decoders differ
(nvJPEG: up to 50 levels in our runs), as do `fast_dct`, `fast_upsample` and a decoder set with `set_cpu_decoder`.
**Fix:** compare with the reference as above, or a folder with `python tools/check_dir.py <dir>` (counts only).

## See where the time goes

```python
old = egj.timing(True)                             # or EGJ_TIMING=1 in the environment; off by default
info = {}
img = egj.decode("photo.jpg", cpu_below=0, info=info)      # prints the route, the wall time and each stage's GPU time
info["ms"], info["total_ms"]                       # the same values: stage -> milliseconds, and the call
data = egj.encode(img, 90, info=info)              # an encode prints its line too
egj.timing(old)                                    # as it was
```

One line per call on stderr, `egj.device_report()` once before the first; lists (batches) are not timed. On, it adds
about 0.25 ms to a small decode and 0.13 ms to a small encode (488x480), and 0.2 to 0.3 ms to a 5 to 8 ms call at
24 MP, near the noise of that measurement (standard error 0.12 to 0.14 ms); off, the marks behind it cost at most
about a microsecond a call (GTX 1070 + i7-6700). Take end-to-end numbers with it off.
<!-- numbers: bench/results/20261118/timing_cost.json (cells: "decode 488x480" profile=True, "encode 488x480" timings={}, "decode, large image" profile=True, "encode, large image" timings={}: diff_vs_off_ms, se, median_ms of off; off.ns_per_mark_call_when_off x host_mark_calls_per_call) -->

## Speed on your card differs from the published numbers

```bash
python bench/reproduce.py                       # the benchmark tables on your GPU, from generated images
```

**Cause:** every published number is from one GTX 1070 + i7-6700 on Windows 10, and the fixed tuning values were
chosen there; on another card they can cost speed, never a byte. **Fix:** run the script and send the output.

## Related

- [Run on a GPU it has never seen](new-gpu.md)
- [The first call, warm-up and an idle GPU](first-call.md)
- [Check what "identical" means](exactness.md)
- [Install, decode, encode](getting-started.md)

**Next:** [Who it is for](use-cases.md)
