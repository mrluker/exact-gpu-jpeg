**Tasks**

# Run on a GPU it has never seen

On any NVIDIA card the library either returns the same bytes as libjpeg-turbo from the GPU, or uses the CPU
decoder and says why.

## The device check

```python
import exact_gpu_jpeg as egj
report = egj.device_report()                     # runs no kernel
report["compute_capability"], report["gpu_path"]
```

Before any kernel is compiled the library reads the compute capability (5.0, the oldest CUDA 12 supports, is the
minimum), total and free memory, and whether NVRTC loads. If a requirement is not met, the GPU path is off for the
process, with the requirement named. Speed on cards other than the GTX 1070 has not been measured.

## The self-check

```python
r = egj.self_check()
r["ok"], r["reason"]
```

| part | cases | compared with |
|---|---|---|
| first decode | 4:2:0 with restart markers, 4:2:0 without restarts (long enough for many Huffman chunks), 4:2:2, 4:4:4, gray | the pixels libjpeg-turbo decodes from the same built-in JPEGs |
| first batch | two files decoded by one call | the same |
| first encode | a small image (the few-launch route) and a larger one (the general route) | the files Pillow writes |

It runs by itself once per process and per GPU, before the first decode, the first batch and the first encode;
results are compared with SHA-256 hashes stored in the package. There is no switch to skip it.

Besides the start-up check, each of these modes is checked on the card the first time it is asked for in a
process: the fast IDCT (`fast_dct`), the fast upsampling (`fast_upsample`), and float output / CHW / BGR. A failure
switches the GPU path off for the process, like a failed start-up check. A shortage of GPU memory during any check
declines that call only, and the check runs again on the next call. `self_check()` runs them all and reports each
part's time.

## When a check fails

```
exact_gpu_jpeg: self-check failed on <GPU name>, driver <version>: <case>. The GPU path is off for this process: ...
exact_gpu_jpeg: self-check failed on <GPU name>, driver <version>: <case> (<error type>: <message>). The GPU path ...
exact_gpu_jpeg: device check failed on <GPU name>, driver <version>: <requirement>. The GPU path is off ...
```

A wrong byte, a kernel that does not compile, missing NVRTC or any later CUDA error switches the GPU path off with
one `RuntimeWarning`. Then `decode()` returns the CPU decoder's pixels, `Decoder.decode()` and `fallback=False`
decline every file with that reason, and `encode()` raises it. Running out of GPU memory only declines that file.

## Test your own handling

```python
import os
os.environ.get("EGJ_SELFCHECK_FAIL")             # set it to "1" before the process starts to force a failure
```

`EGJ_SELFCHECK_FAIL=1` forces the self-check to fail, so you can see what your program does when the GPU path is
off: `decode()` on the CPU with one warning, `encode()` raising.

## The check of the copy kernels

```python
egj.device_report()["copy_check"]                # packed: "passed", "failed: <why>" or "not run yet"
```

The kernels behind packed copies have a check of their own: on each GPU, a packed copy of a built-in test vector
is made and unpacked, and compared with stored hashes. It runs once per GPU, at the first use of the packed form
there (`keep_coefs="packed"`, a `RepeatCache`); never at start-up, and never in a process that keeps no packed
copies.

```
exact_gpu_jpeg: the check of the copy kernels failed on <GPU name>: <part>: <why> (<what follows>). Decodes are not affected.
```

If it fails, one `RuntimeWarning` says so; the GPU path stays on and no decode is affected: copies are kept in the
raw form, a `RepeatCache`'s too. `EGJ_COPYCHECK_FAIL=packed`, set before the first use of the packed form, makes
the check fail, to test what your program does then.

## Settings that follow from the card

```python
egj.device_report()["derived_settings"]          # memory budget share and the data one batch may hold
```

A decode may plan to use 90% of the GPU memory free at that moment (a larger image is declined to the CPU decoder),
and one batch holds entropy-coded data in proportion to the card's memory (16 MB on a 2 GB card, 64 MB from 8 GB
up). Nothing is timed or stored for these two. The one measured value is `cpu_below`; `device_report()["cpu_below"]["source"]` says
whether it was measured, saved, clamped to an end of the range the measurement can see (50,000 to 500,000 pixels,
from the sizes of the three built-in timing images, 0.06 to 0.25 MP), still the default, or pinned. Either route
gives the same pixels: [Small images](small-images.md#the-value-measured-on-your-machine).

## Values tuned on the GTX 1070

```python
dec = egj.Decoder(chunk_bits=None, read="auto")  # chunk length chosen per image; the default read mode
```

The chunk length of the Huffman passes (`chunk_bits`), the size below which a file is read in place
(`EGJ_MAPPED_BELOW`) and the read mode (`read`, `EGJ_STREAM_READ`) are fixed at the GTX 1070 + i7-6700's values and
can be overridden. On another card those values can cost speed; they cannot change a byte.

## What to send in a bug report

```bash
python bench/reproduce.py                       # the benchmark tables on your GPU, from generated images
```

It prints the machine, the versions and `egj.device_report()` first, then checks every result against
libjpeg-turbo and Pillow, exiting with status 1 if a pixel or a byte differs. Please send the output from other
cards; `egj.device_report()` is the dict to print or to put in a bug report (with `copy_check`, the check of the
copy kernels, once a copy form has been used).

## Run the tests there

```bash
pip install -e ".[cuda12,test]"
pytest                                          # decode + encode exactness, API, malformed input (~10 s)
EGJ_CANARY=1 pytest                             # the same with guard bands around every device buffer
```

Nothing is downloaded: the tests generate their JPEGs. Not yet run anywhere: Linux, a recent GPU (Ampere or newer),
a GPU with nvJPEG's hardware engine, CUDA 11 ([Not done yet](how-it-works.md#not-done-yet)).

## Related

- [Install, decode, encode](getting-started.md)
- [The first call, warm-up and an idle GPU](first-call.md)
- [When something is wrong](troubleshooting.md)
- [Reproduce the benchmark](benchmarks.md)

**Next:** [Reproduce the benchmark](benchmarks.md)
