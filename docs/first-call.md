**Tasks**

# The first call, warm-up and an idle GPU

The first calls of a process carry the self-check and the loading of the kernels, and a card that has been idle
can be slow to answer; this page says what that costs and where to pay it.

## Pay at start-up: `self_check()`

```python
import exact_gpu_jpeg as egj
r = egj.self_check()                             # at start-up, at a time of your choosing
r["ok"], r["ms"]                                 # passed?, ms per part
```

`egj.self_check()` runs every part of the self-check (decode, batch, encode, and the first-use checks of the fast
IDCT, the fast upsampling and the output options) and compiles or loads the kernels. Call it once at start-up if your first image matters: it is the place to pay.

## What the first calls cost

```python
img = egj.decode("photo.jpg")                    # after self_check(): a few milliseconds
```

| | nothing called before | after `egj.self_check()` | the same call in a loop |
|---|---|---|---|
| `egj.self_check()` itself (six parts) | - | 453 to 477 | - |
| first decode, 488x480, from a file | 63 | 3 | 1.5-1.9 |
| first decode, a 5.3 MB file | 82 | 15 | 8.6-8.8 |
| first encode, 64x64, to a file | 65 | 3 | 1.2 |
| first encode, 3840x2560 at 4:4:4, to a file | 110 | 9 | 4.7 |

ms, fresh processes, kernels cached, on the GTX 1070 + i7-6700 (Windows 10); "in a loop" is the median of the
next three calls. What remains after the warm-up call is each Decoder's and Encoder's own buffers. The first time
on a machine, the first decode also starts the background measurement of `cpu_below`
([small images](small-images.md)).
<!-- numbers: bench/results/20261118/first_calls.json -->

## The first run on a machine: kernel compilation

```python
egj.self_check()                                 # on a new machine: compiles every kernel, once
egj.device_report()["nvrtc"]                     # the kernel compiler the library found
```

The very first run on a machine compiles the kernels with NVRTC. In fresh processes with an empty kernel cache the
first decode took 2.3 to 3.3 s, the first encode 1.65 s and the first batch 7.2 to 8.4 s (two rounds each, GTX
1070 + i7-6700). A decode-only program never compiles the encoder.

This is paid once per machine. CuPy keeps the compiled kernels in its disk cache: the folder `.cupy/kernel_cache`
in your home folder, or the folder the environment variable `CUPY_CACHE_DIR` names. Later processes load them from
there, which is the "nothing called before" column above. `self_check()` is the place to pay it: it compiles every
kernel the library has, so call it once after installing, or at start-up.

The kernels are compiled again when a cached entry no longer matches: a new CuPy version, a new NVRTC (CUDA
runtime) version, a GPU of another architecture, or a version of this library whose kernels changed. CuPy's cache
key holds those, not the GPU driver's version, so a driver update alone does not clear it. Deleting the folder is
harmless: the next run compiles again.
<!-- numbers: bench/results/20261118/first_calls.json (first_run_on_a_machine_ms) -->

## The first large encode

```python
n = egj.encode(img, 95, "444", path="first.jpg")
```

The large encode's row above includes about 40 ms that CuPy spends preparing its running sum the first time an
encode of more than 65,536 blocks uses it. `self_check()` has already used it.

## The first call after a pause

```python
import time
time.sleep(0.25)                                 # a quarter of a second of idleness
img = egj.decode("photo.jpg")                    # can be slower than in a loop
img = egj.decode("photo.jpg")                    # back to normal
```

A small image is slowed most. After a quarter of a second of idleness the first decode of a 488x480 JPEG took a
median of 1.34 ms against 0.96 ms in a loop, and more than 1.5 times as long in 3 of 15 tries; after one second,
1.41 ms, and 7 of 15 tries. It was back to normal after 4 to 8 ms of work. A 24 MP decode was not slowed by pauses
of a quarter of a second to two seconds (one GTX 1070, Windows 10).
<!-- numbers: bench/results/20261118/idle_gpu.json -->

## Longer pauses: the card's low power states

```python
time.sleep(5)                                    # five seconds or more: the card may drop to a lower power state
img = egj.decode("photo.jpg")
```

After five or ten seconds the card had sometimes dropped to a lower power state (P5 at 708 to 911 MHz, instead of
P2 at 1,500 to 1,870 MHz). A 24 MP decode then took 16 to 18 ms instead of about 6, in 1 of 8 tries after five seconds
and 1 of 6 after ten, and decodes were back to normal after 135 to 165 ms of work.
<!-- numbers: bench/results/20261118/idle_gpu.json -->

## A small GPU call during the pause does not reliably help

```python
for _ in range(20):                              # five seconds, a small GPU call every quarter of a second
    egj.decode(jpeg_bytes, cpu_below=0)          # a tiny image, decoded on the GPU
    time.sleep(0.25)
img = egj.decode("photo.jpg")
```

A 64x64 GPU decode every quarter, half or whole second during a five-second pause did not keep the card awake: the
first 24 MP decode after it was still slow in 1, 2 and 1 of 8 tries. Once it took 38 ms, with the card in P8 at
215 MHz. If the first call after a pause matters, plan for its cost rather than for a keep-alive call. What a
keep-alive does to the card's idle power draw was not measured, and other cards were not measured.
<!-- numbers: bench/results/20261118/idle_gpu.json -->

## Time a warm loop, not one cold call

```python
for _ in range(10):
    img = egj.decode("photo.jpg")                # warm up
t0 = time.perf_counter()
for _ in range(10):
    img = egj.decode("photo.jpg")
ms = (time.perf_counter() - t0) / 10 * 1000
```

`decode` returns when the pixels are written, so a loop of calls times the whole decode. The published runs had
the GPU to themselves on an otherwise idle machine: see [Reproduce the benchmark](benchmarks.md).

## Related

- [Install, decode, encode](getting-started.md)
- [Run on a GPU it has never seen](new-gpu.md)
- [Small images and the CPU route](small-images.md)
- [When something is wrong](troubleshooting.md)

**Next:** [Run on a GPU it has never seen](new-gpu.md)
