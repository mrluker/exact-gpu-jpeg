**Tasks**

# Small images and the CPU route

`egj.decode` sends one small image to the CPU decoder and uploads the result, because that is faster than a GPU
decode for it and gives the same pixels.

## The default: one small image goes to the CPU

```python
import exact_gpu_jpeg as egj
info = {}
img = egj.decode(jpeg_bytes, info=info)          # a small image: CPU decoder, then uploaded
info["path"]                                     # 'cpu'
```

A single image of 200,000 pixels or fewer (about 540×360) is decoded by the CPU decoder and uploaded. On a
GTX 1070 + i7-6700 (Windows 10) that is 0.51 ms instead of 0.96 ms at 0.05 MP, and the two routes meet at about
0.2 MP for 4:2:0 files; at 4:4:4 the GPU was already faster at 0.1 MP. The image is on the GPU either way, with the
same pixels.
<!-- numbers: bench/results/20261118/small_image.json (medians of 8) -->

## Keep a small image on the GPU

```python
img = egj.decode(jpeg_bytes, cpu_below=0)          # keep a small image on the GPU
```

`cpu_below=0` (or `EGJ_SMALL_CPU_PIXELS=0`) always uses the GPU.

## The value measured on your machine

```python
egj.device_report()["cpu_below"]                 # the value, where it came from, and the file
```

The crossover depends on the CPU as much as on the GPU. On first use the library times three small built-in images
(generated, not photographs) through both routes, each timed as a decode takes it (the CPU route with its header
checks and the upload, the GPU route until the GPU has finished), about a quarter of a second, in a short
background thread after
your first `decode` has returned and only while you are not decoding; the result is kept between 50,000 and 500,000.
That range is what the measurement can see: the three built-in images are 0.06 to 0.25 MP. Either route gives the
same pixels, so the value only moves time.

`source` says where the value in use came from:

| `source` | what it means |
|---|---|
| `"measured"`, `"saved"` | measured in this process, or read from the settings file; `crossing` is the pixel count where the two timed routes cross |
| `"measured, at the end of the range (clamped)"`, `"saved, at the end of the range (clamped)"` | the routes crossed outside 50,000 to 500,000 pixels (`crossing`, or `None` when they did not cross at all); the value in use is the nearest end of the range (`range`), a bound of the measurement, not the machine's own crossing |
| `"measured, an unconfirmed end of the range (...)"` | the same, measured by this process and not yet confirmed: it is used by this process only, and the next process measures again |
| `"default (not measured yet; the settings file holds an unconfirmed end of the range, measured again)"` | a new process after an unconfirmed end: the default until it has measured |
| `"default (not measured yet)"` | the fixed default, 200,000, until a measurement exists |
| `"default: measuring given up, <why>"` | every measurement was thrown away; the default stays in use |
| `"pinned by EGJ_SMALL_CPU_PIXELS"`, `"pinned by the program (SMALL_CPU_PIXELS)"` | pinned: nothing is measured |

## Your own decodes during a measurement

```python
img = egj.decode("photo.jpg")                    # CuPy uint8 (H, W, 3), RGB, on the GPU
```

A measurement starts only while your thread is idle; one that your own work overlaps is thrown away and retried,
three times at most.

## Measure it now instead of in the background

```python
r = egj.self_check()                             # also measures cpu_below, if it has not been measured
r["cpu_below"]
```

`egj.self_check()` measures it at once. A measurement whose first and second half disagree is thrown away. A
value at either end of the range is used by the process that measured it but not saved as a value: a disturbed
machine can give an end twice in a row. That process measures once more at once, and a value inside the range
replaces the end. An end is saved only when a measurement in a later process, at least 10 minutes after the first,
lands on the same end; the other end starts the 10 minutes again. A confirmed end stays, like any saved value,
until the machine, the library version or `egj.clear_settings()` changes it. A disturbance lasting longer than
10 minutes across two processes (a long compile, another GPU job, a laptop on battery) can confirm a wrong end;
`egj.clear_settings()` resets it.

## Pin the value

```python
img = egj.decode(jpeg_bytes, cpu_below=100_000)  # pinned for this call: nothing is measured
```

Passing `cpu_below=` or setting `EGJ_SMALL_CPU_PIXELS` pins the value: nothing is measured, no thread is started
and no file is written.

## The settings file

```python
egj.clear_settings()                             # delete the settings file; the 200,000 default applies again
```

The measured value is saved in `exact-gpu-jpeg/settings.json` under the local application data folder on Windows,
under `~/.cache` elsewhere, keyed by GPU, driver, CPU, CPU decoder and library version. A missing, damaged or
unwritable file just means the 200,000 default; deleting it is harmless. Nothing about any decoded file is stored.

## Where the CPU route is not taken

```python
dec = egj.Decoder()
img = dec.decode(jpeg_bytes)                     # a Decoder object always decodes on the GPU
```

The route applies to `egj.decode` with `fallback=True` and the default CPU decoder, for one image. Lists are batched
on the GPU whatever their size. With a fast mode and only Pillow installed, the small-image CPU route is not taken.

## Many small images: a batch

```python
imgs = egj.decode(files)                         # a list is decoded together on the GPU
```

For many small images a batch is faster than either route: 7,310 images/s at 0.25 MP in batches of 64, against
887 with one `decode` per image (GTX 1070 + i7-6700, Windows 10). See [Decode many files in one call](batches.md).
<!-- numbers: bench/results/20261118/nvjpeg_direct.json (sets.0.25mp: "exact-gpu-jpeg decode_batch", "exact-gpu-jpeg decode, one call per image": images_per_s) -->

## What one call costs

```bash
python bench/small_image.py out.json <dir>      # one small image: the CPU route (the default) vs always the GPU
```

Each `decode` call on the GPU has a fixed cost of about 0.58 ms (13 GPU steps and one host round trip; about 1.1 ms
at 0.25 MP), a small `encode` about 0.3 ms. Measured on the GTX 1070 + i7-6700: [benchmarks](benchmarks.md#one-small-image-and-the-fixed-cost-of-a-call).
<!-- numbers: bench/results/20261118/call_floor.json (16x16); bench/results/20261118/small_image.json (0.25 MP, always the GPU) -->

## Related

- [Decode many files in one call](batches.md)
- [The first call, warm-up and an idle GPU](first-call.md)
- [Run on a GPU it has never seen](new-gpu.md)
- [Every name, option and variable](reference.md)

**Next:** [The first call, warm-up and an idle GPU](first-call.md)
