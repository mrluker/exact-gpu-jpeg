**Reference**

# Every name, option and variable

Everything a caller can name is listed on this page: module functions, `Decoder`, `Encoder`, their jobs, every
option and environment variable, and the scripts.

## Module functions

```python
import exact_gpu_jpeg as egj
egj.__version__
```

| name | what it does | example |
|---|---|---|
| `egj.decode(src, colorspace="RGB", *, fallback, device, out, info, dtype, layout, scale, bias, fast_dct, fast_upsample, cpu_below, cache, check)` | decode one file (`src`: a path or bytes-like) to the GPU; a list or tuple of files is a batch | `img = egj.decode("photo.jpg")` |
| `egj.decode_batch(srcs, ...)` | decode several files together (`srcs`: a list of paths or bytes-like); the options are `decode`'s | `imgs = egj.decode_batch(files)` |
| `egj.encode(img, quality=75, subsampling="420", icc_profile=None, exif=None, *, device, path, ready, info)` | JPEG bytes for an (H, W, 3) uint8 RGB image (`img`: CuPy array, CUDA torch tensor or NumPy array), or a file with `path=` | `data = egj.encode(img, 90, "444")` |
| `egj.decode_cpu(src, colorspace="RGB", fast_dct=False, fast_upsample=False)` | decode with the CPU decoder (a NumPy array): what the fallback uses | `a = egj.decode_cpu("photo.jpg")` |
| `egj.self_check(device=None)` | run the start-up self-check now, all three parts (decode, batch decode, encode), and return `{'ok', 'reason', 'gpu', 'device', 'ms'}`. It runs by itself before the first decode, batch and encode of a process; call it at warm-up to choose when to pay for it and for compiling the kernels |
| `egj.FastModeUnavailable` | the exception (a `RuntimeError`) raised when `fast_dct` / `fast_upsample` is asked for, the GPU path declines the file, and no CPU decoder is at hand that decodes in that mode (simplejpeg is not installed, or a decoder of your own takes no fast-mode keywords): the library does not return the accurate mode's pixels in its place. `fallback=False` returns None instead |
| `egj.timing(on=True)`, `EGJ_TIMING=1` | per-call timing for `egj.decode` of one image and `egj.encode`, off by default: while on, each call prints one line to stderr (the route taken and why, the wall time, the GPU time of each stage; `egj.device_report()` once before the first), and a dict passed as `info=` gets `ms` and `total_ms` ([When something is wrong](troubleshooting.md#see-where-the-time-goes)). Returns the previous setting |
| `egj.clear_settings()` | delete the one settings file the library writes (the measured `cpu_below` value) and forget what this process measured; the fixed default applies until the next measurement. Harmless at any time |
| `egj.device_report(device=None)` | what the library found on this GPU and what it chose, as a dict to print or to put in a bug report: GPU name and driver, compute capability, memory, CUDA runtime and kernel compiler versions, the minimum it needs, the settings derived from them, which parts of the self-check have passed, whether the GPU path is on (or why it is off), `cpu_below` (the value in use and its `source`: measured, saved, clamped to an end of the range, the default, or pinned; [small images](small-images.md#the-value-measured-on-your-machine)), and `copy_check`: the check of the copy kernels ([new GPU](new-gpu.md#the-check-of-the-copy-kernels)). Runs no kernel |
| `egj.inspect(src)` | read the header only, without the GPU: `supported`, `reason`, `width`, `height`, `components`, `subsampling`, `restart_interval` | `egj.inspect("photo.jpg")["supported"]` |
| `egj.set_cpu_decoder(fn)` | replace the fallback decoder: `fn(data, colorspace)` returns a NumPy (H, W, C) uint8 array; `None` restores the default | `egj.set_cpu_decoder(my_decoder)` |
| `egj.to_torch(arr)` | a CuPy array `arr` as a torch tensor on the same GPU, sharing its memory | `t = egj.to_torch(img)` |
| `egj.apply_epilogue(img, dtype, layout, scale, bias)` | the float / CHW / scale / bias output step on an existing uint8 image (NumPy or CuPy), with the decoder's rounding | `x = egj.apply_epilogue(img, "float32", "CHW")` |
| `egj.Unsupported` | the exception the header parser raises for a file the GPU path does not decode (`decode` catches it and declines; `egj._header.parse` raises it) | `except egj.Unsupported as e:` |
| `egj.__version__` | the version string | `egj.__version__` |

## `egj.Decoder(device=None, *, max_pixels, max_gpu_bytes, chunk_bits, canary, fast_dct, fast_upsample, read, keep_coefs, cache)`

```python
dec = egj.Decoder()
img = dec.decode("photo.jpg")
dec.stats
```

One per thread. Every result equals `egj.decode`'s.

| name | what it is |
|---|---|
| `decode(src, colorspace="RGB", timings=None, *, dtype, layout, scale, bias, check)` | decode one file; `None` when declined. `timings`: an optional dict that gets the GPU seconds of `upload`, `huffman` and `pixels` added to it |
| `decode_batch(srcs, colorspace="RGB", *, dtype, layout, scale, bias)` | decode several files together; a list with `None` for declined files |
| `decode_async(src, colorspace="RGB", out=None, **output)` | queue a decode and return a `DecodeJob` ([pipelines](pipelines.md)); `out`: an array of yours to decode into |
| `upload_ahead(src)` | stage a file ahead of its decode ([pipelines](pipelines.md)); `True` when staged |
| `cache` | `None` (the default) or a `RepeatCache`: `decode` of one file then looks for a copy first and keeps one after a normal decode; `check` (`"full"` or `"light"`) says how a copy is known to be current ([the same file again](repeat.md)) |
| `keep_coefs` | `None` (the default: nothing is kept), `"packed"` or `"raw"`: each decode also keeps its coefficients in GPU memory ([the same file again](repeat.md)) |
| `take_coefs()` | the `CoefHandle` of the last decode, once; `None` when nothing was kept |
| `decode_from_handle(handle, colorspace="RGB", timings=None, *, dtype, layout, scale, bias)` | the image a `CoefHandle` holds, the same bytes as the decode it came from, from the pixel stage alone; `timings` gets `pixels`; the decoder's `max_pixels` applies. `keep_coefs` applies when the handle is of another form: `take_coefs()` then gives its coefficients in the `keep_coefs` form |
| `reason` | why the last `decode` declined its file (`None` if it did not) |
| `reasons` | after `decode_batch`: one entry per file, `None` or why it was declined |
| `stats` | facts about the last decode: `subsampling`, `restart_interval`, `passes` (synchronisation passes), `batched` (files that shared a batch), `read` (how the source was read: `"streamed"`, `"whole"`, `"staged"` by `upload_ahead`, or `"memory"` for bytes), `profile_ms` |
| `profile` | set it to `True` and each decode puts the GPU milliseconds of every step in `stats["profile_ms"]` (for tuning; it adds events to the stream) |
| `stream`, `device` | the decoder's CUDA stream and GPU (for `wait_event`, [pipelines](pipelines.md)) |
| `max_pixels`, `max_gpu_bytes`, `canary`, `fast_dct`, `fast_upsample`, `read` | the constructor's settings, readable and settable between decodes |
| `S` | the `chunk_bits` setting (`None`: chosen per image) |

## `egj.RepeatCache(memory_bytes=1 << 30, *, min_pixels=None, form="packed")`

```python
cache = egj.RepeatCache(memory_bytes=1 << 30)
img = egj.decode("photo.jpg", cache=cache)
cache.stats
```

Copies of decoded files in GPU memory, for decoding the same file again ([Decode the same file again](repeat.md)). Off unless you make one and pass it.

| name | what it is |
|---|---|
| `memory_bytes` | the GPU memory the copies may hold together; the least recently used are dropped first |
| `min_pixels` | images below this are not kept (default 0: every image is) |
| `form` | how a copy is kept: `"packed"` (default) or `"raw"` |
| `stats` | counts since the cache was made: `used`, `made`, `evicted`, `changed`, `too_small`, `over_budget`, and what is held now: `copies`, `bytes` |
| `prepare(srcs, decoder=None)` | make the copies of these files now; returns how many have a copy afterwards |
| `drop(src)` | forget one file's copy; `True` if there was one |
| `clear()` | forget every copy |

## `CoefHandle`

```python
dec = egj.Decoder(keep_coefs="raw")
img = dec.decode("photo.jpg")
handle = dec.take_coefs()
handle.form, handle.nbytes, handle.width, handle.height, handle.device, handle.freed
handle.free()
```

A decode's coefficients kept in GPU memory (`egj.CoefHandle`, from `Decoder.take_coefs()`): `form` (`"packed"` or `"raw"`), `nbytes` (the GPU memory it holds), `width`, `height`, `device`, `free()` and `freed`; nothing else is public, and these cannot be set. Only a decode makes a handle: `egj.CoefHandle(...)` raises `TypeError`. What a handle guarantees and what it does not ([the same file again](repeat.md#what-a-handle-guarantees-and-what-it-does-not)):

| | |
|---|---|
| guaranteed | `decode_from_handle(h)` gives the same bytes as the decode it came from would, for the same options (the decoding decoder's `fast_dct`, `fast_upsample` and `max_pixels` apply); any `Decoder` on `h.device` can decode from it |
| valid | in the process that made it, on that device, until `free()` or until it is dropped; another device or a freed handle raises `ValueError` |
| not offered | saving it, sending it to another process, turning it into bytes or building one from bytes; its layout in memory is not part of the API and may change in any release |
| not watched | whether the file changed since: a handle holds one decode (`RepeatCache` checks the file for you) |
| may fail | `decode_from_handle` returns `None` with `dec.reason` when the GPU path is off, memory is short, the image is above the decoder's `max_pixels`, or the copy is found damaged; the normal decode gives the same pixels |

## `DecodeJob`

```python
job = dec.decode_async("photo.jpg")
img = job.result()
```

A `DecodeJob` (from `decode_async`) has `event` (the GPU event reached when the pixels are written), `out` (the array
being written), `result()` (waits; that same array, or `None`), `reason` (why `result()` gave `None`; the decoder's
own `reason` belongs to its latest call), `redone` (`True` when the image was decoded a second time into that array
after `event`; [pipelines](pipelines.md)) and `dec` (its decoder). An exception raised by `result()` is raised again
by every later call.

## `egj.Encoder(device=None)`

```python
enc = egj.Encoder()
data = enc.encode(img, 90, "444")
enc.stats
```

One per thread. Every result equals `egj.encode`'s, and so Pillow's file. An `Encoder` refuses an image on
another GPU; `egj.encode()` given a GPU array uses the encoder of the GPU the image is on.

| name | what it is |
|---|---|
| `encode(img, quality=75, subsampling="420", icc_profile=None, exif=None, timings=None, path=None, ready=None)` | encode one image; bytes, or the file's size with `path=`. `timings`: an optional dict that gets the GPU seconds of `dct`, `huffman` and `stuff_download` added to it |
| `encode_async(img, quality=75, subsampling="420", icc_profile=None, exif=None, path=None, ready=None)` | queue an encode and return an `EncodeJob` ([pipelines](pipelines.md)) |
| `stats` | after an encode: `blocks` (8×8 blocks) and `scan_bytes` (compressed bytes) |
| `stream`, `device` | the encoder's CUDA stream and GPU |

## `EncodeJob`

```python
ejob = enc.encode_async(img, 90, "444")
data = ejob.result()
```

An `EncodeJob` (from `encode_async`) has `event` (the GPU event reached when the compressed data is complete on the
device), `result()` (waits, downloads, writes the file with `path=`) and `enc` (its encoder). An exception raised by
`result()`, a failed write among them, is raised again by every later call.

## Every option and environment variable

```python
img = egj.decode("photo.jpg", colorspace="RGB", fallback=True, out="cupy")    # the defaults, written out
```

Every option keeps the guarantee: what the GPU returns equals a named reference bit for bit, or the file is
declined. "Same bytes" means: equal to the default call, and so to libjpeg-turbo (decode) or Pillow (encode).

| option | where | default | what it does | output |
|---|---|---|---|---|
| `colorspace` | decode | `"RGB"` | `"RGB"`, `"BGR"`, `"GRAY"` (the Y channel) | libjpeg-turbo's in that colour space |
| `fallback` | `egj.decode`, `egj.decode_batch` | `True` | declined files go through the CPU decoder and are uploaded; `False`: `None` | the CPU decoder's (simplejpeg = libjpeg-turbo; else Pillow) |
| `cpu_below`, `EGJ_SMALL_CPU_PIXELS` | `egj.decode`, environment | 200,000 pixels until measured on this machine (then 50,000 to 500,000, the range the three built-in timing images of 0.06 to 0.25 MP can show; setting either one pins the value and nothing is measured) | one image of at most this many pixels is decoded by the CPU decoder and uploaded (faster for a single small image; [small images](small-images.md)); `0` always uses the GPU; lists are batched on the GPU whatever their size | same bytes |
| `out` | `egj.decode`, `egj.decode_batch` | `"cupy"` | `"torch"`: CUDA tensors sharing the memory | same bytes |
| `info` | `egj.decode`, `egj.decode_batch` | `None` | a dict (a list for batches) filled with `path` and `reason` | |
| `device` | `egj.*`, `Decoder`, `Encoder` | current | the GPU to use | same bytes |
| `dtype`, `layout`, `scale`, `bias` | decode | `"uint8"`, `"HWC"`, none | float output, channel planes, per-channel scale and bias, which must be finite (`ValueError` otherwise) ([decode](decode.md#straight-to-a-model-input)) | NumPy's `img.astype(np.float32) * np.float32(scale) + np.float32(bias)`, bit for bit (scale and bias rounded to float32 first; float16 rounds that to nearest even) |
| a list or tuple of files; `decode_batch` | decode | | files decoded together ([batches](batches.md)) | same bytes |
| `fast_dct` | decode, `Decoder` | `False` | libjpeg-turbo's fast integer IDCT (`JDCT_IFAST`) | **not** the default bytes: equal to simplejpeg `fastdct=True`; blocks where its SIMD and C versions disagree are declined |
| `fast_upsample` | decode, `Decoder` | `False` | libjpeg-turbo's fast upsampling (chroma repeated) | **not** the default bytes: equal to simplejpeg `fastupsample=True` |
| `decode_async(src, out=None)` | `Decoder` | | queue a decode, get a job ([pipelines](pipelines.md)) | same bytes |
| `upload_ahead(src)` | `Decoder` | | stage a file ahead of its decode; a relative path is resolved when it is staged, bytes changed in place afterwards are decoded as they were when staged, and `False` always comes with a reason ([pipelines](pipelines.md)) | same bytes |
| `cache`, `check` | decode, `Decoder` | none, `"full"` | decode a file seen before from a copy kept on the GPU; `check="light"` skips reading the file and can miss a change ([the same file again](repeat.md)) | same bytes |
| `keep_coefs`, `take_coefs()`, `decode_from_handle(handle)` | `Decoder` | `None` | keep a decode's coefficients on the GPU and decode from them again ([the same file again](repeat.md)) | same bytes |
| `max_pixels` | `Decoder` | 2^28 | larger images are declined | |
| `max_gpu_bytes` | `Decoder` | 90% of free memory | images whose decode would need more are declined | |
| fixed limits | decode | | declined: a file over 1 GB (before any buffer is made), a header longer than 16 MB or with more than 4,096 marker segments before the scan (or after it, once the scan is decoded), a file for which page-locked memory cannot be had; at the top level a declined file goes to the CPU decoder | |
| `strict` | decode | `True` | a file libjpeg-turbo decodes only with a warning is declined, and the CPU decoder raises for it too; `False`: libjpeg-turbo's non-strict image, and the warning is named; `info["warnings_checked"]`: `False` when the pixels came from Pillow, which does not report the warnings inside the scan; unused bytes at the end of the scan data are declined by the GPU path in both modes and decided by the CPU decoder ([exactness](exactness.md)) (keyword of `decode()`, `decode_batch()`, `decode_cpu()` and `Decoder(strict=)`; the warning: `info["warning"]`, `Decoder.warning`, `DecodeJob.warning`, `Decoder.warnings`) | libjpeg-turbo's (as djpeg gives) |
| `EGJ_SELFCHECK_FAIL=1` | environment | unset | forces the start-up self-check to fail, to test what your program does when the GPU path is off (`decode()` on the CPU with one warning, `encode()` raising). There is no switch that skips the check | the CPU decoder's bytes (the same) |
| `EGJ_COPYCHECK_FAIL` | environment | unset | `packed`: the check of the copy kernels is made to fail, to test what your program does then (copies kept in the raw form); decodes are not affected ([new GPU](new-gpu.md#the-check-of-the-copy-kernels)) | same bytes |
| `canary`, `EGJ_CANARY=1` | `Decoder`, environment | off | guard bands around every device buffer, checked after each decode (a test mode, slower) | same bytes |
| `chunk_bits`, `EGJ_CHUNK_BITS`, `EGJ_NCP` | `Decoder`, environment | chosen per image | Huffman chunk size and checkpoints per chunk (testing, tuning) | same bytes |
| `EGJ_MAPPED_BELOW` | environment | 131,072 bytes | a JPEG of at most this many bytes given in memory is not copied to the GPU: the kernels read it where it is, in page-locked host memory mapped into the GPU's address space (a copy command costs more than those reads until the file is about this long); `0` always copies; devices that do not report host-memory mapping and unified addressing always copy | same bytes |
| `read`, `EGJ_STREAM_READ` | `Decoder`, environment | `"auto"` | how `decode(path)` reads the file: `"streamed"` (in pieces, each copied to the GPU while the next is read: faster for one decoder), `"whole"` (the whole file, then one copy: faster when several decoders work at once), `"auto"` (streamed unless another `Decoder` is decoding, or decoding a batch, at that moment). `EGJ_STREAM_READ=1` / `0` force streamed / whole for every decoder. 30 MP photos: auto 76 photos/s with one thread (whole: 71 to 73), 98 to 101 with two (streamed: 83 to 85) | same bytes |
| `quality`, `subsampling` | encode | 75, `"420"` | 1–100; `"444"`, `"422"`, `"420"` (or Pillow's 0, 1, 2) | Pillow's file for those settings |
| `icc_profile`, `exif` | encode | none | bytes written as APP2 / APP1, as Pillow does | Pillow's file |
| `path` | encode | `None` | write the file instead of returning bytes; returns its size | the same file |
| `ready` | `encode`, `encode_async` | `None` | a GPU event the encoder waits for on the GPU; building the tables of a size, quality and subsampling waits on the host (the first time they are used on a GPU, and again once more than 64 others have been used since), so the host then waits for the producer ([pipelines](pipelines.md)) | same bytes |
| `encode_async(...)` | `Encoder` | | queue an encode, get a job ([pipelines](pipelines.md)) | same bytes |
| `egj.set_cpu_decoder(fn)` | module | simplejpeg, else Pillow | the decoder used for the fallback: `fn(data, colorspace) -> numpy (H, W, C) uint8`, optionally also taking `fast_dct` and `fast_upsample` | whatever `fn` returns (your responsibility) |
<!-- numbers: read row: bench/results/20261118/decode_alone.json (auto), decode_alone_whole.json (EGJ_STREAM_READ=0), decode_alone_streamed.json (EGJ_STREAM_READ=1): photos_per_s, gpu_1_threads and gpu_2_threads, two runs each; 12 camera photos, GTX 1070 + i7-6700, Windows 10 -->

## The CPU decoder in a fast mode

```python
egj.set_cpu_decoder(None)                        # None restores the default (simplejpeg, else Pillow)
```

`fast_dct` and `fast_upsample` reach simplejpeg, which has those modes; Pillow has only the accurate mode, so without
simplejpeg a fast mode skips the small-image CPU route and a declined file raises `egj.FastModeUnavailable`. A decoder
set with `set_cpu_decoder` gets the fast-mode keywords if it takes them; if not, it is not used in a fast mode.

## Check scripts

```bash
python tools/check_dir.py <dir>                    # decoder on a folder of your JPEGs, against libjpeg-turbo
python tools/check_dir.py --batch 16 <dir>         # the same through decode_batch
python tools/check_encode.py <dir>                 # encoder on a folder of your photos, against Pillow
```

All print counts only, never file names:

| script | what it does |
|---|---|
| `tools/check_dir.py [--batch N] <dir>` | decoder against libjpeg-turbo on a folder of your JPEGs |
| `tools/exactness_table.py <out.json> <work dir> [--decode LABEL=PATH ...] [--encode LABEL=DIR] [--keep LABEL=FILE:WHEN]` | the exactness table (`bench/results/exactness.json`) from counts only: `check_simd.py` on the corpus and the crafted files, `check_dir.py` on the folders or lists given, `check_encode.py` on a folder of photos; rows carry the labels given, never a file or folder name; exits 1 on any GPU result that is not identical |
| `tools/check_simd.py <work dir>` | decoder against libjpeg-turbo's AVX2, SSE2 and C code paths (corpus and crafted extreme files) |
| `tools/check_fastdct.py <work dir>` | the same for `fast_dct=True` |
| `tools/check_encode.py <dir>` | encoder against Pillow on a folder of your photos |
| `tools/fuzz.py <out dir> [--iters N]` | decoder fuzzer: mutated JPEGs must decode exactly or be declined |
| `tools/fuzz_batch.py <out.json>` | `decode_batch` against `decode()`, damaged files among good ones, guard bands on |
| `tools/fuzz_encode.py <out.json>` | encoder parameter fuzzer |
| `tools/fuzz_suite.sh`, `tools/fuzz_report.py` | the 12-part fuzz run behind the published numbers, and its summary |
| `tools/gpu_load.py <seconds>` | keeps the GPU busy from a second process while another check runs |
| `tools/refdecode.py` | reference decodes in one libjpeg-turbo code path (used by `check_simd.py` and `check_fastdct.py`) |
| `tools/make_selfcheck.py [--check]` | writes the self-check's built-in cases and hashes (`src/exact_gpu_jpeg/_selfcheck_data.py`) from libjpeg-turbo and Pillow, without a GPU; `--check` says whether the committed file is current |
| `tools/make_local_corpus.py <photos> <out>` | re-encodes of your own photos in every layout, for `check_dir.py` (keep the output out of git) |
| `tools/sanitize_run.py` | a small decode run for NVIDIA compute-sanitizer (Linux) |
| `tools/privacy_scan.py [root] [--name TEXT ...]` | scans a source tree for personal details before a release (user-profile paths, this computer's account and host names, e-mail addresses); exits 1 on any hit, and runs in the test suite |

## The test suite

```bash
pip install -e ".[cuda12,test]"
pytest                                             # exactness of both codecs, the API, malformed input
EGJ_CANARY=1 pytest                                # the same with guard bands around every device buffer
python tools/gpu_load.py 120 & pytest              # the same while a second process keeps the GPU busy
python tools/fuzz.py work/fuzz --iters 20000       # decoder fuzzer (set EGJ_CANARY=1 for guard bands)
python tools/fuzz_batch.py work/bfuzz.json         # decode_batch against decode(), damaged files among good ones
python tools/fuzz_encode.py work/enc.json          # encoder parameter fuzzer
```

The run under GPU load matters: a fault where one CUDA stream reads memory another has not finished writing almost
never shows on an idle GPU. The CPU-only part of `pytest` also runs in GitHub Actions
(`.github/workflows/cpu-tests.yml`); the GPU tests need a machine with an NVIDIA GPU and run on the test PC.

## Benchmark scripts

```bash
python bench/reproduce.py                       # the benchmark tables on your GPU, from generated images
```

Each writes a JSON of counts and timings:

| script | what it measures |
|---|---|
| `bench/reproduce.py [--quick] [--out results.json]` | the benchmark tables on your own GPU, from images the script makes itself (no photos, no downloads): decode against libjpeg-turbo (and nvJPEG if torchvision is installed), batches, encode against Pillow, with every result checked for identical pixels / bytes; exits 1 if any differs |
| `bench/bench.py <out.json> name=<dir> ...` | decode against libjpeg-turbo and nvJPEG, one image at a time and by throughput |
| `bench/batch_bench.py <out.json> <dir>` | batches of small images against one call each, CPU threads and nvJPEG |
| `bench/small_image.py <out.json> <dir>` | one small image at a time: the CPU route `egj.decode` takes by default against always the GPU (`cpu_below=0`) |
| `bench/first_calls.py <out.json>` | what the first calls of a process cost with nothing called before and after `egj.self_check()`, in fresh processes it starts itself, on generated images |
| `bench/idle_gpu.py <out.json> [--file large.jpg]` | a decode after pauses of 0.25 to 10 s, how long until it is normal again, what the driver reports during a slow one, and the same with a small GPU call during the pause |
| `bench/timing_cost.py <out.json> [large.jpg]` | what `timings=`, `profile` and the marks behind them cost, on and off |
| `bench/repeat_sizes.py <out.json> <work dir>` | decoding the same file again, by image size: one decode from a file and from bytes without a cache, from a copy with the full check, and from a copy with the light check |
| `bench/repeat_paths.py <out.json> [--photos DIR] [--public DIR]` | a repeat decode end to end through the API, from a path, against a normal decode of the same file: the memory cache with `check="light"` and with `check="full"`, and the identity's host cost alone; on up to 12 photos of a folder (reported as photo01 ...) and the imagecompression.info images made into JPEGs at quality 95, 4:2:0, in a temporary folder; pixels compared before timing |
| `bench/copy_forms.py <out.json>` | what each form of a kept copy costs, by image size (generated JPEGs, 0.1 to 24 MP, from bytes): a normal decode, a decode that also keeps a packed or a raw copy, a decode from a packed handle and from a raw one; the GPU memory each form holds |
| `bench/sync_worst_launch.py <out.json> [MB ...]` | how long one synchronisation launch can run on a JPEG made so that its Huffman data never synchronises, by file size; each launch is waited for before and after and timed on its own; the files are made in memory |
| `bench/sync_pass_limit.py <out.json>` | what the limit on a file's synchronisation passes does: the whole `egj.decode` call with the limit at 64, 5 and 4 passes, next to the CPU decoder alone and the GPU decoder alone with 64 passes, on files made never to synchronise and on blank pages with a few lines (64 to 144 MP, 4:2:0 and 4:4:4, default and optimised tables); about 8 GB of host memory |
| `bench/public_set.py <out.json> [--cache DIR] [--no-download]` | the public image set (imagecompression.info's 8-bit test images and libjpeg-turbo's three benchmark frames, downloaded into a cache folder outside the repository and checked by SHA-256; `--no-download` uses the cached archives): 102 JPEGs made in memory, each decode against libjpeg-turbo, each colour encode against Pillow, a decode from a packed copy, and the other decoders and nvJPEG's encoder when installed; nothing is timed |
| `bench/call_floor.py <out.json> [dir]` | the fixed cost of one call: where the time of `decode()` and `encode()` goes for a 16×16 and a 64×64 image, step by step, and what a kernel launch, a small allocation, a readback and a wait cost on this machine; `--sched spin\|yield\|block` sets the CUDA host-wait mode |
| `bench/one_image_paths.py <out.json> <dir>` | one image at a time through the single-image path against a batch of one |
| `bench/nvjpeg_direct.py <out.json> <dir>` | nvJPEG through its own C API (`nvjpegDecode` with one decoder state and with eight, `nvjpegDecodeBatched`, every back end the GPU accepts) against this library and torchvision's route, on the same files, with nvJPEG's pixel differences from libjpeg-turbo |
| `bench/decode_alone.py <out.json> <dir>` | decode from disk: stage times, threads, and the `upload_ahead` stream |
| `bench/download_bench.py <out.json>` | copying a decoded image back to host memory: `img.get()` against `img.get(out=)` into a page-locked buffer, 1 to 30 MP |
| `bench/epilogue_bench.py <dir>` | decode straight to a normalised float16 CHW array against decode-then-convert |
| `bench/encode_bench.py <out.json> <dir>` | encode against Pillow and simplejpeg |
| `bench/encode_pair.py <out.json> <dir>` | two exports of a photo: two `encode()` calls against `encode_async` on two encoders |
| `bench/encode_two_small.py <out.json>` | two small exports (488x480, generated): two `encode()` calls against `encode_async` on two encoders; checked against Pillow |
| `bench/make_ladder.py <image> <out dir>` | one image (the published ladder: `big_building.ppm` of the public set) cropped to 3:2 and resized (Lanczos) to 2, 6, 12 and 24 MP in four layouts, with a `ladder.json` saying so, as input for `bench.py --about` |
| `bench/charts.py`, `bench/report.py`, `bench/readme_numbers.py` | the charts, Markdown tables and fuzz numbers from the committed results |
| `bench/with_lock.py`, `bench/speed_step.sh` | how the published runs were made on a shared PC: a lock file, an idle check, and the conditions recorded next to each result |

## Words that look alike

```python
img = egj.decode("photo.jpg", fast_dct=True, fast_upsample=True)    # libjpeg-turbo's two faster decoding modes
```

`fast_dct` and `fast_upsample` are options of this library: they select libjpeg-turbo's two faster decoding modes,
and change the pixels to what libjpeg-turbo gives in those modes. They are off by default and are not faster on
the GPU; they exist so a caller that used those modes on the CPU gets the same bytes.

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Encode a GPU image to a JPEG](encode.md)
- [Decode into your own GPU work](pipelines.md)
- [When something is wrong](troubleshooting.md)

**Next:** [Parallel Huffman decoding by self-synchronisation](huffman-sync.md)
