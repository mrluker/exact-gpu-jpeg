**Background**

# Who it is for

It is for anyone who decodes or encodes many JPEGs, has an NVIDIA GPU, and cannot accept pixel differences; this
page shows what that means for five kinds of work, and where it does not help.

## Photo editors

```python
import exact_gpu_jpeg as egj
img = egj.decode("photo.jpg")                    # the same pixels as the CPU path
out = my_kernel(img)                             # your own GPU work
data = egj.encode(out, quality=90, subsampling="444", icc_profile=icc, exif=exif)   # Pillow's bytes
```

**Need:** open, edit and save photos on the GPU without changing any result. **Calls:** `decode`, `encode`, and
[pipelines](pipelines.md) for overlap. **Expect:** outputs match the CPU path, so tests, caches and file hashes keep
working.

## Culling a shoot

```python
for f in files:
    img = egj.decode(f)                          # one camera photo after another
    use(img)
```

**Need:** look through many camera photos quickly. **Calls:** `decode` in a loop; `upload_ahead` to stage the next
file ([pipelines](pipelines.md)). **Expect:** a 30 MP camera photo in 11.3 ms instead of 185 ms with libjpeg-turbo on
one core; with the next photo staged, about 100 photos/s for a stream decoded one at a time (GTX 1070 + i7-6700, Windows 10).
<!-- numbers: bench/results/20261118/nvjpeg_direct.json (camera_photos: "exact-gpu-jpeg decode, one call per image", "libjpeg-turbo, 1 thread": images_per_s); staged stream: bench/results/20261118/decode_alone.json (stream: ahead_run0/1 photos_per_s) -->

## Re-export jobs

```python
master, small = my_resize(img)                   # your own resize, on the GPU
e1, e2 = egj.Encoder(), egj.Encoder()
j1 = e1.encode_async(master, 95, "444", path="master.jpg")
j2 = e2.encode_async(small, 90, "444", path="small.jpg")
n1, n2 = j1.result(), j2.result()
```

**Need:** write several exports of each photo, byte-identical to what Pillow writes. **Calls:** `encode` or
`encode_async` on several `Encoder`s. **Expect:** three web exports in 7.35 ms instead of 103 ms with Pillow; but
where the GPU is already the bottleneck it adds GPU work, so use it for latency or when the CPU is the bottleneck.
<!-- numbers: bench/results/20261118/encode.json (three_exports_ms: gpu, pillow; GTX 1070 + i7-6700, Windows 10) -->

## Image servers

```python
egj.self_check()                                 # at start-up, so the first request does not pay for it
data = egj.encode(egj.decode(jpeg_bytes), 90)    # safe to call from several threads
```

**Need:** decode and encode on request, with the same bytes whichever machine runs. **Calls:** `self_check` at
start-up, `decode` / `encode` (one `Decoder` / `Encoder` per thread, kept for you). **Expect:** after `self_check()`
the first calls take 3 to 15 ms instead of 63 to 110; after an idle pause a call can be slower ([first call](first-call.md)).
<!-- numbers: bench/results/20261118/first_calls.json (cases: nothing and self_check, first_decode_ms and first_encode_ms); idle pause: bench/results/20261118/idle_gpu.json (GTX 1070 + i7-6700, Windows 10) -->

## Training loops

```python
import numpy as np
mean = np.array([0.485, 0.456, 0.406], np.float32)   # float32, as the output step computes
std = np.array([0.229, 0.224, 0.225], np.float32)
xs = egj.decode(files, dtype="float16", layout="CHW", scale=1 / (255 * std), bias=-mean / std)
```

**Need:** the same pixels in training and in production, as model inputs. **Calls:** a list to `decode` (a batch),
the output options, `out="torch"`. **Expect:** batches of 64 at 0.25 MP: 7,310 images/s; a normalised float16
input from a 30 MP photo in 14.2 ms instead of 26.9 ms for decode then convert (GTX 1070 + i7-6700, Windows 10).
<!-- numbers: bench/results/20261118/nvjpeg_direct.json (sets.0.25mp "exact-gpu-jpeg decode_batch": images_per_s); bench/results/20261118/epilogue.json (median of fused_ms and of decode_then_cupy_ms) -->

## Not for: progressive web images

```python
egj.inspect("progressive.jpg")["reason"]         # 'progressive JPEG': decoded on the CPU
```

Progressive JPEGs are not decoded on the GPU: they are decoded by the CPU decoder and uploaded (the same pixels).
Web images are often progressive; camera JPEGs almost never.

## Not for: one thumbnail at a time

```python
img = egj.decode(jpeg_bytes)                     # one small image: below the measured limit, on the CPU, then uploaded
```

One small image at a time is slower than the CPU: at 0.25 MP the default call took 1.03 ms against 0.80 ms for
libjpeg-turbo (in that run on the GPU, since the measured limit was 166,294 pixels; below the limit the call is the
CPU's time plus the upload). Batches are where small images win ([batches](batches.md)).
<!-- numbers: bench/results/20261118/reproduce.json (decode, mp 0.25: gpu_ms, cpu_ms, route; GTX 1070 + i7-6700, Windows 10) -->

## Not measured: cards with a hardware JPEG engine

```bash
python bench/reproduce.py                       # the benchmark tables on your GPU, from generated images
```

Every number is from one GTX 1070, which has no hardware JPEG engine. On cards with one (A100 class) nvJPEG may
well be faster: not measured. Run the script there to see; results from other cards are welcome.

## Related

- [Decode many files in one call](batches.md)
- [Decode into your own GPU work](pipelines.md)
- [Reproduce the benchmark](benchmarks.md)
- [How it works, limits, what is not done](how-it-works.md)

**Next:** [How it works, limits, what is not done](how-it-works.md)
