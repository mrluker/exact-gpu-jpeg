**Tasks**

# Encode a GPU image to a JPEG

`egj.encode` turns an RGB image on the GPU into the JPEG file Pillow would write for it, byte for byte.

![Encoding a photo's three web exports one at a time on a GTX 1070 + i7-6700, ms per image (lower is better): at 3840 x 2560 (quality 95, 4:4:4) exact-gpu-jpeg takes 15.0 ms from host memory and 5.18 ms with the pixels already on the GPU, against 60.4 ms for simplejpeg and 90.6 ms for Pillow on one CPU thread; at 1500 x 1000, 2.64 and 1.05 ms against 7.90 and 11.3 ms; at 1200 x 800 (quality 87, 4:2:0), 1.73 and 0.78 ms against 2.84 and 5.21 ms. exact-gpu-jpeg's files are byte-identical to Pillow's; simplejpeg's are not.](images/encode.png)

## To bytes

```python
import exact_gpu_jpeg as egj

img = egj.decode("photo.jpg")                      # CuPy uint8 (H, W, 3) RGB on the GPU
data = egj.encode(img, quality=90, subsampling="444", icc_profile=icc, exif=exif)   # bytes, as Pillow writes them
```

Only the compressed bytes are downloaded. Three web exports of a 30 MP photo: 7.35 ms against 103 ms with Pillow,
on one GTX 1070 + i7-6700 (Windows 10). More numbers: [Reproduce the benchmark](benchmarks.md#encoding).
<!-- numbers: bench/results/20261118/encode.json (three_exports_ms: gpu, pillow) -->

## To a file

```python
n = egj.encode(img, 90, "444", path="out.jpg")     # written to the file (download and write overlap); returns its size
```

With `path=` the download and the file write overlap, and `encode` returns the file's size instead of bytes.

## Quality and subsampling

```python
data = egj.encode(img)                             # quality 75, "420"
data = egj.encode(img, 90, 2)                      # Pillow's 0, 1, 2 mean "444", "422", "420"
```

`quality` is 1–100 (default 75); `subsampling` is `"444"`, `"422"` or `"420"` (default), or Pillow's 0, 1, 2.

## ICC profile and EXIF

```python
data = egj.encode(img, 95, "444", icc_profile=icc, exif=exif)
```

`icc_profile` and `exif` are bytes, written as APP2 / APP1 as Pillow does: EXIF (APP1) after the JFIF header,
then the ICC profile (APP2, 65 519-byte chunks).

## From a NumPy array

```python
a = egj.decode_cpu("photo.jpg")                    # a NumPy (H, W, 3) uint8 array
data = egj.encode(a, 90)                           # uploaded, then encoded
```

A numpy array is uploaded first. The image must be (H, W, 3) uint8 RGB, C-contiguous.

## From a PyTorch tensor

```python torch
t = egj.decode("photo.jpg", out="torch")
data_t = egj.encode(t, 90)                         # a CUDA torch tensor goes in through DLPack
```

Torch tensors go through DLPack, without copies.

## The Pillow call it equals

```python
import io
from PIL import Image
b = io.BytesIO()
Image.fromarray(img.get()).save(b, "JPEG", quality=90, subsampling=0)
assert egj.encode(img, 90, "444") == b.getvalue()
```

`encode` writes the same bytes as Pillow 13.0.0 (libjpeg-turbo 3.2.0) with `Image.fromarray(img).save(f, "JPEG",
quality=q, subsampling=s, icc_profile=icc, exif=exif)`. Pillow's default kind of file only: baseline, standard
tables, no restart markers, no progressive, no optimised tables, no custom quantisation tables, RGB input only.

## With an Encoder of your own

```python
enc = egj.Encoder()
data = enc.encode(img, 90, "420")
enc.stats                                          # {'blocks': ..., 'scan_bytes': ...}
```

One `Encoder` per thread; `egj.encode` keeps one per thread for you and is safe to call from several threads.
The encoder has no batched call: one image at a time per `Encoder`.

## An image still being computed

```python
import cupy as cp
s = cp.cuda.Stream(non_blocking=True)
with s:
    flipped = cp.ascontiguousarray(img[::-1])      # produced on another stream
s.synchronize()                                    # finish that stream yourself before encoding
data = egj.encode(flipped, 90)
```

The image must be completely written before the encoder reads it. `encode` waits for the calling thread's current
CuPy stream; an image made on another stream must be finished by you, or passed with `ready=` ([pipelines](pipelines.md)).

## When GPU encoding does not pay

```python
n = egj.encode(img, 90, "444", path="out.jpg")     # use it for latency, or when the CPU is the bottleneck
```

In a pipeline whose GPU is already the bottleneck, GPU encoding adds GPU work while Pillow would have used idle
CPU cores, so it can shorten one photo's latency and still lower the throughput of a GPU-bound batch: use it for
latency, or when the CPU is the bottleneck. The files are byte-identical either way.

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Decode into your own GPU work](pipelines.md) (`encode_async`, `ready=`)
- [Check what "identical" means](exactness.md)
- [Every name, option and variable](reference.md)

**Next:** [Decode many files in one call](batches.md)
