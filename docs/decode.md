**Tasks**

# Decode a JPEG to a GPU array

`egj.decode` turns a JPEG file or bytes into a CuPy array on the GPU whose pixels equal libjpeg-turbo's.

## From a file

```python
import exact_gpu_jpeg as egj

img = egj.decode("photo.jpg")                      # CuPy uint8 (H, W, 3) RGB on the GPU
```

`decode` returns when the pixels are written. It keeps one `Decoder` per thread and is safe to call from several
threads.

## From bytes, in another colour space

```python
img = egj.decode(jpeg_bytes, colorspace="BGR")     # a path or any bytes-like; "RGB", "BGR" or "GRAY" (H, W, 1)
```

`"GRAY"` is the Y channel, shape (H, W, 1). Each colour space equals libjpeg-turbo's output in that colour space.

## Straight to a model input

```python
import numpy as np
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
x = egj.decode("photo.jpg", dtype="float16", layout="CHW", scale=1 / (255 * std), bias=-mean / std)
```

`dtype` is `"uint8"` (default), `"float32"` or `"float16"`; `layout` is `"HWC"` (default) or `"CHW"`; `scale` and
`bias` are one number or one per channel, and must be finite (`ValueError` otherwise). The values equal NumPy's
`img.astype(np.float32) * np.float32(scale) + np.float32(bias)`, bit for bit: `scale` and `bias` are rounded to
float32 first, then one float32 multiply and one float32 add; `"float16"` rounds that to nearest even. With float64
`scale` or `bias` NumPy itself computes in float64 and gives other values, so give float32 arrays as above. On
30 MP photos 14.2 ms instead of 26.9 ms for decode then CuPy conversion.
<!-- numbers: bench/results/20261118/epilogue.json (median of fused_ms and of decode_then_cupy_ms; GTX 1070 + i7-6700, Windows 10) -->

## The same output step on an image you already have

```python
y = egj.apply_epilogue(img, "float32", "CHW")      # uint8 image (NumPy or CuPy) -> float32 channel planes
```

`apply_epilogue` is the float / CHW / scale / bias step with the decoder's rounding; the CPU fallback's images go
through it too, so the float options give the same values whichever path decoded the file.

## As a PyTorch tensor

```python torch
t = egj.decode("photo.jpg", out="torch")           # a torch CUDA tensor sharing the memory (DLPack)
```

`out="torch"` gives a CUDA tensor without a copy. `egj.to_torch(arr)` does the same for a CuPy array you already have.

## Back to host memory

```python
import cupyx
img = egj.decode("photo.jpg")
host = cupyx.empty_pinned(img.shape, img.dtype)        # page-locked host memory: allocate once, reuse it
img.get(out=host)                                      # about 3x faster than img.get() for a large image
```

The library returns GPU arrays only. A program that wants a NumPy image copies it back itself, and how it copies
matters. `img.get()` copies into new, pageable host memory, which the driver stages at about 1.8 to 2.0 GB/s on
this PC. Copied into a page-locked buffer allocated once, the same image moves at 5.4 to 5.9 GB/s. A decoded 30 MP
photo (90 MB) took 49.9 ms with `img.get()` and 15.7 ms into a page-locked buffer; a 6 MP image 9.0 against
3.1 ms (GTX 1070 on PCIe x8). Page-locked memory cannot be paged out by the OS, so allocate it once and reuse it,
not once per image.
<!-- numbers: bench/results/20261118/download.json (bench/download_bench.py) -->

## When the GPU declines a file

```python
info = {}
img = egj.decode("progressive.jpg", info=info)     # the GPU declines it -> CPU decoder -> uploaded
info                                               # {'path': 'cpu', 'reason': 'progressive JPEG'}
```

A declined file is decoded on the CPU (simplejpeg = libjpeg-turbo, else Pillow) and uploaded: the caller still
gets the reference pixels. `info["path"]` says which path ran. What is declined: [Check what "identical" means](exactness.md).

## Without the CPU fallback

```python
img = egj.decode(data, fallback=False)             # None instead of the CPU fallback
```

With `fallback=False` a file the GPU path does not decode returns `None`.

## With a Decoder of your own

```python
dec = egj.Decoder(device=0)                      # one per thread (each has its own CUDA stream)
img = dec.decode(data)
if img is None:
    print(dec.reason)                            # e.g. 'bad Huffman code', 'restart markers missing ...'
```

A `Decoder` object is for one thread at a time and always decodes on the GPU: it returns `None` where
`egj.decode` would fall back. Every `Decoder` option is in the [reference](reference.md).

## Look at the header first

```python
egj.inspect("photo.jpg")                         # header only, no GPU: size, subsampling, supported?
```

`inspect` returns `supported`, `reason`, `width`, `height`, `components`, `subsampling`, `restart_interval`.
`supported` means the header is one the GPU path handles; the data can still turn out to be corrupt.

## GPU memory a decode needs

```python
dec = egj.Decoder(max_gpu_bytes=None)            # None: 90% of the memory free at that moment
```

About 133 bytes per 8×8 block for the coefficients, the Y/Cb/Cr planes, the output and twice the compressed data.
The decoder estimates this before it allocates anything and declines the file (`reason`) if it would not fit;
larger than `max_pixels` (default 2^28) is declined too.

## Opt-in: libjpeg-turbo's fast modes

```python
img = egj.decode("photo.jpg", fast_dct=True)    # opt-in: libjpeg-turbo's fast IDCT
```

`fast_dct=True` and `fast_upsample=True` give the bytes of those libjpeg-turbo modes, not the default bytes.
They are compatibility options, never the default: see [Check what "identical" means](exactness.md#opt-in-fast-modes).

## Related

- [Decode many files in one call](batches.md)
- [Decode into your own GPU work](pipelines.md)
- [Small images and the CPU route](small-images.md)
- [Every name, option and variable](reference.md)

**Next:** [Encode a GPU image to a JPEG](encode.md)
