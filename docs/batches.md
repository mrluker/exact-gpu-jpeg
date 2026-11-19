**Tasks**

# Decode many files in one call

Pass a list of files and the ones that share their Huffman tables and sampling factors are decoded together on the GPU.

## A list to `decode`

```python
import exact_gpu_jpeg as egj

imgs = egj.decode(["a.jpg", "b.jpg", jpeg_bytes])          # a list or tuple: decoded together, a list of images
```

Each image equals what `decode` returns for that file, byte for byte. Lists are batched on the GPU whatever the
size of their images: the small-image CPU route ([Small images](small-images.md)) applies to single images only. A
batch can decode a file that a single `decode()` declines because its stream did not synchronise in time (the
pixels are libjpeg-turbo's); never the reverse. A list neither uses nor fills a `RepeatCache`.

## `decode_batch`, the same call by name

```python
imgs = egj.decode_batch(files, fallback=False)             # the same call by name; None where the GPU declines
info = []
imgs = egj.decode_batch(files, info=info)                  # info[i] = {'path': 'gpu' | 'cpu', 'reason': ...}
```

The options are `decode`'s. With `fallback=False` a declined file is `None` in the list; otherwise it is decoded
on the CPU and uploaded, and `info[i]["path"]` says which path ran.

## On a Decoder of your own

```python
dec = egj.Decoder()
imgs = dec.decode_batch(files)                             # on a Decoder of your own; dec.reasons[i] says why not
```

A `Decoder` returns `None` for a declined file and never falls back; `dec.reasons` has one entry per file.

## Which files were decoded together

```python
dec.stats["batched"]                                       # files that shared a batch in the last call
```

Files that share their Huffman tables and sampling factors are decoded as one stream by about 15 GPU launches
for the whole batch (one call per file takes 13 each). Files from one camera, phone or encoder usually share
them; the others, and all files with `fast_dct=True`, are decoded one by one inside the same call.

## Float output for a batch

```python
xs = egj.decode(files, dtype="float32", layout="CHW", scale=1 / 255)       # batches take the same options
```

`dtype`, `layout`, `scale` and `bias` work as for one file: [Decode a JPEG to a GPU array](decode.md#straight-to-a-model-input).

## Keep one image, release the rest

```python
keep = imgs[0].copy()                                      # a copy of its own; the batch buffer can be freed
```

The images of a batch are views of one GPU buffer: keeping one alive keeps the whole buffer. Copy an image if you
keep it and want the rest released.

## Batch size and memory

```python
egj.device_report()["derived_settings"]                    # includes the data one batch may hold on this card
```

A batch needs memory for all its files at once. A batch that would not fit is split automatically; so is one above
64 MB of compressed data or 1,024 files. The data one batch holds follows the card's memory: 16 MB on a 2 GB card,
64 MB from 8 GB up.

## What to expect

```python
imgs = egj.decode(files)                                   # one call for the whole list
```

Batches of 64, 0.25 MP: 6,774 images/s, against 817 with one `decode` per image, 2,906 for libjpeg-turbo on 8
CPU threads and 966 for nvJPEG's batched call through torchvision. One small image per call is the slow way: at
0.25 MP libjpeg-turbo on one thread (1,076) and nvJPEG's batched call both beat it. A batch is where small images
win; at 0.25 MP it takes batches of 16 (4,124) to beat 8 CPU threads, and from 0.5 MP batches of 8 do. One
GTX 1070 + i7-6700, Windows 10: [benchmarks](benchmarks.md#batches).
<!-- numbers: bench/results/20261118/batch.json -->

## Files that gain nothing

```python
imgs = egj.decode(files, fast_dct=True)                    # with fast_dct=True, decoded one by one
```

Files with per-image optimised Huffman tables (some web encoders, Pillow's `optimize=True`) are decoded one by one,
with the same result; so are all files with `fast_dct=True`. A batch on a `Decoder` drops a file staged with
`upload_ahead` ([pipelines](pipelines.md)).

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Small images and the CPU route](small-images.md)
- [Reproduce the benchmark](benchmarks.md#batches)
- [Who it is for](use-cases.md)

**Next:** [Decode into your own GPU work](pipelines.md)
