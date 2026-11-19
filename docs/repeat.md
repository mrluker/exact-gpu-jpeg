**Tasks**

# Decode the same file again

*UnHuff: a Huffman-free coefficient format for exact repeat JPEG decodes.*

Decode a photo once, keep an UnHuff copy in GPU memory (its decoded coefficients, packed with no Huffman
coding), and every later decode of that photo skips the entropy decode.

The pixels are the same as a normal decode's, byte for byte. Nothing on this page happens unless you ask for it:
by default a decode keeps nothing and uses no extra memory. New in 0.1.0. Copies live in GPU memory only, for the
life of the process: the library writes no copy to disk, and has no form of a copy that can be saved, sent or
loaded.

## Keep fast copies in GPU memory

```python
import exact_gpu_jpeg as egj

cache = egj.RepeatCache(memory_bytes=1 << 30)    # up to 1 GB of GPU memory for copies
img = egj.decode("photo.jpg", cache=cache)       # first time: a normal decode, and a copy is kept
img = egj.decode("photo.jpg", cache=cache)       # later: decoded from the copy
dec = egj.Decoder(cache=cache)                   # the same for a Decoder of your own
img = dec.decode("photo.jpg")
```

A decode from a copy runs only the pixel stage (inverse DCT, upsampling, colour), in any colour space and with
every output option. One `RepeatCache` can be shared by decoders on several threads. It does nothing for the first
decode of a photo or for a job that reads every photo once. Whether it saves time on later decodes depends on how
a copy is checked: on the test PC it pays with `check="light"` and not with the default full check ([When it
pays](#when-it-pays)).

## How a copy is known to be current

```python
img = egj.decode("photo.jpg", cache=cache)                   # check="full", the default
img = egj.decode("photo.jpg", cache=cache, check="light")    # for pixels that only go to a screen
```

A copy is filed under the identity of the JPEG's bytes: the SHA-256 of each 64 KiB piece of the file, then the
SHA-256 of the file's length and those digests. So a copy is used only for a file with exactly the bytes it was
made from, also when someone else made the file.

- **`check="full"` (the default):** the file is read and its identity computed on every call. A changed file is
  always decoded afresh, whatever its size and modification time. The price: the file read and the identity are
  paid each time, so what a copy saves is the GPU work and the upload, not the read. On the test PC (i7-6700, 8
  threads) the identity alone cost 6.8 to 9.0 ms for a 30 MP camera photo, and 8.8 ms for the 13.5 MB public-set
  image (`bench/repeat_paths.py`).
- **`check="light"`:** three things are compared with what they were when the file was last read in full: its
  size, its modification time, and its first 64 KiB. If all three match, the copy is used without reading the
  rest of the file. It can miss a change: a file rewritten past its first 64 KiB with its size and modification
  time kept is **not** noticed, and gives the old image. Every full read of a path updates what `check="light"`
  compares with.

JPEG bytes passed in memory are always checked in full. No file is locked or kept open.

## When it pays

![Decoding the same file again, ms per call (lower is better): on 12 camera photos (30 MP, 4:2:2) a normal decode takes 13.1 ms, a decode from the UnHuff copy with check="light" 4.1 ms (3.2 times faster) and with check="full", the default, 18.3 ms; on 14 public-set photos 4.2, 1.4 and 5.5 ms. The full check reads and hashes the whole file, so it is slower than a normal decode.](images/repeat-cache.png)

```python
cache = egj.RepeatCache()
img = egj.decode("photo.jpg", cache=cache, check="light")    # a viewer, a preview, a culling pass
```

In short: with `check="full"` the cache pays below about 2 MP on this PC; with `check="light"` it pays at every
size measured. On camera photos and the public set, all 3.4 MP and up, a decode from a copy with the default full
check was slower than a normal decode on every file: reading the file and computing its identity cost more than
the entropy decode the copy saves. `bench/repeat_paths.py`, GTX 1070 + i7-6700, from a path, the result on
the GPU, medians of 7 rounds of 5 calls, ms per call:

| files | normal decode | from a copy, `check="light"` | from a copy, `check="full"` | the identity alone |
|---|---|---|---|---|
| 12 camera photos, 30 MP, 4:2:2 | 13.1 (12.4 to 15.4) | 4.1 (4.0 to 4.3) | 18.3 (17.3 to 21.2) | 7.3 (6.8 to 9.0) |
| 14 public-set images, quality 95, 4:2:0, 3.4 to 39 MP | 4.2 (2.3 to 15.9) | 1.4 (1.0 to 4.9) | 5.5 (2.6 to 21.7) | 2.6 (1.3 to 8.8) |
<!-- numbers: bench/results/20261118/repeat_paths.json -->

For a 30 MP photo, the copy saves about 9 ms of entropy decode; the full check spends about 7 ms on the identity
and reads the whole file again. So:

- **Use `check="light"`** for viewers, previews, culling and pipelines that do not rewrite files in place with
  their size and modification time kept. There a later decode of the same photo took about a third of the time.
- **Do not use the cache where a full check is needed.** If your program must never show an image from a file
  that has changed since, and files may change in ways the light check misses, decode normally: on this PC that
  is faster than a decode from a copy with the full check. The full check stays the default so that the cache is
  never wrong by default, not because it is fast.

A faster identity (several files hashed at once on the CPU, or on the GPU) is planned for a later release.

## Make the copies ahead of time

```python
n = cache.prepare(["photo.jpg", "a.jpg", "b.jpg"])   # at import, or while idle; returns how many have a copy now
```

`prepare` decodes each file that has no copy yet and keeps one. `decoder=` names the `Decoder` to use (default:
this thread's).

## What is kept, and for how long

```python
cache.stats                                      # used, made, evicted, changed, too_small, over_budget, copies, bytes
cache.drop("photo.jpg")                          # forget one file's copy; True if there was one
cache.clear()                                    # forget all
cache.memory_bytes, cache.min_pixels             # the budget, and the smallest image that is kept
```

A copy is kept in the UnHuff form (`"packed"`) by default: 13 to 54 bytes of GPU memory for every 8x8 block depending on the image, about 27
for a camera photo (about 25 MB for a 30 MP 4:2:2 photo), unpacked on the GPU when it is used.
`RepeatCache(form="raw")` keeps the decoder's arrays as they are instead: 133 bytes a block (about 125 MB for that
photo) and nothing to unpack. When the copies together would pass `memory_bytes`, the least recently used are
dropped. An image below `min_pixels` is decoded as usual and not kept. The default is 0 (every image is kept):
with `check="light"`, a decode from a copy was faster than a normal GPU decode at every size measured, down to
0.1 MP, below the small-image CPU route's threshold (`bench/repeat_sizes.py`, generated photo-like JPEGs, GPU
route forced, from a path, ms per call):

| image | normal decode | from a copy, `check="light"` | from a copy, `check="full"` |
|---|---|---|---|
| 0.1 MP | 1.31 | 0.46 | 0.52 |
| 0.48 MP | 1.44 | 0.50 | 0.88 |
| 1.92 MP | 1.82 | 0.70 | 1.59 |
| 3.84 MP | 2.32 | 0.93 | 2.68 |
| 8.64 MP | 3.33 | 1.40 | 4.99 |
| 24 MP | 6.72 | 2.92 | 7.87 |
<!-- numbers: bench/results/20261118/repeat_sizes.json -->

On these images the full check paid up to about 2 MP and not above; there is no size limit for it, so above that
size it costs time (use `check="light"`, or no cache). Small images usually take the CPU route anyway
(`cpu_below`), where the cache is not used. Copies end with the process; a new process makes them again.

## What does not use the cache

```python
imgs = egj.decode(["a.jpg", "b.jpg"])            # a list: decoded together, as always
job = dec.decode_async("photo.jpg")              # a queued decode
```

Lists (batches), queued decodes, images the CPU route takes (small ones) and files the GPU path declines are
decoded as they always are, and no copy is kept. A list neither uses nor fills a cache, also for the files a batch
decodes one by one. A file libjpeg-turbo warns about is never kept, whatever `strict` is: a strict decoder sharing
the cache could otherwise get its pixels.

## Your own policy: the handle calls

```python
dec = egj.Decoder(keep_coefs="packed")           # or "raw"; off by default (None): nothing is kept
img = dec.decode("photo.jpg")                    # a normal decode
h = dec.take_coefs()                             # a CoefHandle: that decode's coefficients, still on the GPU
h.form, h.nbytes, h.width, h.height              # "packed", the GPU memory it holds, the image's size
again = dec.decode_from_handle(h)                # the same bytes as img
bgr = dec.decode_from_handle(h, "BGR")           # any colour space
x = dec.decode_from_handle(h, dtype="float32", layout="CHW", scale=1 / 255)   # every output option of decode()
h.free()                                         # or drop the handle
h.freed                                          # True
```

`RepeatCache` is built on these, for programs that keep copies by their own rules. `keep_coefs` is `"packed"` (the UnHuff form) or
`"raw"`, and can also be set on an existing decoder (`dec.keep_coefs = "packed"`, `None` to stop). `take_coefs()`
returns the handle of the decoder's last decode, once: the decoder keeps no reference, the handle and its memory are
yours. It returns `None` when nothing was kept: the file was declined or warned about, `keep_coefs` was not set, or
the last call was a batch. After `decode_async`, `take_coefs()` concludes the job first.

`keep_coefs` also applies to `decode_from_handle()`: with a handle of the other form, `take_coefs()` then gives the
coefficients in the decoder's `keep_coefs` form (a raw handle becomes packed, or the reverse).

### What a handle guarantees, and what it does not

- **The same pixels.** `decode_from_handle(h)` returns the same bytes as the decode the handle came from would for the
  same options: any colour space, every output option, and the decoding decoder's own `fast_dct` and
  `fast_upsample` settings, as for a file. Its `max_pixels` applies.
- **This process, this GPU.** A handle lives in GPU memory on the device that made it (`h.device`), and is valid
  only in the process that made it, until it is freed or dropped. Any `Decoder` on that device can decode from
  it; a decoder on another device raises `ValueError`, and so does a freed handle.
- **Opaque.** A handle cannot be saved, sent to another process, turned into bytes or built from bytes; only a
  decode makes one (`egj.CoefHandle(...)` raises `TypeError`). Its layout in memory is not part of the API and may
  change in any release. `form`, `nbytes`, `width`, `height`, `device` and `freed` are all there is to read, and
  they cannot be set.
- **Not a watch on the file.** A handle holds one decode. It does not know whether the file has changed since:
  deciding when it is stale is yours. `RepeatCache` does that for you.
- **No promise of success.** `decode_from_handle` returns `None` (with `dec.reason`) when the GPU path is off,
  memory is short, the image is above the decoder's `max_pixels`, or the copy is found damaged; decode the JPEG the
  normal way then, with the same pixels.

What a handle costs and saves (`bench/copy_forms.py`, GTX 1070, generated photo-like JPEGs at quality 90, from
bytes, medians, ms per call; sizes in MB of GPU memory). The gain is paid for in GPU memory: a packed copy holds
about 2.8 times the JPEG's size, a raw one about 22 times.

| image | JPEG | packed copy | raw copy | decode | decode + keep packed | from a packed handle | from a raw handle |
|---|---|---|---|---|---|---|---|
| 0.1 MP | 0.01 | 0.04 | 0.32 | 1.00 | 1.57 | 0.29 | 0.23 |
| 0.96 MP | 0.13 | 0.37 | 2.99 | 1.24 | 1.75 | 0.43 | 0.32 |
| 3.84 MP | 0.53 | 1.48 | 11.97 | 1.74 | 2.30 | 0.71 | 0.58 |
| 8.64 MP | 1.21 | 3.36 | 26.93 | 2.45 | 3.36 | 1.21 | 1.01 |
| 24 MP | 3.37 | 9.34 | 74.81 | 5.24 | 6.78 | 2.67 | 2.24 |
<!-- numbers: bench/results/20261118/copy_forms.json -->

The kernels behind the packed form are checked once per GPU at their first use; if that check fails, copies are
kept in the raw form and decodes are not affected ([The check of the copy kernels](new-gpu.md#the-check-of-the-copy-kernels)).

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Decode into your own GPU work](pipelines.md)
- [Every name, option and variable](reference.md) (`RepeatCache`, `CoefHandle`)

**Next:** [Check what "identical" means](exactness.md)
