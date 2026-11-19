**Tasks**

# Decode into your own GPU work

Three calls let the GPU keep working while the host reads, waits or writes: `decode_async` in front of your
kernels, `upload_ahead` for the next file, and `encode_async` with `ready=` behind your kernels.

## Your kernels behind a decode: `decode_async`

```python
import exact_gpu_jpeg as egj
import cupy as cp
dec = egj.Decoder()
mine = cp.cuda.Stream(non_blocking=True)

job = dec.decode_async("photo.jpg")           # returns at once; None if the header is declined (dec.reason)
mine.wait_event(job.event)                    # work queued on `mine` from here on starts when the pixels are written
with mine:
    y = my_kernel(job.out)                    # queue it now; nothing has waited on the host yet
img = job.result()                            # waits; None if the data turned out to be one the GPU declines
if img is None or job.redone:
    ...                                       # declined, or decoded a second time after job.event: redo your work
```

Always call `result()` before you trust the pixels: errors in the data are known only once the GPU has finished.
`job.redone` is `True` when the quick synchronisation did not settle and `result()` decoded the image a second
time, with host checks, after `job.event`: the right pixels are then in `job.out` (the same array, your `out=` array
if you gave one), `result()` returns it, and work you queued behind `job.event` read unfinished pixels and must be
run again. Photos do not take this path; large flat images with optimised Huffman tables and very large blank pages
do ([huffman-sync](huffman-sync.md)). When `result()` gives `None`, `job.reason` says why; the decoder's own
`reason` belongs to its latest call. An exception raised by `result()` is raised again by every later call. `out=`
decodes into a C-contiguous array of yours with the output's shape and dtype, aligned to its element size (a
float32 array at an odd address is refused); the output options apply. One job per `Decoder` at a time.

## The next file staged ahead: `upload_ahead`

```python
import threading
decs = [egj.Decoder(), egj.Decoder()]         # two decoders take turns
decs[0].upload_ahead(files[0])
for k, f in enumerate(files):
    t = None
    if k + 1 < len(files):                    # a reader thread stages the next file on the idle decoder
        t = threading.Thread(target=decs[(k + 1) % 2].upload_ahead, args=(files[k + 1],))
        t.start()
    img = decs[k % 2].decode(f)               # no file read, no header parse, no copy at its start
    use(img)
    if t:
        t.join()
```

`upload_ahead(src)` reads the file, parses the header and queues the copy to the GPU; the next `decode(src)` of the
same `src` starts from it (`False` when not staged, always with a `reason`). A relative path is resolved when it is
staged; bytes changed in place after staging are decoded as they were when staged. One staged file per decoder; a
batch drops it.
30 MP camera photos: 9.6 to 9.8 ms per `decode()` instead of 12.9 to 13.5 ms, about 100 photos/s instead of 73 to 77
(one GTX 1070, Windows 10).
<!-- numbers: bench/results/20261118/decode_alone.json (stream: ahead_run0/1 and plain_run0/1, decode_call_ms and photos_per_s; 12 camera photos) -->

## Encoding behind your kernels: `encode_async` and `ready=`

```python
done = cp.cuda.Event()
with mine:
    master, small = my_resize(img)            # the two exports, on your stream
    done.record(mine)                         # recorded after the last kernel that writes them

e1, e2 = egj.Encoder(), egj.Encoder()
j1 = e1.encode_async(master, 95, "444", path="master.jpg", ready=done)     # both scans queued back to back,
j2 = e2.encode_async(small, 90, "444", path="small.jpg", ready=done)       # no host wait so far
n1, n2 = j1.result(), j2.result()             # the size readback, the download and the file write
```

The encoder's stream waits for `ready` on the GPU, so the host waits once, at `result()`, which returns what
`encode()` returns. One exception: building the tables of a size, quality and subsampling waits on the host, so
with `ready=` the host then waits for the producer. That happens the first time they are used on a GPU, and again
once more than 64 other combinations have been used since (the table cache is then emptied). The image must not change until
`result()` has returned; one job per `Encoder` at a time. An exception raised by `result()`, a failed file write
among them, is raised again by every later call. `egj.encode()` given a GPU array uses the encoder of the GPU the
image is on; an `Encoder` refuses an image on another GPU.
Two 488x480 exports to bytes: 0.78 ms by two `encode()` calls, 0.58 ms queued on two encoders (GTX 1070 + i7-6700;
`bench/encode_two_small.py`).
<!-- numbers: bench/results/20261118/encode_two_small.json (median of 9 rounds of 200 pairs) -->

## The same with PyTorch

```python torch
import torch
t = egj.decode("photo.jpg", out="torch")      # a CUDA tensor sharing the memory (DLPack)
ev = torch.cuda.Event()
ev.record()                                   # recorded after the work that writes t
n = egj.encode(t, 90, "444", path="t.jpg", ready=ev)
```

`ready` also takes a recorded `torch.cuda.Event`, and `encode(..., ready=...)` takes it as well as `encode_async`.
Torch tensors go in and come out through DLPack, without copies.

## Without `ready`

```python
n = egj.encode(img, 90, "444", path="plain.jpg")    # waits for this thread's current CuPy stream first
```

Without `ready`, `encode` and `encode_async` wait on the host for the calling thread's current stream before they
queue anything. An image produced on another stream must be finished first, or passed with `ready=`.

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Encode a GPU image to a JPEG](encode.md)
- [Decode many files in one call](batches.md)
- [Every name, option and variable](reference.md) (`DecodeJob`, `EncodeJob`)

**Next:** [Decode the same file again](repeat.md)
