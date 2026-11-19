# The fixed cost of one call: before and after (2026-11-17)

One GTX 1070 + i7-6700, Windows 10, driver 560.94, CuPy 14.2.0. "Before" is version 0.1 as it stood on
2026-11-16 with step timing added; "after" is the same code with the per-call changes (fewer launches, 1024-bit
chunks for small files, small files read in place, a few-launch encoder for small images, one host wait). Both
builds ran in the same sitting with the machine otherwise idle, in alternating processes. Median ms per call; a
range is the spread of the per-process or per-round medians. Every result of both builds was byte-identical to
libjpeg-turbo (decode) or Pillow (encode). Scripts: `bench/call_floor.py`, `bench/one_image_paths.py`,
`bench/small_image.py`, `bench/batch_bench.py`; the sized sets are resized camera photos (quality 90, 4:2:0), the
16x16 and 64x64 images are synthetic.

## Decode, one image per call, always on the GPU

| image | before | after |
|---|---|---|
| 16x16 | 1.62-1.75 | 0.52-0.57 |
| 64x64 | 1.65-1.70 | 0.63-0.64 |
| 0.25 MP | 1.67-1.77 | 1.05-1.12 |
| 0.5 MP | 1.74 | 1.29 |
| 1 MP | 1.87 | 1.39 |
| 2 MP | 2.02 | 1.75 |
| 6 MP | 2.88 | 2.42 |
| camera photos (about 2.6 MB each) | 6.31-6.70 | 5.68-5.76 |

Where a 16x16 decode's time went before: almost all of it was the host queueing about 45 small launches (the
steps "unstuff" 0.26-0.30, "prefix sums" 0.31-0.36, "DC prediction" 0.30-0.35 ms of host time), then a wait of
0.11-0.12 ms. After: about 0.35 ms of queueing for 13 launches and one wait of about 0.2 ms; a launch plus a
readback costs 0.10-0.13 ms on this driver model whatever the work.

## The CPU route for one small image (`cpu_below`)

Default route (CPU decoder below 200,000 pixels, then upload) against always the GPU, after the changes:

| image | default route | always the GPU |
|---|---|---|
| 0.02 MP 4:2:0 | 0.34 (CPU) | 1.04 |
| 0.05 MP 4:2:0 | 0.43 (CPU) | 1.02 |
| 0.1 MP 4:2:0 | 0.62 (CPU) | 1.08 |
| 0.1 MP 4:4:4 | 0.88 (CPU) | 0.77 |
| 0.15 MP 4:2:0 | 0.84 (CPU) | 1.17 |
| 0.2 MP 4:2:0 | 1.16 (CPU) | 1.18 |
| 0.2 MP 4:4:4 | 1.36 (CPU) | 0.84 |
| 0.25 MP 4:2:0 | 1.18 (GPU) | 1.19 |

Before, always-the-GPU was 1.66-1.85 ms at every one of these sizes. The crossover for 4:2:0 files is still at
about 0.2 MP on this machine, so the default stays at 200,000 pixels; 4:4:4 files would already be faster on the
GPU from about 0.1 MP.

## Encode, one image per call (quality 90, 4:2:0)

| image | to bytes: before | after | to a file: before | after |
|---|---|---|---|---|
| 16x16 | 0.85-0.89 | 0.27 | 2.38 | 1.22 |
| 64x64 | 0.85-0.88 | 0.26-0.28 | 2.53 | 1.30 |
| 488x480 | 0.96 | 0.47 | 2.08 | 1.40 |
| 1500x1000 | 1.18 | 0.99 | 3.33 | 2.07 |
| 6000x4000 | 10.5 | 10.8 | 11.8 | 11.9 |

Of a small export to a file, opening, writing and closing the file take about 0.9 ms (0.25-0.3, 0.05-0.1 and
0.45-0.6 ms), which is the operating system's and is the same before and after. Two 488x480 exports of one photo
to bytes: 1.92 before, 0.95 after (0.68 with both queued on two encoders before either is collected).

## What did not change

Batches (64 different files per call, images per second): 0.25 MP 6,389 before and 6,075 after; 1 MP 2,250 and
2,286 (one run each; the CPU rows of the same runs moved by the same few percent). A single image sent through
the batch path is slower than the single-image path at every size (0.25 MP: 2.2 against 1.1 ms), so single images
stay on the single-image path.

## Tried and not kept, or not built

- The host wait mode (CUDA's default, spin, yield, blocking): default, spin and yield are the same within noise;
  blocking is slower. The default already spins on this machine.
- A stream query after the upload, or a copy from pageable memory: no change in call time.
- A 64 kB instead of 1 MB minimum encoder buffer: no change.
- Replaying the launches as one captured CUDA graph: 13 empty launches cost 0.040 ms to queue one by one and 0.003
  ms as a graph; a graph of the real decode would need fixed buffers and kernels that read their sizes from the
  device, for at most about 0.1 ms of a 0.55 ms call. Not built.
- One call that encodes two images: about 0.1 ms a pair better than two calls; `encode_async` on two encoders
  already gives that. Not built.

## Chunk length and read-in-place, by file size (decode ms; 24 files a size, 4 rounds)

| file | copy, 2048-bit chunks | copy, 1024 | in place, 2048 | in place, 1024 (kept) |
|---|---|---|---|---|
| 0.02 MP, 6 kB | 1.02 | 0.96 | 0.98 | 0.89 |
| 0.05 MP, 11 kB | 1.11 | 0.98 | 1.02 | 1.00 |
| 0.1 MP, 19 kB | 1.11 | 1.07 | 1.09 | 1.01 |
| 0.25 MP, 42 kB | 1.22 | 1.14 | 1.18 | 1.06 |
| 0.5 MP, 76 kB | 1.35 | 1.26 | 1.29 | 1.21 |
| 1 MP, 144 kB | 1.52 | 1.41 | 1.47 | 1.40 |
| 2 MP, 278 kB | 1.67 | 1.68 | 1.68 | 1.73 |
| 6 MP, 764 kB | 2.36 | 2.35 | 2.59 | 2.65 |

Hence 1024-bit chunks below 200 kB of entropy-coded data and reading in place up to 128 kB of file. Two
synchronisation passes sufficed for every file at both chunk lengths.

## Paired rounds

The same comparison as eight paired rounds, with the difference per cell, its standard error and the rounds won:
`call_floor_paired.md`. nvJPEG and libjpeg-turbo on the same files after the tuning: `nvjpeg_direct_tuned.json`.
