# s6: file read, Huffman decode and GPU encode (branch `speed-io`, merged: no)

2026-11-14, GTX 1070 + i7-6700, every run under the benchmark lock (`bench/with_lock.py`) with the machine otherwise idle at the start.
All changes are byte-identical (checks at the end). Timings: medians; "main" = d34d5b2 measured in the same sitting,
interleaved.

## Getting the file in

* The card runs at **PCIe 3.0 x8** (`nvidia-smi`: link width 8 of 16, also under load), so a page-locked copy moves
  5.5-5.9 GB/s instead of ~12: 2.0 ms for an 11 MB camera JPEG. That is the slot or the board, not software.
* Reading the file (warm OS cache) took 3.7 ms, then the copy 2.0 ms, one after the other. Now the file is read in
  pieces (256 KB for the header, then 2 MB), each copied to the GPU as soon as it is read.
  Decode from a path, one photo at a time, 12 camera photos x 4, 3 rounds: **16.9 -> 15.6 ms**.

| piece sizes (first / rest) | main (whole read) | 256 KB / 2 MB | 256 KB / 4 MB | 2 MB / 2 MB | 64 KB / 1 MB |
|---|---|---|---|---|---|
| decode wall, ms (3 rounds) | 16.5-17.4 | 15.6-16.3 | 15.6-15.7 | 15.5-15.7 | 16.1-16.2 |

## Huffman decode (camera photos: 30 MP 4:2:2, ~11 MB)

Each step measured against the one before, same sitting (GPU events, Huffman stage = unstuff .. DC prediction):

| step | Huffman ms | what changed |
|---|---|---|
| main | 6.6-7.3 | 4096-bit chunks, 7 checkpoints |
| chunk size per image | 6.5-6.6 | 8192-bit chunks for > 6 MB of data (2048 below), checkpoints every 512 (256) bits; the start-finding pass 2.1 -> 1.7 ms |
| table slots in registers | 5.8-5.9 | the table number of each block of the MCU was read from memory for every symbol; now 3 bits per block in two registers |
| symbol entries | 5.6-5.7 | the LUT gives total length, value bits and coefficient step directly |
| unstuffing fast path | **5.4** | plain 16-byte runs (no 0xFF) skip the per-byte rules; output through shared memory as whole words |

Smaller files (local re-encodes, by size of file): < 0.5 MB 1.9-2.2 -> 1.9; 0.5-2 MB 3.6-3.7 -> 2.9;
2-6 MB 4.4-4.6 -> 3.4; > 6 MB 6.3-6.5 -> 5.4 ms.

Tried and not kept: only the table slots in use in shared memory (more blocks fit, but no faster on this card; kept
because it costs nothing and leaves more shared memory to kernels running beside the decode); prefetching the next
word in the bit reader (no change).

Encoder, tried and not kept: colour conversion + downsampling into planes first (coalesced reads), then the DCT
from the planes: exact, but the DCT stage stayed at 2.3 ms (the RGB reads were not what limits it).

## GPU encode (camera photos resized like the exports)

| | main: wall / DCT / Huffman / stuff+download | speed-io |
|---|---|---|
| 3840 q95 4:4:4 | 12.9-13.5 / 2.1 / 5.8-5.9 / 2.1 | **5.7-6.5** / 2.3 / 1.3 / 1.7 |
| 2048 q90 4:4:4 | 4.1-4.4 / 0.65 / 2.3-2.4 / 0.65-0.8 | **1.8-1.9** / 0.7 / 0.4 / 0.4 |
| 1200 q87 4:2:0 | 1.2-1.4 / 0.2 / 0.5 / 0.3 | **1.0-1.1** / 0.25 / 0.25 / 0.3 |

* AC bit counts in `enc_dct` (the coefficients are in shared memory there; a 64-bit mask of non-zero
  coefficients gives each run), DC bits in `enc_dc_bits` once the dummy blocks' DC values are final. The old
  `enc_count` read every coefficient with its own 2-byte load.
* `enc_write`: 16-byte coefficient loads into shared memory, only non-zero coefficients visited, bits gathered in
  a 64-bit accumulator and written as whole words; atomics only for each block's first and last word (two global
  atomics per symbol before).
* Byte stuffing: plain runs skip the per-byte loop; output as whole words through shared memory.
* The file is assembled in one page-locked buffer (one copy instead of three); `encode(..., path=...)` writes it
  piece by piece as it arrives.

## Exactness

* pytest 62 passed, 2 skipped (new: streamed reads with APP segments pushing the scan past the first pieces, piece
  sizes down to 64 bytes; encode to a path with odd piece sizes).
* libjpeg-turbo's 3 code paths (`tools/check_simd.py`): corpus 318 equal to all three, extreme 50, 0 failures.
* 232/232 local re-encodes (guard bands on), 376/376 camera photos.
* Encoder vs Pillow: 72/72 on the camera photos (3840/1500/1200 exports, full size 4:4:4/4:2:2/4:2:0);
  encoder fuzz 2418/2418 identical, 582 invalid inputs raised cleanly.
* Decoder fuzz, guard bands on, 2 x 10 000 inputs (seeds 211, 212; the second from local re-encodes): 4306 decoded
  on the GPU, all identical to libjpeg-turbo; 15 694 declined, every leak check exact; no findings.
