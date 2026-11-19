**Background**

# How it works, limits, what is not done

The whole decode runs on the GPU, Huffman stage included, with libjpeg's own integer arithmetic after it; this page
gives the design in short, how bad input is handled, the limits, what is not done yet, and the prior work.

## Decoding

```python
import exact_gpu_jpeg as egj
dec = egj.Decoder()
timings = {}
img = dec.decode("photo.jpg", timings=timings)
timings                                          # GPU seconds of 'upload', 'huffman' and 'pixels'
```

Header (Python, `_header.py`, strict) → unstuffing (GPU) → Huffman synchronisation passes and the write pass (GPU,
[huffman-sync.md](huffman-sync.md)) → DC prediction (prefix sums) → IDCT (libjpeg's `jpeg_idct_islow`, with range
checks) → upsampling and colour conversion (libjpeg's fancy upsampling and fixed-point YCbCr tables).

## Self-synchronising Huffman decoding

```python
img = dec.decode(data)                           # no restart markers needed
```

Decoding starts at many positions in the bit stream at once; each decoder falls into step with the true code
boundaries, and the passes repeat until neighbouring chunks agree. Restart markers are not needed; intervals make
decoding faster. The method, checkpoints and error bits in depth: [huffman-sync.md](huffman-sync.md).

## Batches

```python
imgs = dec.decode_batch(files)
dec.stats["batched"]                             # files that shared a batch
```

Batches (`_batch.py`, `_batch_kernels.cu`) are the same kernels with the image looked up first. The images' data is
laid end to end as one bit stream in which every image is one or more independent segments, as restart intervals
are inside one file, so the Huffman synchronisation runs over the whole batch at once; errors are kept per image.

## Encoding

```python
enc = egj.Encoder()
timings = {}
data = enc.encode(img, 90, "420", timings=timings)
timings                                          # GPU seconds of 'dct', 'huffman' and 'stuff_download'
```

Colour conversion, downsampling and edge padding, islow forward DCT and libjpeg-turbo's quantisation by reciprocals,
per block on the GPU → Huffman bit counts per block → prefix sum of bit offsets → every block writes its bits in
parallel → 0xFF00 stuffing → only the compressed bytes are downloaded; the header is built on the host.

## Fast DCT: why blocks are declined

```python
img = egj.decode("photo.jpg", fast_dct=True)    # opt-in: libjpeg-turbo's fast IDCT
```

On x86-64 libjpeg-turbo has one SIMD fast IDCT (`jidctfst-sse2.asm`, 16 bits throughout, so its multiply inputs
wrap beyond 14 bits) and the C one (some sums in 32 bits, a range-limit table that wraps where the SIMD saturates).
The GPU computes exactly and declines every block where one of those 16-bit limits is crossed.

## Safety with bad input

```python
bad = data[: len(data) // 2]                     # a truncated file
img = dec.decode(bad)                            # None, with a reason: never a crash or a guess
dec.reason
```

| where | what is checked |
|---|---|
| before any GPU work | a file over 1 GB is declined before any buffer is made; a header longer than 16 MB, or with more than 4,096 marker segments before the scan, is declined; a file for which page-locked memory cannot be had is declined; every marker length against the file, dimensions (0 and > 65500 rejected; `max_pixels`), component counts and sampling factors, table presence and validity (over-subscribed Huffman codes, DC symbols > 15), scan/frame consistency; the GPU memory a decode needs is estimated against `max_gpu_bytes` before anything is allocated; an entropy-coded segment too short for the image is rejected |
| inside the kernels | the bit reader cannot read past its buffer; every other index is bounded by sizes from the header; invalid codes, runs past coefficient 63, data ending early, overrunning restart intervals and out-of-order restart markers set error bits; every loop is bounded, and none by the file alone: one synchronisation launch decodes at most 600,000,000 bytes of entropy-coded data, and a file gets at most 8 launches (the eager group of 3 passes and a redo of at most 5); a file whose chunks have not synchronised within them is declined (a file can be made that never does; it is a valid JPEG, and `decode()` gives it to the CPU decoder, which raises with the default `strict=True` because the crafted files also carry extraneous bytes, and returns libjpeg-turbo's image with `strict=False`; `bench/sync_worst_launch.py` measures how long its launches run); more than 4,096 marker segments after the scan are declined once the scan is decoded |
| guard bands | with `EGJ_CANARY=1` every device buffer sits between 64 KB guard bands that are checked after each decode (a test mode; the fuzzer runs in it) |
| fuzzing | 120,000 mutated JPEGs in 12 parts of 10 000 (`tools/fuzz_suite.sh`; results in `bench/results/fuzz/summary_20261118.json`, run on the release source), two decoding threads at once, guard bands on: 23,621 decoded on the GPU, every one byte-identical to libjpeg-turbo; 96,379 declined with a reason, and after each a known-good file still decoded exactly. 0 findings: no exception, crash, hang, CUDA error or guard-band write. The encoder parameter fuzzer (`tools/fuzz_encode.py`, 3,000 cases): 2,392 of 2,392 valid cases byte-identical to Pillow, 608 of 608 invalid ones raised ValueError/TypeError, 0 other exceptions |
<!-- numbers: README.md "Safety with bad input" / bench/results/fuzz/summary_20261118.json -->

A decode returns an image or `None` with a reason; `encode` raises `ValueError` or `TypeError` for invalid input.
Earlier runs found two real problems, both fixed and kept as regression tests (`tests/regressions/`): a corrupt DQT
after the scan (libjpeg-turbo fails, we returned an image) and a final 0xFF byte (one pixel differed).

## Limitation: progressive JPEGs

```python
egj.inspect("progressive.jpg")["reason"]         # 'progressive JPEG'
```

Their refinement scans cannot use the self-synchronising trick: parsing a refinement symbol depends on which
coefficients of that block are already non-zero. That needs a different design (or CPU entropy decoding feeding
the GPU pixel stage). Web images are often progressive; camera JPEGs almost never.

## Limitation: a fixed cost per call

```python
imgs = egj.decode(files)                         # a batch removes the per-call cost
```

Each `decode` call costs about 0.58 ms (13 GPU steps and one host round trip; about 1.1 ms at 0.25 MP), a
small `encode` about 0.3 ms, on the GTX 1070 + i7-6700. Batches remove it only for files that share their Huffman
tables and sampling factors, and not with `fast_dct=True`. Header parsing is Python, the limit for very small images.
<!-- numbers: bench/results/20261118/call_floor.json (16x16); bench/results/20261118/small_image.json (0.25 MP, always the GPU) -->

## Limitation: the pixel stage and the encoder

```python
x = egj.decode("photo.jpg", dtype="float16", layout="CHW", scale=1 / 255)   # conversions fused into the decode
```

The pixel stage is bound by memory traffic: a ceiling probe gave at most about 0.05 ms per 30 MP photo for fusing
IDCT, upsampling and colour, so it was not built; conversions after decoding are fused instead. The encoder writes
Pillow's default kind of file only and has no batched call. Batch images are views of one GPU buffer.

## Limitation: one GPU, one operating system

```bash
python bench/reproduce.py                       # the benchmark tables on your GPU, from generated images
```

Tested on one GPU (GTX 1070, compute capability 6.1), Windows 10, CUDA 12 through CuPy 14. Not yet run on Linux,
newer GPUs or CUDA 11. The bytes are libjpeg-turbo's as built for x86-64 (its C and SSE2/AVX2 code); ARM64
builds (NEON) were not analysed. The first
call after a pause is slower on that card: [The first call](first-call.md#the-first-call-after-a-pause).

## Not done yet

```bash
pip install -e ".[cuda12,test]"
pytest                                          # decode + encode exactness, API, malformed input (~10 s)
```

| | |
|---|---|
| **Other machines** | Everything here ran on one GTX 1070 under Windows 10. Not yet run: Linux, a recent GPU (Ampere or newer), a GPU with nvJPEG's hardware engine, CUDA 11. `bench/reproduce.py` and the test suite are the things to run there; results from other cards are welcome. |
| **NVIDIA compute-sanitizer memcheck** | (`compute-sanitizer --tool memcheck python tools/sanitize_run.py`): it reports the GTX 1070 as unsupported under Windows, so it has not run; the guard-band mode and the fuzzing stand in for it. |
| **More real files** | Android phones, HDR gain-map files from real phones, coverage-guided fuzzing as well as the mutation fuzzer. Done so far: 883 iPhone and Sony camera JPEGs and files with data after the end marker, all byte-identical. |
| **Speed** | batches for files with different Huffman tables, a faster header parser (Python is the limit for very small images), batched encoding. |
| **Encoder options** | beyond Pillow's defaults (optimised tables, restart markers, grayscale input). |
| **Progressive JPEGs** | on the GPU (CPU entropy decoding feeding the GPU pixel stage would still be exact). |
| **CI** | CI runs the CPU-only tests (`.github/workflows/cpu-tests.yml`); the GPU tests need a machine with a GPU and run on the test PC. |

## Prior work and credit

**The same idea as theirs.** The Huffman stage is self-synchronising parallel decoding: start decoding at many
positions in the bit stream at once, let each decoder fall into step with the true code boundaries, and repeat until
neighbouring chunks agree. That is the method of André Weißenberger and Bertil Schmidt, "Massively Parallel Huffman
Decoding on GPUs", ICPP 2018, doi:10.1145/3225058.3225076, which the same authors applied to JPEG in "Accelerating
JPEG Decompression on GPUs", arXiv:2111.09219 (2021). The observation that Huffman-coded JPEG data resynchronises
and can therefore be decoded in parallel goes back to Shmuel T. Klein and Yair Wiseman, "Parallel Huffman Decoding
with Applications to JPEG Files", The Computer Journal 46(5), 2003, doi:10.1093/comjnl/46.5.487. This
implementation was written from the papers; no code was taken from them.

**What this library adds.** The goal here is not a faster picture but the reference library's bytes:

- every stage after the Huffman decoding is libjpeg-turbo's own integer arithmetic (IDCT, upsampling, colour), so
  the decoded pixels are equal to libjpeg-turbo's, checked against its AVX2, SSE2 and C code paths, and a file on
  which those disagree with each other is declined instead of guessed;
- restart intervals as independent segments, a write pass that re-decodes from stored checkpoints, error bits for
  every way the data can be wrong, and strict declines with a CPU fallback
  ([docs/huffman-sync.md](huffman-sync.md));
- many files decoded as one stream (batches), and an encoder that writes Pillow's bytes;
- the tests behind the claim: the exactness sets, 120,000 fuzzed inputs, a guard-band mode, and a self-check on
  the user's own GPU ([Exactness](exactness.md), [Safety with bad input](#safety-with-bad-input)).

**Other ways to decode Huffman data in parallel**, none of which this library uses, each of which needs something
an ordinary JPEG file does not carry or a pass this library does not make:

- a sequential scan on the CPU that finds where the coefficient blocks start, then parallel decoding on the GPU:
  Wasuwee Sodsong, Minyoung Jung, Jinwoo Park and Bernd Burgstaller, "JParEnt: Parallel Entropy Decoding for JPEG
  Decompression on Heterogeneous Multicore Architectures", PMAM 2016, doi:10.1145/2883404.2883423;
- a pre-scan that produces boundary markers which later decodes reuse: Lipeng Wang, Qiong Luo and Shengen Yan,
  "Accelerating Deep Learning Tasks with Optimized GPU-assisted Image Decoding", ICPADS 2020,
  doi:10.1109/ICPADS51040.2020.00045;
- a gap array written by the encoder next to the code words: Naoya Yamamoto, Koji Nakano, Yasuaki Ito, Daisuke
  Takafuji, Akihiko Kasagi and Tsuguchika Tabaru, "Huffman Coding with Gap Arrays for GPU Acceleration", ICPP 2020,
  doi:10.1145/3404397.3404429;
- decoding speculatively from every bit position and joining the results: Beau Johnston and Eric McCreath,
  "Parallel Huffman Decoding: Presenting a Fast and Scalable Algorithm for Increasingly Multicore Devices",
  IEEE ISPA 2017.

This library finds the code boundaries of each file from its bytes alone, on the GPU, in every decode. It keeps no
index, markers or positions in the compressed data from one decode of a file to the next. Nothing about a file is
kept unless you use a [RepeatCache](repeat.md). That cache holds the decoded coefficients of the files you decode
through it, in GPU memory, for as long as it lives. It never holds positions or markers in the compressed data.

**Code and libraries.** The arithmetic (IDCT, upsampling, colour conversion, forward DCT, quantisation, Huffman
encoding) is ported from libjpeg-turbo / the Independent JPEG Group's library, which is also what every result is
checked against: see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). **nvJPEG** (NVIDIA) and **GPUJPEG** (CESNET)
are the GPU JPEG codecs to know; neither aims at byte-identical output with libjpeg-turbo, which is the point of
this one, and only nvJPEG was measured here. Built on **CuPy** (kernels compiled with NVRTC, memory pool, DLPack).

## Related

- [Check what "identical" means](exactness.md)
- [Parallel Huffman decoding by self-synchronisation](huffman-sync.md)
- [Who it is for](use-cases.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)

**Next:** [Every name, option and variable](reference.md)
