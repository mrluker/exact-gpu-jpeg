# Changelog

## 0.1.0 (unreleased)

First version.

Decoding (CUDA through CuPy, byte-identical to libjpeg-turbo 3.1.2's accurate IDCT and fancy upsampling):

- Baseline and extended-sequential Huffman JPEGs, 8-bit, one scan: grayscale, 4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1
  and other integral sampling factors, with or without restart markers, any valid Huffman tables (or the standard
  ones when a file has none), 8- or 16-bit quantisation tables. Everything else is declined with a reason, and
  `decode()` falls back to the CPU decoder.
- Huffman decoding on the GPU by self-synchronising parallel decoding (Weißenberger and Schmidt, ICPP 2018):
  restart markers are not needed.
- Hardened for bad input: strict header checks and a GPU memory estimate before any GPU work, bounded kernels
  (no launch's run time is the file's to decide: a file made so that its Huffman data never synchronises is
  declined after a bounded amount of work, in launches of bounded length),
  error bits instead of crashes, an optional guard-band mode (`EGJ_CANARY=1`); 120,000 fuzzed inputs, 0 findings.
  A file's synchronisation gets at most 8 launches (an eager group of 3 passes and a redo of at most 5), each
  decoding at most 600,000,000 bytes of entropy-coded data; a file that has not settled then is decoded by the CPU
  decoder (flat images with optimised tables, pure white at 4:2:0, very large blank pages, files made for it).
- Output options written by the colour kernel itself: `dtype="float32" | "float16"`, `layout="CHW"`, per-channel
  `scale` and `bias` (for example a normalised model input), bit-identical to the same steps in NumPy.
- `Decoder.decode_async`: queue a decode and return at once; `job.event` lets your own GPU work start the moment
  the pixels are written; `out=` decodes into your buffer.
- `Decoder.upload_ahead`: stage the next file's data on the GPU ahead of its decode (opt-in, for pipelines).
- Opt-in, off by default: decoding the same file again. `RepeatCache` keeps UnHuff copies (a Huffman-free coefficient format) of decoded files in GPU memory
  (`decode(src, cache=cache)`, `Decoder(cache=cache)`): a file seen before is decoded from its copy, the same
  bytes without the entropy decode. A copy is filed under the identity of the JPEG's bytes (a SHA-256 tree);
  `check="full"` (the default) reads and checks the file on every call, `check="light"` does not and can miss a
  change. Underneath: `Decoder(keep_coefs="packed" or "raw")`, `take_coefs()` and `decode_from_handle()` (a
  `CoefHandle`, opaque, valid in the process and on the GPU that made it). Copies live in GPU memory only; there
  is no form of a copy that can be saved or loaded. `keep_coefs` also converts a handle's form in
  `decode_from_handle()`. The packed form's kernels are checked once per GPU at its first use
  (`device_report()["copy_check"]`, `EGJ_COPYCHECK_FAIL=packed` to test); if the check fails, copies are kept in the
  raw form, with one warning, and no decode is affected.
- Opt-in compatibility modes, never the default: `fast_dct=True` (libjpeg-turbo's `JDCT_IFAST`) and
  `fast_upsample=True` (libjpeg-turbo's fast upsampling), each byte-identical to that libjpeg-turbo mode.
- Batches: `decode()` given a list (or `decode_batch`) decodes files that share their Huffman tables and sampling
  factors together by one set of kernel launches, byte-identical to one call each (`docs/batches.md`; figures in
  `docs/benchmarks.md`, "Batches").
- One small image (200,000 pixels or fewer) is decoded on the CPU and uploaded by `decode()`, where that is faster
  than a GPU decode (the same pixels; `cpu_below=0` or `EGJ_SMALL_CPU_PIXELS=0` turns it off).
- PyTorch interop through DLPack (`out="torch"`, torch input tensors), without copies.
- The size below which one image is decoded on the CPU is measured once per machine (three generated images,
  about a quarter of a second, off the caller's decodes), kept between 50,000 and 500,000 pixels and saved in one
  settings file; a measurement that disagrees with itself, or a single one at an end of that range, is not saved;
  `cpu_below` or `EGJ_SMALL_CPU_PIXELS` pins it. A device check before any kernel is compiled, and
  `device_report()` for bug reports.
- Documentation as short task pages under `docs/` (the README is the front door); every Python example on them is
  run by the test suite.
- `timing()` / `EGJ_TIMING=1`: per-call stage times, route and reason for `decode` and `encode`, printed and in
  `info=`; off by default (the marks behind it cost about a microsecond per call when off).
- `self_check()` is also the warm-up call: after it the first decode and the first encode of a process no longer
  load the kernels or run the self-check (`docs/first-call.md`).
- A start-up self-check on the user's own GPU (`self_check()`; it also runs by itself before the first decode,
  batch and encode): built-in cases against stored SHA-256 hashes. A failed check, a kernel that does not
  compile or any CUDA error turns the GPU path off for the process with one warning, and `decode()` then uses
  the CPU decoder.

Encoding (byte-identical to Pillow 13.0.0 / libjpeg-turbo 3.2.0):

- Baseline JPEGs with standard Huffman tables, 4:4:4 / 4:2:2 / 4:2:0, qualities 1–100, ICC profiles and EXIF.
- Only the compressed bytes are downloaded; `encode(..., path=...)` overlaps the download with the file write.
  The file is written directly: if it cannot be opened, nothing is changed; if the encode fails after that, no
  partial file is left.

Speed (the figures, on a GTX 1070, are in `README.md` and `docs/benchmarks.md`):

- Decode speed-ups, each measured on its own and each producing the same bytes: branch-free coefficient writer,
  in-block synchronisation without host waits, 16-byte unstuffing, a write pass from checkpoints, streamed file
  reads, then (the decoder speed loop) 16-byte coefficient loads in the IDCT, a colour kernel specialised per
  subsampling, bit-mask unstuffing and fewer launches before the synchronisation passes.
- A lower fixed cost of one call, for small decodes and small encodes: fewer launches (one state buffer,
  single-kernel prefix sums and DC prediction), 1024-bit chunks and reading in place for small files, a 4-5
  launch encoder for small images, one host wait per encode, the output file opened while the GPU works.
- Encoder: one size readback instead of three host waits, page-locked downloads, bit counts taken while the
  coefficients are in shared memory, word-wise bit writing and byte stuffing, a forward DCT without per-component
  branches and 16-byte coefficient stores.

Tested on 883 local iPhone (X, 7) and Sony camera JPEGs, all byte-identical to libjpeg-turbo, and on JPEGs with
data after the end marker (phone multi-picture files, HDR gain maps), which decode like libjpeg-turbo.
And on a public set anyone can download (`bench/public_set.py`: imagecompression.info's 8-bit test images and
libjpeg-turbo's three benchmark frames, 17 images at 6 settings, not in the repository): 102 of 102 decodes
identical to libjpeg-turbo, 68 of 68 colour encodes byte-identical to Pillow's.

Fixed before release: a CPU decoder standing in for the GPU could return pixels of another mode than the one
asked for. (1) With `fast_dct=True` or `fast_upsample=True` and simplejpeg not installed, a file the GPU path
declined (progressive, damaged, too large for the GPU's memory) was decoded by Pillow in the accurate mode and
returned without a word; it now raises `FastModeUnavailable`, which says that the fast modes need simplejpeg for
such files (with `fallback=False` it returns None as before). (2) With `colorspace="GRAY"` and simplejpeg not
installed, a CPU-decoded colour image was Pillow's conversion from RGB instead of libjpeg-turbo's grayscale (the
luma channel), a few levels off on most pixels; it is now libjpeg-turbo's. With simplejpeg installed (the `cpu`
extra) neither case existed. `decode_cpu()` follows the same rule, and a decoder set with `set_cpu_decoder()` is
passed the fast-mode keywords if it takes them and is not used in a fast mode if it does not.

Also fixed before release: an encode whose data needed more room than its first output buffer was encoded again
with a buffer sized by the same rule, so data made mostly of 0xFF bytes (no such image is known) could have been
encoded again without end. The second attempt now has room for the worst case, and an encode is two attempts at
most.

Also fixed before release: `encode()` could start before the caller's image had arrived on the GPU (CuPy uploads a
NumPy array asynchronously on the caller's stream, the encoder works on its own), giving wrong bytes when the
GPU was busy; it now waits for the caller's stream. Shared table uploads wait on the stream they were made on.

Documentation: `docs/README.md` lists the task pages; `docs/reference.md` has every name, option and environment
variable and the scripts; `docs/exactness.md` has the checks to run on your own files.

Tests and tools: `pytest` (synthetic corpus generated locally, nothing downloaded), libjpeg-turbo SIMD / C
cross-checks (`tools/check_simd.py`, `tools/check_fastdct.py`), decoder and encoder fuzzers, exactness checks on
your own photo folders, benchmarks against libjpeg-turbo, nvJPEG and Pillow, and a CPU-only GitHub Actions
workflow (`.github/workflows/cpu-tests.yml`).
