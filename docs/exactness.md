**Tasks**

# Check what "identical" means

What the GPU returns equals a named reference bit for bit, or the file is declined; this page names the
references and shows how to check your own files.

## The claim in one table

```python
import exact_gpu_jpeg as egj
import numpy as np
import simplejpeg
ref = simplejpeg.decode_jpeg(open("photo.jpg", "rb").read(), fastdct=False, fastupsample=False)
assert np.array_equal(egj.decode("photo.jpg", cpu_below=0).get(), ref)    # decoded on the GPU, equal to libjpeg-turbo
```

| | decoding | encoding |
|---|---|---|
| equal to | **libjpeg-turbo 3.1.2** (as bundled in simplejpeg 1.9.0) with the accurate integer IDCT and fancy upsampling, in its AVX2, SSE2 and plain C code paths; Pillow 13.0.0 (libjpeg-turbo 3.2.0) gave the same images | **Pillow 13.0.0** (libjpeg-turbo 3.2.0) with its default settings |
| what is equal | every byte of the decoded pixels (RGB, BGR or GRAY) | every byte of the file: header, tables and compressed data |
| for | 8-bit baseline and extended-sequential Huffman JPEGs with one scan: grayscale or YCbCr; 4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1 and other integral sampling factors; with or without restart markers; any valid Huffman tables | RGB uint8 images, quality 1–100, 4:4:4 / 4:2:2 / 4:2:0, with or without an ICC profile and EXIF |
| declined, never approximated | progressive, arithmetic-coded, lossless, hierarchical, 12-bit, RGB-coded and multi-scan files, fractional sampling factors, and coefficients on which libjpeg-turbo's own code paths disagree. `decode()` then uses the CPU decoder (the reference library itself), so the caller still gets the reference pixels | everything Pillow's defaults do not write (optimised tables, progressive, restart markers, grayscale or CMYK input): not offered |
| files libjpeg-turbo warns about | no end marker or data that ends early, extraneous bytes before a marker, a JFIF version other than 1.x (before or after the scan), an unknown Adobe transform, a malformed ICC profile, any other libjpeg-turbo warning: declined by the GPU path, and the CPU decoder raises for them too, by default; `strict=False` gives libjpeg-turbo's image and names the warning. A few unused bytes at the end of the scan data: libjpeg-turbo reports them for some files and not for others, so the GPU path declines them in both modes and the CPU decoder decides. Where the CPU decoder is Pillow (simplejpeg not installed, or a sampling layout TurboJPEG cannot read), see [the rule for the CPU stand-in](#the-rule-for-the-cpu-stand-in) | |
| rejected | a file libjpeg-turbo rejects is rejected by both routes | |
| outside the claim | CMYK and YCCK (four-component) files: decoded by the CPU decoder with its own colour conversion, because libjpeg-turbo has none to RGB; the reason says so | |
| not checked | other libjpeg-turbo versions; ARM64 (NEON) builds were not analysed: the bytes are libjpeg-turbo's as built for x86-64 (its C and SSE2/AVX2 code) | other Pillow versions |

`strict` is a keyword of `egj.decode()`, `egj.decode_batch()` (and `decode()` given a list), `egj.decode_cpu()` and the `Decoder` constructor (`Decoder(strict=False)`, or set `dec.strict`). With `strict=False` the warning is in `info["warning"]` at the top level (one per file for a list), `Decoder.warning` after `decode()`, `DecodeJob.warning`, and `Decoder.warnings` for `decode_batch()`; it is `None` for a file without one. The GPU path's reason when it declines such a file starts `libjpeg-turbo warns:` and names the warning (for example `Premature end of JPEG file (no end marker)`); for unused bytes at the end of the scan data it starts `unused bytes after the last block of a restart interval`. The CPU decoder's error carries libjpeg-turbo's own text (for example `Corrupt JPEG data: 16 extraneous bytes before marker 0xd9`). With `strict=False` the GPU path itself decodes a file with no end marker, or with a JFIF segment of another version after the scan, and names the warning; the other warned files, and files with unused bytes in the scan, it declines, and at the top level the CPU decoder gives libjpeg-turbo's image and names its warning, if it gave one.

With `strict=False`, a file that ends early gives libjpeg-turbo's image with the missing part grey, as djpeg does
(at the top level, from the CPU decoder; `Decoder.decode()` declines it).

The decoding reference is TurboJPEG's defaults: `simplejpeg.decode_jpeg(..., fastdct=False, fastupsample=False)`,
Pillow's `Image.open(...).convert("RGB")`. The encoding reference is in [Encode](encode.md#the-pillow-call-it-equals).

## Which JPEGs decode on the GPU

```python
egj.inspect("photo.jpg")["supported"]
```

| | |
|---|---|
| colour | YCbCr (3 components), grayscale (1) |
| subsampling | 4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, other integral factors (libjpeg's box upsampling) |
| restart markers | with or without; intervals make decoding faster |
| tables | any valid Huffman tables (optimised or not); the standard tables when a file has none (Motion-JPEG); 8- or 16-bit quantisation tables |
| sizes | 1×1 up to `max_pixels` (default 2^28, about 268 MP) and the GPU memory budget |
| output | RGB, BGR, or GRAY (the Y channel), uint8, height × width × channels |

## What is declined, and why

```python
egj.inspect("progressive.jpg")                   # {'supported': False, 'reason': 'progressive JPEG'}
```

Where libjpeg-turbo's result would depend on the build, the GPU path declines: coefficients on which its 16-bit SIMD
IDCT and its C IDCT give different pixels (no real 8-bit encoder produces them), and markers after the scan other
than APPn/COM. Data libjpeg-turbo decodes only with a warning is declined by default, and the CPU decoder raises for
it too; data libjpeg-turbo rejects is rejected by both routes.

## Declined for the work it would take: Huffman data that does not synchronise

```bash
python bench/sync_pass_limit.py out.json         # the whole decode call at a limit of 64, 5 and 4 passes
```

A file's synchronisation gets the eager group of 3 passes and, if that has not settled, a redo of at most 5 more:
at most 8 launches, each decoding at most 600,000,000 bytes of entropy-coded data. Then the GPU path declines the
file ("Huffman chunks did not synchronise") and `egj.decode` goes to the CPU decoder, which gives the reference's
pixels. (The files made never to synchronise also carry extraneous bytes, which libjpeg-turbo warns about: with
the default `strict=True` the call raises for them, with `strict=False` it returns libjpeg-turbo's image.) Every real photo measured settles in 2 passes. What goes to the CPU decoder for this reason: flat
images coded with optimised Huffman tables, pure white images at 4:2:0, very large blank pages, and files made never
to synchronise. These were slower on the GPU path than on the CPU decoder before the limit as well.

A blank page with a few lines, 4:2:0, default tables (`bench/results/20261118/sync_pass_limit.json`, ms, GTX 1070):

| image | passes needed | CPU decoder | GPU decoder alone (64 passes allowed) | whole call at the limit of 5 |
|---|---|---|---|---|
| 64 MP | 4 | 137 | 93 | 93, on the GPU |
| 100 MP | 5 | 206 | 128 | 129, on the GPU |
| 144 MP | 61 | 329 | 853 | 548, by the CPU decoder |
<!-- numbers: bench/results/20261118/sync_pass_limit.json -->

A file made never to synchronise takes at most about 0.81 s for the whole call (with `strict=False`), where 64
passes took up to 6.0 s.
A known edge: a blank page between 100 and 144 MP may need 6 or more passes and be decoded by the CPU while the GPU
would have been faster.

## The rule for the CPU stand-in

```python
egj.device_report()["cpu_decoder"]               # which CPU decoder stands in, and for which modes
```

A CPU decoder never stands in with pixels of another mode than the one asked for. With only Pillow installed, a fast
mode on a declined file raises `egj.FastModeUnavailable` (with `fallback=False`: None). A decoder of your own
(`egj.set_cpu_decoder(fn)`) cannot be checked by the library: what it returns is your responsibility.

The default CPU decoder is simplejpeg (libjpeg-turbo's TurboJPEG API), which reports libjpeg-turbo's warnings, so
`strict` holds on the CPU route as on the GPU path. Pillow stands in where simplejpeg is not installed, and for the
sampling layouts TurboJPEG has no name for (most of the unusual ones the GPU path decodes). Pillow decodes past
libjpeg-turbo's warnings without saying so. There, `strict=True` still raises for the header warnings (the library
reads the header itself) and for any warning the GPU path found before it declined the file, including unused bytes
in the scan, which then cannot be settled and raise too. It does not see a warning inside the scan of a file the GPU
path declined for another reason (a progressive file, for example). So that no result claims more than was checked,
`info["warnings_checked"]` says which: `True` for the GPU path and for simplejpeg, `False` for pixels from Pillow (the
route reason then says Pillow decoded the file), `None` for a declined file or a decoder of your own; for a list, one
per file. A caller who needs the guarantee can refuse results where it is `False`. The small-image CPU route takes
only files simplejpeg decodes; other small images go to the GPU path. simplejpeg is installed with the package; on
a platform without its wheels, Pillow decodes every file the GPU path declines, with the limits above.

## Opt-in fast modes

```python
img = egj.decode("photo.jpg", fast_dct=True)          # equals simplejpeg fastdct=True
img = egj.decode("photo.jpg", fast_upsample=True)     # equals simplejpeg fastupsample=True
```

Never the default, and not the default bytes. `fast_dct` is libjpeg-turbo's fast integer IDCT (`JDCT_IFAST`);
blocks where its SIMD and C versions disagree are declined. `fast_upsample` repeats chroma instead of interpolating:
a compatibility option, not faster on a GTX 1070. Both reach the default CPU fallback (simplejpeg).

## Check your own files

```bash
python tools/check_dir.py <dir>                    # decoder on a folder of your JPEGs, against libjpeg-turbo
python tools/check_dir.py --batch 16 <dir>         # the same through decode_batch
python tools/check_encode.py <dir>                 # encoder on a folder of your photos, against Pillow
```

Folder checks print counts only, so they can run on private photos. Nothing is downloaded.

## Against every libjpeg-turbo code path

```bash
python tools/check_simd.py work/                   # decoder against libjpeg-turbo's AVX2, SSE2 and C code paths
python tools/check_fastdct.py work/                # the same for fast_dct=True
```

The reference is libjpeg-turbo 3.1.2 as bundled in simplejpeg 1.9.0, in each x86-64 code path: AVX2, SSE2 and plain
C, chosen with `JSIMD_FORCEAVX2=1`, `JSIMD_FORCESSE2=1`, `JSIMD_FORCENONE=1`.

## The evidence

```bash
python tools/fuzz.py work/fuzz --iters 20000       # decoder fuzzer (set EGJ_CANARY=1 for guard bands)
```

| set | files | GPU decoded | equal to libjpeg-turbo |
|---|---|---|---|
| synthetic corpus (`tests/corpus.py`, committed): every subsampling, 1×1 to 1920×1080, qualities 1–100, optimised tables, restart intervals, colour-space markers, no Huffman tables | 347 | 320 (the rest: progressive, RGB-coded, files libjpeg-turbo warns about) | 320, in all three code paths |
| crafted extreme coefficients (`tools/check_simd.py`) | 400 | 34 | 34, in all three code paths; the 365 on which the code paths disagree with each other are all declined, and 1 more where they agree |
| local re-encodes of 12 camera photos (`tools/make_local_corpus.py`, not committed): 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 / 4:1:1 / gray, qualities 50–95, optimised tables, three restart intervals, crops down to 1×1, resized sets of 2–24 MP | 244 | 232 (the rest: progressive) | 232 |
| camera JPEGs (6000×4000 to 7008×4672, 4:2:2), local only | 376 | 376 | 376 |
| phone and other camera JPEGs found locally by EXIF make (`bench/real_photos.py`): iPhone X and 7 (4:2:0 + restart markers), Sony ILCE-7M4, 7M3 and 6300 (4:4:4, 4:2:2, 4:2:0, with and without restart markers); the files stay local (`bench/results/20261118/real_photos.json`) | 883 | 883 | 883 |
| fuzzed inputs (`tools/fuzz.py`) | 120,000 | 23,621 | 23,621 |
<!-- numbers: README.md "Exactness" / "Decoding" table; bench/results/exactness.json; bench/results/fuzz/summary_20261118.json -->

Encoding: every quality 1–100 × 4:4:4 / 4:2:2 / 4:2:0 in the tests, 2,392 random parameter sets
(`tools/fuzz_encode.py`), and 12 camera photos at six settings: 72/72 identical (`tools/check_encode.py`).
<!-- numbers: bench/results/fuzz/summary_20261118.json (encoder.counts: valid, identical); bench/results/exactness.json (row "Encoder: camera photos at 6 settings") -->

Public images anyone can download (imagecompression.info's 8-bit test images and libjpeg-turbo's three benchmark
frames, 17 images at 6 settings, `bench/public_set.py`): 102 of 102 decoded on the GPU and identical to
libjpeg-turbo, 68 of 68 colour encodes byte-identical to Pillow's
([the public set](benchmarks.md#the-public-image-set)).
<!-- numbers: bench/results/20261118/public_set_run3.json (summary) -->

## The evidence for `fast_dct`

```bash
python tools/check_fastdct.py work/                # DEFAULT, AVX2, SSE2 and C, each in its own process
```

| set | files | references agree (SIMD = C) | GPU decoded | equal to libjpeg-turbo (all four modes) | declined although SIMD = C |
|---|---|---|---|---|---|
| synthetic corpus | 347 | 334 | 307 | 307 | 0 (the other 27: progressive, RGB-coded, one file libjpeg-turbo warns about) |
| crafted extreme coefficients | 400 | 29 | 28 | 28 | 1 |
| local re-encodes of camera photos | 228 | 228 | 216 | 216 | 0 (the other 12: progressive) |
| camera photos (6720×4480 4:2:2), local only | 12 | 12 | 12 | 12 | 0 |
<!-- numbers: bench/results/20261118/fastdct.json (counts; tools/check_fastdct.py, 2026-11-19, release source) -->

libjpeg-turbo's SIMD and C fast IDCTs disagree more often than the accurate ones, even in real encodes; the GPU
declines every block where one of their 16-bit limits is crossed, so what it returns equals both
([why](how-it-works.md#fast-dct-why-blocks-are-declined)).

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Encode a GPU image to a JPEG](encode.md)
- [When something is wrong](troubleshooting.md)
- [How it works, limits, what is not done](how-it-works.md)

**Next:** [Small images and the CPU route](small-images.md)
