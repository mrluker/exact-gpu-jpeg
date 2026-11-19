**Get started**

# Install, decode, encode

Install the library with CuPy, decode a JPEG to the GPU, encode it back, and let the library check your card.

## Install

```bash
git clone https://github.com/mrluker/exact-gpu-jpeg
cd exact-gpu-jpeg
pip install "cupy-cuda12x[ctk]"   # CuPy with the CUDA 12 runtime and NVRTC from pip
pip install -e .                  # from the checkout (with simplejpeg, the CPU decoder for the fallback)
```

You need an NVIDIA GPU and its driver, Python 3.10 or newer and CuPy 14. The CUDA Toolkit does not have to be
installed: CuPy's `ctk` extra brings the CUDA runtime and the kernel compiler (NVRTC) as pip packages.
Install from GitHub; not on PyPI yet.

Every number and test result in these docs is from one PC, a GTX 1070 on Windows 10. The library has not been run
on Linux or on newer cards. PyTorch is optional: `out="torch"` and torch input tensors go through DLPack, without copies.

## Cards older than Turing: CUDA 12

```bash
pip install "cupy-cuda12x[ctk]"   # CuPy with the CUDA 12 runtime and NVRTC from pip
```

CUDA 13 can no longer build code for Maxwell, Pascal and Volta cards (GTX 900 and 10 series, Titan V; compute
capability below 7.5), and this library compiles its kernels on your machine. On those cards install
`cupy-cuda12x`; RTX 20 series and newer can use either. Only the CUDA 12 package was tested.

## A CPU decoder for the files the GPU declines

The package installs **simplejpeg** (libjpeg-turbo's TurboJPEG API) as its CPU decoder: it covers every mode and
reports libjpeg-turbo's warnings, so `strict` holds on the CPU route too. Pillow, when it is installed, stands in for
the few files TurboJPEG cannot read (unusual sampling layouts); its pixels are libjpeg-turbo's, but it does not
report the warnings inside the scan, and `info["warnings_checked"]` is then `False`. If simplejpeg is missing (a
platform without its wheels), Pillow decodes every file the GPU path declines, in the accurate mode only:
`fast_dct=True` or `fast_upsample=True` on such a file raises `FastModeUnavailable` instead of returning other
pixels, and `strict` is weaker ([the rule for the CPU stand-in](exactness.md#the-rule-for-the-cpu-stand-in)). See [Check what "identical" means](exactness.md).

## Decode

```python
import exact_gpu_jpeg as egj

img = egj.decode("photo.jpg")                    # CuPy uint8 (H, W, 3), RGB, on the GPU
```

The source can be a path or any bytes-like. More options: [Decode a JPEG to a GPU array](decode.md).

## Encode

```python
data = egj.encode(img, quality=90, subsampling="444", icc_profile=icc, exif=exif)   # bytes, as Pillow writes
n = egj.encode(img, 90, "444", path="out.jpg")     # written to the file (download and write overlap); returns its size
```

`encode` takes a CuPy array, a CUDA torch tensor or a numpy array (uploaded), (H, W, 3) uint8 RGB, C-contiguous.
More: [Encode a GPU image to a JPEG](encode.md).

## See it work on your card

```python
r = egj.self_check()                             # every part (decode, batch, encode, the modes), against stored hashes
r["ok"], r["reason"]
```

`self_check()` runs a few small built-in cases on the GPU and compares them with SHA-256 hashes stored in the
package. It returns `{'ok', 'reason', 'gpu', 'device', 'ms'}`. If a case fails, the GPU path is off for the
process with one `RuntimeWarning`, and `decode()` uses the CPU decoder: see [Run on a GPU it has never seen](new-gpu.md).

## What the first run costs

```python
img = egj.decode("photo.jpg")                    # after self_check(): a few milliseconds
```

The very first run on a machine compiles the kernels (CuPy then keeps them in its disk cache): about 2.8 s for the
first decode, 1.6 s for the first encode and 7.8 s for the first batch (medians of two runs). In later processes
`egj.self_check()` takes 450 to 480 ms (six parts); call it once at start-up if your first image matters. Details: [The first call](first-call.md).
<!-- numbers: bench/results/20261118/first_calls.json (first_run_on_a_machine_ms: decode, encode, batch medians; cases.*.self_check.self_check_ms; GTX 1070 + i7-6700, Windows 10) -->

## Tested configuration

```bash
python bench/reproduce.py
```

The only tested configuration: Windows 10 (64-bit), an NVIDIA GTX 1070 (8 GB, compute capability 6.1) with driver
560.94, Python 3.12, CuPy 14.2.0 with the CUDA 12.9 runtime packages from pip. Linux, newer GPUs and CUDA 11 have
not been run. `bench/reproduce.py` checks your card and prints the benchmark tables: [Reproduce the benchmark](benchmarks.md).

## Related

- [Decode a JPEG to a GPU array](decode.md)
- [Encode a GPU image to a JPEG](encode.md)
- [The first call, warm-up and an idle GPU](first-call.md)
- [When something is wrong](troubleshooting.md)

**Next:** [Decode a JPEG to a GPU array](decode.md)
