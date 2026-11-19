# `python bench/reproduce.py` on the GTX 1070 (2026-11-17, machine otherwise idle)

## Environment

- os: Windows-10-10.0.19045-SP0
- cpu: Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
- cpu_threads: 8
- python: 3.12.10
- exact_gpu_jpeg: 0.1.0
- cupy: 14.2.0
- numpy: 2.5.4
- pillow: 13.0.0
- pillow_libjpeg_turbo: 3.2.0
- simplejpeg: 1.9.0
- torchvision: not installed (no nvJPEG column)
- gpu: NVIDIA GeForce GTX 1070
- driver: 560.94
- gpu_memory: 8192 MiB

## What the library found on this GPU (exact_gpu_jpeg.device_report())

- device: 0
- gpu: NVIDIA GeForce GTX 1070, driver 560.94
- compute_capability: 6.1
- total_memory_mb: 8191
- free_memory_mb: 7205
- cuda_runtime: 12.9
- nvrtc: 12.9
- derived_settings: {'memory_budget_share': 0.9, 'batch_data_bytes': 67106816}
- minimum_compute_capability: 5.0
- self_check: {'decode': 'passed', 'batch': 'passed', 'encode': 'passed'}
- gpu_path: on
- cpu_below: {'pixels': 254290, 'source': 'saved'}
- cpu_decoder: simplejpeg 1.9.0 (accurate and fast modes)
- self-check, ms per part (first use, kernel compilation included): {'decode': 366.0, 'batch': 110.2, 'encode': 77.4}

## Decode, one image at a time (ms per image, lower is better)

| size | file | libjpeg-turbo (simplejpeg), CPU | exact-gpu-jpeg | speed-up | decoded on | pixels equal to libjpeg-turbo's |
|---|---|---|---|---|---|---|
| 0.25 MP (616x408) | 35.7 kB | 0.79 | 1.05 | 0.8x | CPU, then uploaded | yes |
| 1 MP (1224x816) | 136.3 kB | 4.53 | 1.32 | 3.4x | GPU | yes |
| 2 MP (1736x1160) | 279.2 kB | 8.88 | 1.90 | 4.7x | GPU | yes |
| 6 MP (3000x2000) | 840 kB | 25.19 | 2.30 | 11.0x | GPU | yes |
| 12 MP (4240x2824) | 1655 kB | 43.99 | 3.11 | 14.1x | GPU | yes |
| 24 MP (6000x4000) | 3359.7 kB | 90.23 | 5.52 | 16.4x | GPU | yes |

The exact-gpu-jpeg column is the default call, with the pixels on the GPU at the end. Below the size limit measured on this machine (cpu_below, above) it decodes on the CPU and uploads, because that is the faster way to get them there; such a row is the CPU's time plus the upload.

## Decode, 64 different images per call (images per second, higher is better)

| size | decode() in a loop | decode_batch() | speed-up | pixels equal |
|---|---|---|---|---|
| 0.25 MP (616x408) | 868 | 8,190 | 9.4x | yes |
| 1 MP (1224x816) | 735 | 3,365 | 4.6x | yes |

## Encode, quality 90, 4:2:0 (ms per image, lower is better)

| size | Pillow (libjpeg-turbo), CPU | exact-gpu-jpeg | speed-up | file equal to Pillow's |
|---|---|---|---|---|
| 0.25 MP (616x408) | 0.98 | 0.42 | 2.3x | yes |
| 1 MP (1224x816) | 6.00 | 0.73 | 8.2x | yes |
| 2 MP (1736x1160) | 11.95 | 1.13 | 10.5x | yes |
| 6 MP (3000x2000) | 34.16 | 2.31 | 14.8x | yes |
| 12 MP (4240x2824) | 76.38 | 4.53 | 16.9x | yes |
| 24 MP (6000x4000) | 151.63 | 8.70 | 17.4x | yes |

Every result was identical to libjpeg-turbo's.
