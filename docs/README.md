# Documentation

Short pages, one task each, with the code to do it.

| page | what is on it |
|---|---|
| [Install, decode, encode](getting-started.md) | Requirements, the two pip lines (CUDA 12 for cards before Turing), first decode, first encode, `egj.self_check()`, what the first run costs. |
| [Decode a JPEG to a GPU array](decode.md) | From a file or bytes, colour space, straight to a model input (`dtype`, `layout`, `scale`, `bias`), as a torch tensor, what comes back when a file is declined. |
| [Encode a GPU image to a JPEG](encode.md) | To bytes or a file, quality and subsampling, ICC and EXIF, the Pillow call it equals. |
| [Decode many files in one call](batches.md) | A list to `decode`, `decode_batch`, which files share a batch, batch size and memory. |
| [Decode into your own GPU work](pipelines.md) | `decode_async` in front of your kernels, `upload_ahead` for the next file, `encode_async` / `ready=` behind them, the same with PyTorch. |
| [Decode the same file again](repeat.md) | `RepeatCache`: UnHuff copies on the GPU for files decoded more than once, how a copy is known to be current (`check`), `prepare()`, the handle calls underneath and what a handle guarantees. |
| [Check what "identical" means](exactness.md) | The claim in one table, which JPEGs decode on the GPU, what is declined and why, the fast modes, how to check your own files. |
| [Small images and the CPU route](small-images.md) | `cpu_below`, the value measured on your machine, pinning it, the settings file, when a batch is the better answer. |
| [The first call, warm-up and an idle GPU](first-call.md) | Kernel compilation, `self_check()` as the warm-up call, the first large encode, the first call after a pause, how to time. |
| [Run on a GPU it has never seen](new-gpu.md) | The device check, the self-check, `device_report()`, what happens when either fails, what to send in a bug report. |
| [Reproduce the benchmark](benchmarks.md) | `python bench/reproduce.py` and its tables, the camera-photo tables and charts, the nvJPEG method, what was not measured. |
| [When something is wrong](troubleshooting.md) | Symptom, cause and fix, each with the command that shows it; `egj.timing()` to see where a call's time goes. |
| [Who it is for](use-cases.md) | Photo editors, culling a shoot, re-export jobs, image servers, training loops, and who it is not for. |
| [How it works, limits, what is not done](how-it-works.md) | The design in short, safety with bad input, limitations, what is not done yet, prior work and credit. |
| [Every name, option and variable](reference.md) | Module functions, `Decoder`, `Encoder`, every option and environment variable, the scripts. |
| [Parallel Huffman decoding by self-synchronisation](huffman-sync.md) | The Huffman design in depth. |
| [White paper](white-paper.md) | The method, the evidence of exactness, every measurement and the limitations, in one document. |

All numbers in these pages are from one PC: an NVIDIA GTX 1070 and an Intel i7-6700, on Windows 10.

**Next:** [Install, decode, encode](getting-started.md)
