# Third-party notices

**This software is based in part on the work of the Independent JPEG Group.**

exact-gpu-jpeg's own code is under the MIT license (`LICENSE`). Several algorithms are ported from
libjpeg-turbo 3.1.2 (which includes the Independent JPEG Group's libjpeg; the encoder's output is checked against Pillow 13.0.0, which bundles libjpeg-turbo 3.2.0) so that the output matches
libjpeg-turbo byte for byte. Those ported portions remain covered by the IJG License, under which
libjpeg-turbo distributes them:

- `licenses/README.ijg`: the IJG README with its copyright, no-warranty notice and license terms,
  included unaltered as IJG condition (1) requires;
- `licenses/libjpeg-turbo-LICENSE.md`: libjpeg-turbo's license summary and its Modified (3-clause) BSD
  License, included unaltered.

Neither the Independent JPEG Group nor the libjpeg-turbo Project endorses this software.

## What is ported, and what was changed

As IJG condition (1) asks, the changes to the original code are listed here. No file of libjpeg-turbo is
copied; the routines below were re-implemented from the libjpeg-turbo 3.1.2 sources for CUDA or Python.

| Here | From libjpeg-turbo 3.1.2 | Changes |
|---|---|---|
| `_kernels.cu`: `idct8`, `idct_islow` | `src/jidctint.c` `jpeg_idct_islow` (Thomas G. Lane, Guido Vollbeding; libjpeg-turbo modifications D. R. Commander) | One 8-point pass per CUDA thread, columns then rows through shared memory; dequantisation from zig-zag order; unsigned 32-bit intermediates; outputs clamped by saturation (as libjpeg-turbo's SIMD code does) instead of the range-limit table; added checks that flag inputs where libjpeg-turbo's SIMD and C versions could differ. |
| `_kernels.cu`: `ifast8`, `idct_ifast`; `_decoder.py`: `AANSCALES`, `ifast_table` | `src/jidctfst.c` `jpeg_idct_ifast` (Thomas G. Lane; libjpeg-turbo modifications D. R. Commander), `src/jddctmgr.c` (`aanscales`, IFAST multiplier table) | As for `idct_islow`; exact 32-bit integer arithmetic with checks that flag the blocks where libjpeg-turbo's 16-bit SIMD version and its C version could differ; multiplier table computed in Python. Opt-in (`fast_dct=True`). |
| `_kernels.cu`: `up`, `color`, `color_ep` | `src/jdsample.c` (`h2v1_fancy_upsample`, `h1v2_fancy_upsample`, `h2v2_fancy_upsample`, `h2v1_upsample`, `h2v2_upsample`, `int_upsample`, `jinit_upsampler`), `src/jdmainct.c` (edge context rows), `src/jdcolor.c` (`ycc_rgb_convert`, `build_ycc_rgb_table`, `grayscale_convert`); for `fast_upsample=True`, `src/jdmerge.c` (`h2v1_merged_upsample`, `h2v2_merged_upsample`, `build_ycc_rgb_table`), which libjpeg-turbo uses for 4:2:2 / 4:2:0 RGB output without fancy upsampling | Computed per output pixel instead of per row; the colour tables are evaluated inline from the same fixed-point constants; specialised per subsampling at compile time. Merged upsampling is computed as box upsampling followed by the same colour conversion, which gives the same values. `color_ep` adds our own epilogue (float output, channel planes, scale and bias) after the same conversion. |
| `_decoder.py`: `huffman_arrays` | `src/jdhuff.c` `jpeg_make_d_derived_tbl` | Lookup table of 10 bits instead of 8; arrays laid out for the GPU. |
| `_header.py`: `parse`, `_STD`, `check_huffman`, colour-space rules | `src/jdmarker.c` (`get_sof`, `get_sos`, `get_dht`, `get_dqt`, `get_dri`: what each marker segment must hold), `src/jdinput.c` (`initial_setup`, `per_scan_setup`: size limits, sampling factors, blocks per MCU), `src/jstdhuff.c` (standard tables, from the JPEG standard Annex K), `src/jdhuff.c` (table checks), `src/jdapimin.c` `default_decompress_parms` | Re-implemented in Python; stricter than libjpeg-turbo (anything unusual is declined). |
| `NAT` in `_decoder.py`, `tests/jpegwriter.py` | `src/jutils.c` `jpeg_natural_order` | Without the 16 guard entries. |
| `_enc_kernels.cu`: `conv`, `comp_sample` | `src/jccolor.c` (`rgb_ycc_start`, `rgb_ycc_convert`), `src/jcsample.c` (`fullsize_downsample`, `h2v1_downsample`, `h2v2_downsample`, `expand_right_edge`), `src/jcprepct.c` (`expand_bottom_edge`) | Computed per sample (coefficients inline instead of a table); edge replication as clamped coordinates. |
| `_enc_kernels.cu`: `fdct8`, `quant`, `enc_dct` | `src/jfdctint.c` `jpeg_fdct_islow`, `src/jcdctmgr.c` (`forward_DCT`, `quantize`, `compute_reciprocal`) | One 8-point pass per CUDA thread through shared memory; quantisation by the reciprocal tables of a 16-bit DCTELEM build (as libjpeg-turbo's SIMD builds use). |
| `_enc_kernels.cu`: `enc_dummy_dc` | `src/jccoefct.c` `compress_data` (dummy blocks) | One thread per MCU. |
| `_enc_kernels.cu`: `encode_block`, the bit counts in `enc_dct` (AC) and in `enc_dc_bits` / `enc_plan` (DC), `enc_write`, byte stuffing (`stuff_count`, `stuff_scatter`, `stuff_small`) | `src/jchuff.c` (`encode_one_block`, `flush_bits`, `emit_byte` stuffing, `jpeg_make_c_derived_tbl`) | Bit lengths counted per block, offsets from a prefix sum, each block's bits written in parallel, then byte stuffing in a separate pass. |
| `_kernels.cu`: `unit`, `huff_slow`, `run_cp`, `sync_pass`, `decode_write`, `dc_scan`; `_batch_kernels.cu`: `decode_write_b`, `dc_apply_b` | `src/jdhuff.c` (`decode_mcu_slow`, `HUFF_DECODE`, `HUFF_EXTEND`, the DC prediction with `last_dc_val`, `process_restart`) | What a symbol means (code lookup, sign extension, end-of-block and zero-run codes, the DC difference and its reset at restart markers) follows libjpeg-turbo's decoder; how the stream is cut into chunks and decoded in parallel does not (see "Algorithms from papers"). |
| `_kernels.cu`: `classify16`, `unstuff_count`, `unstuff_scatter`; `_batch_kernels.cu`: `unstuff_count_b`, `unstuff_scatter_b` | `src/jdhuff.c` (`jpeg_fill_bit_buffer`: removal of the 0x00 after 0xFF, markers inside the data), `src/jdmarker.c` (`read_restart_marker`, `next_marker`), `src/jdatasrc.c` (`fill_input_buffer`: the end marker supplied after a file that ends early) | The same rules applied to 16 bytes per thread; anything libjpeg-turbo would only warn about is declined. |
| `_encoder.py`: `quant_tables`, `reciprocals`, `huffman_codes`, `header` | `src/jcparam.c` (`std_luminance_quant_tbl`, `std_chrominance_quant_tbl` from the JPEG standard Annex K, `jpeg_quality_scaling`, `jpeg_add_quant_table`), `src/jcdctmgr.c`, `src/jchuff.c`, `src/jcmarker.c` (marker order) | Re-implemented in Python. The header's layout (JFIF, then EXIF and ICC chunks) follows what Pillow 13.0.0 writes. |

`_batch_kernels.cu` holds the batch versions of the decoder kernels above (`idct_islow_b`, `color_b` and the
Huffman-stage kernels): the same ported arithmetic with the image looked up first, covered in the same way.

libjpeg-turbo's x86-64 SIMD IDCTs (`simd/x86_64/jidctint-avx2.asm`, `jidctint-sse2.asm`, `jidctfst-sse2.asm`, zlib License) were
read to find where they use 16-bit arithmetic; no code was taken from them.

## Copyright of the ported routines

The routines above come from files of libjpeg-turbo 3.1.2 that carry these notices, which the kernel files repeat
in their headers together with a statement that they are modified ports:

- the Independent JPEG Group's software: Copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding
  (`licenses/README.ijg`);
- libjpeg-turbo's modifications to it: Copyright (C) 2009-2025 D. R. Commander and the libjpeg-turbo contributors
  (`licenses/libjpeg-turbo-LICENSE.md`).

The years are those of the two licence texts shipped here; the individual upstream files carry their own (narrower)
year ranges.

## Pillow

The encoder writes what Pillow 13.0.0 writes, and the order and layout of the file header were matched to it by
comparing output: the JFIF segment without density, then EXIF (APP1), then the ICC profile in APP2 segments of at
most 65,519 bytes each with the "ICC_PROFILE" numbering, then the tables. No code of Pillow (MIT-CMU / HPND
License; Copyright 1997-2011 by Secret Labs AB, 1995-2011 by Fredrik Lundh and contributors, 2010 by Jeffrey 'Alex'
Clark and contributors) is included; Pillow is used by the tests and tools as the reference encoder.

## Built-in test images

The package holds a few small JPEGs for its start-up self-check and for timing its two decode routes
(`_selfcheck_data.py`, `_route_data.py`). They are not photographs and contain no third-party material: every
one is generated by `tools/make_selfcheck.py` from formulas and seeded random numbers and encoded by Pillow.

## If you redistribute this package

The CUDA kernels are shipped as source text inside every wheel and sdist (`_kernels.cu`, `_batch_kernels.cu`,
`_enc_kernels.cu`) and are compiled on the user's machine. Passing the package on therefore distributes the ported
source: keep `LICENSE`, this file and the `licenses/` folder with it (they are inside the wheel, under
`*.dist-info/licenses/`), and keep the statement at the top of this file in your documentation.

## Algorithms from papers

The Huffman stage implements the self-synchronising parallel decoding of André Weißenberger and Bertil
Schmidt, "Massively Parallel Huffman Decoding on GPUs", Proceedings of the 47th International Conference on
Parallel Processing (ICPP 2018), doi:10.1145/3225058.3225076. The method is theirs; the code is our own (see
`docs/huffman-sync.md` for what we added: restart intervals, checkpointed re-decoding, error detection).
Related work by the same authors and its origin, not used as code: André Weißenberger and Bertil Schmidt,
"Accelerating JPEG Decompression on GPUs", arXiv:2111.09219 (2021); Shmuel T. Klein and Yair Wiseman, "Parallel
Huffman Decoding with Applications to JPEG Files", The Computer Journal 46(5), 2003, doi:10.1093/comjnl/46.5.487.

## Not legal advice

This summary records what was checked (the libjpeg-turbo 3.1.2 `LICENSE.md` and `README.ijg`, the copyright
headers of the files above). Check it yourself before publishing.
