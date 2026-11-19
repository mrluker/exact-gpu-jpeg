# The tuning, before and after, in paired rounds (2026-11-17)

One GTX 1070 + i7-6700, Windows 10, driver 560.94, CuPy 14.2.0, machine otherwise idle. "Before" is version 0.1 as
it stood on 2026-11-16; "after" is the same code with the per-call changes described in `call_floor.md`. Eight
rounds; in each round both builds ran once, in separate processes, in alternating order. A cell is the median of
one process; "before" and "after" are the medians of the eight; the paired difference is the mean of (after -
before) over the rounds with its standard error; "rounds won" counts the rounds in which the new build was the
faster one. ms per call unless the row says per second. The sized images are one camera photo resized (quality 90,
4:2:0); "30 MP" is twelve camera originals (4:2:2, about 11 MB each); none of them is in the repository. Every
result of both builds was byte-identical to libjpeg-turbo (decode) or Pillow (encode).

| cell | before | after | paired difference (mean +- standard error) | change | rounds won |
|---|---|---|---|---|---|
| decode 64x64 | 1.842 | 0.750 | -1.249 +- 0.172 ms | -59.3% | 8/8 |
| encode 64x64, to bytes | 0.965 | 0.304 | -0.714 +- 0.071 ms | -68.5% | 8/8 |
| encode 64x64, to a file | 2.172 | 1.108 | -1.363 +- 0.427 ms | -49.0% | 8/8 |
| decode 256x256 | 1.651 | 0.992 | -0.789 +- 0.148 ms | -39.9% | 8/8 |
| encode 256x256, to bytes | 0.875 | 0.359 | -0.571 +- 0.059 ms | -59.0% | 8/8 |
| encode 256x256, to a file | 1.703 | 1.019 | -0.864 +- 0.144 ms | -40.1% | 8/8 |
| decode 488x480 | 1.954 | 1.115 | -0.940 +- 0.142 ms | -42.9% | 8/8 |
| encode 488x480, to bytes | 1.062 | 0.406 | -0.708 +- 0.085 ms | -61.8% | 8/8 |
| encode 488x480, to a file | 1.749 | 1.019 | -1.051 +- 0.265 ms | -41.7% | 8/8 |
| decode 640x480 | 1.692 | 1.176 | -0.607 +- 0.061 ms | -30.5% | 8/8 |
| encode 640x480, to bytes | 0.894 | 0.455 | -0.430 +- 0.019 ms | -49.1% | 8/8 |
| encode 640x480, to a file | 1.801 | 1.114 | -0.686 +- 0.081 ms | -38.1% | 8/8 |
| decode 1 MP | 1.947 | 1.456 | -0.531 +- 0.095 ms | -25.2% | 8/8 |
| encode 1 MP, to bytes | 1.065 | 0.738 | -0.351 +- 0.060 ms | -30.7% | 8/8 |
| encode 1 MP, to a file | 2.268 | 1.473 | -0.735 +- 0.198 ms | -35.0% | 7/8 |
| decode 30 MP | 12.568 | 11.743 | -0.746 +- 0.162 ms | -6.6% | 8/8 |
| encode 30 MP, to bytes | 10.121 | 9.612 | -0.182 +- 0.439 ms | -5.0% | 7/8 |
| encode 30 MP, to a file | 11.023 | 11.039 | +0.055 +- 0.305 ms | +0.1% | 4/8 |
| encode 6000x4000, to bytes | 7.915 | 7.615 | -0.350 +- 0.102 ms | -3.8% | 7/8 |
| encode 6000x4000, to a file | 8.754 | 8.520 | -0.181 +- 0.135 ms | -2.7% | 5/8 |
| batch of 64, 0.25 MP, images per second | 7218.500 | 7362.700 | +98.712 +- 101.021/s | +2.0% | 4/8 |
| batch of 64, 1 MP, images per second | 3151.200 | 3165.150 | +2.112 +- 16.722/s | +0.4% | 5/8 |
| batch of 12, 30 MP, photos per second | 81.600 | 87.600 | +5.100 +- 1.457/s | +7.4% | 7/8 |
| 30 MP photos per second, 1 thread | 82.035 | 87.410 | +5.661 +- 0.810/s | +6.6% | 8/8 |
| 30 MP photos per second, 2 threads | 90.280 | 94.330 | +3.711 +- 0.338/s | +4.5% | 8/8 |
| 30 MP photos per second, 3 threads | 97.040 | 100.835 | +3.699 +- 0.415/s | +3.9% | 8/8 |

Launches per call, read from the code: a decode about 45 before and 13 after (14 for a 30 MP photo); an encode
about 18 before, 4 or 5 after for images up to 16,384 blocks (about 1 MP at 4:2:0) and 10 or 11 above that.

What the rows say: every single-image call up to 1 MP is 25 to 69 % shorter, in all eight rounds (one cell: seven
of eight). A 30 MP decode is 0.75 ms shorter. Large encodes and batches of 64 did not move by more than their
error: nothing that was fast got slower.
