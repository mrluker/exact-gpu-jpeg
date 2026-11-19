# s7: colour kernel writes words (branch `color-words`, 2026-11-15)

The colour kernel stages each 1024-pixel row segment in shared memory and used to copy it out one byte per thread
step. It now copies 16-byte words when the segment's address is 16-byte aligned (4-byte words when 4-aligned, bytes
for the rest), so the same bytes land with ~1/16 of the store instructions.

Measured in a per-stage profile of a decode, edit and encode program (12 camera photos of 6720x4480, GTX 1070, locked, 3 rounds
interleaved): `color` 1.59-1.62 -> **1.50-1.51 ms** per photo.
The read side (fancy upsampling, byte loads) is the rest; `fused-pixels` (s5) already showed that fusing the pixel
stage does not pay.

Exactness: pytest 62 passed, 2 skipped (byte-identical decodes); that program's outputs 336/336 byte-identical
(6720-wide rows take the 16-byte path, the odd 5001-wide test shape the byte path).
