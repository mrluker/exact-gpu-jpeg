# Parallel Huffman decoding by self-synchronisation

The entropy-coded part of a baseline JPEG is one long bit string of variable-length Huffman codes. A code's
meaning depends on everything before it (which table applies, which coefficient comes next), so the obvious
decoder is sequential, and on a CPU it is the most expensive stage of JPEG decoding. Restart markers would cut
the stream into independent pieces, but most camera JPEGs have none, and GPU decoders often keep this stage
on the CPU (nvJPEG, for example, has both CPU and GPU Huffman back ends).

exact-gpu-jpeg decodes the whole stream on the GPU with the self-synchronisation method of Weißenberger and
Schmidt ("Massively Parallel Huffman Decoding on GPUs", ICPP 2018). This note describes the method as
implemented in `_kernels.cu` and `_decoder.py`, including what we added.

## Decoder state

After removing byte stuffing (the `00` after each `FF`) and restart markers, decoding is a deterministic state
machine. Its state at a symbol boundary is

    (bit position, block within the MCU, coefficient index)

packed into 64 bits (`pk` in `_kernels.cu`). Given a state, the next symbol, and so the next state, is fixed.
The block-within-MCU says which component (and so which DC and AC tables) applies; the coefficient index says
whether a DC or an AC symbol comes next.

## Self-synchronisation

Huffman codes self-synchronise: start decoding at a wrong bit position and, after a short while (typically
tens to a few hundred bits for JPEG tables), the decode lands on a true symbol boundary and from then on
follows the true decode exactly. JPEG adds the block/coefficient counters to the state, but those
synchronise too, because the end of a block (EOB, or coefficient 63) is a strong, frequent signal.

The stream is cut into chunks of `S` bits, one GPU thread each. `S` is chosen by the amount of entropy-coded data
(`chunking()`): 1,024 bits under 200,000 bytes, 2,048 under 6,000,000 and 8,192 above, so small files get many short
chunks and large ones fewer long ones.

![Parallel Huffman decoding by self-synchronisation, in four steps: the stream is cut into chunks; each chunk decodes from a guessed state and falls into step with the true code boundaries; passes repeat until every chunk's entry matches the previous chunk's exit; then a write pass decodes every checkpoint piece in its own thread.](images/huffman-sync.svg)


1. **First pass** (`sync_pass`). Every chunk decodes from a guessed state: its own first bit, first block of
   an MCU, DC next. It runs past its end to the first symbol boundary there and records that **exit state**,
   the number of blocks it finished, and its state at its interior **checkpoints**: 7, or 15 in an 8,192-bit chunk.
2. **Synchronising inside a thread block.** A CUDA block holds 128 consecutive chunks. In shared memory, chunk
   *i* compares its entry with chunk *i − 1*'s exit; if they differ it decodes again from that exit, and the
   block repeats this until no exit changes (at most 128 rounds, usually a few). A re-decode stops at the first
   checkpoint whose state equals the one recorded last time (from there on the decode is identical), so most
   re-decodes are short.
3. **Global passes.** The next pass gives each block's first chunk the previous block's last exit from global
   memory and repeats the in-block step. A pass whose predecessor changed nothing exits at once, so passes are
   launched three at a time without the host checking in between; whether they converged is read back with the
   error word at the very end. On camera photos the first pass synchronises everything and the second only
   confirms it. If this eager group of three passes is not enough (no real photo measured needs it; large
   blank pages and flat images with optimised tables do), the decode is redone with host checks, with at most 5
   passes; then the GPU path declines and `decode()` uses the CPU decoder. So a file gets at most 8 launches, each
   limited to 600,000,000 bytes of entropy-coded data decoded
   ([the limit, measured](exactness.md#declined-for-the-work-it-would-take-huffman-data-that-does-not-synchronise)).
4. **Fixed point.** When no exit changes, every chunk's entry is the true state at its position (by induction
   from chunk 0). An exclusive prefix sum of the per-chunk block counts gives each chunk the number of its
   first block.
5. **Write pass** (`decode_write`). After convergence, each chunk's checkpoint states are true states too (a
   re-decode that stopped at a matching checkpoint keeps the earlier, identical later ones), so the write pass
   runs one thread per checkpoint piece (8 or 16 per chunk), each from its true state. A block belongs to the piece in
   which its DC symbol starts: a piece first skips the rest of a block begun before it, and finishes its own
   last block past its end.

## Additions in this implementation

**Restart intervals.** With restart markers (`DRI`), the unstuffing kernel removes the `RSTn` markers,
checks their numbering, and records where each interval starts in the unstuffed bit stream. Chunks are then
cut so that none crosses an interval boundary (`make_chunks`). A chunk that starts an interval has a known
true state (its first bit, first block, DC next, which is exactly the first pass's guess), so it never re-decodes, and
synchronisation can never propagate across intervals. Block numbers come from a segmented prefix sum: interval
*k* starts at block *k* × interval × blocks-per-MCU. DC prediction restarts per interval
(`_dc_prediction`). Restart intervals therefore make decoding faster (shorter chains), but the decoder does not
need them.

**Error detection.** The write pass decodes from true states, so anything wrong there is an error in the file:
a code that is not in the table, an AC run past coefficient 63, a restart interval whose blocks need bits past
its marker, or an interval (or the stream) that ends before its last block. Each sets a bit in an error word;
the image is then declined. The header parser rejects tables libjpeg-turbo would reject (and some it would
accept), and the pixel stage flags coefficient values outside the range where libjpeg-turbo's SIMD and C
IDCTs agree. In every such case the decoder returns None rather than guess what libjpeg-turbo's
error-concealment would produce.

**Bounds and termination.** Every decode loop advances the bit position by at least one bit per symbol and at
most 31, and stops at its piece's end (a block in progress adds at most 64 symbols), so every loop ends. The bit
reader returns zeros for any word past the end of the stream buffer (which also has 4 KB of zero padding), so
no input can make it read outside the buffer. Kernels index every other array with bounds computed on the host
from the header; block numbers are checked against the image's block count before any write. Synchronisation is
capped (in-block rounds by the block size and by the work of a launch, global passes at the eager group of 3 and
a redo of at most 5). A test mode (`EGJ_CANARY=1`) surrounds every
device buffer with 64 KB guard bands and checks them after each decode; the fuzzer runs in that mode.

## Cost

On a GTX 1070, for a 30 MP 4:2:2 camera JPEG (about 10 MB of entropy-coded data, ~20 000 chunks of 4096 bits),
the Huffman stage (unstuffing, synchronisation, write pass, DC prefix sums) went from 9.65 ms to 7.2 ms with the
changes above (bench/results/speed/: branch-free coefficient writer 9.34, in-block synchronisation with
device-side convergence 8.57, vectorised unstuffing 8.45, write pass per checkpoint piece with 4096-bit chunks
7.2). The release took it to 5.33 ms (`s8_loop_decoder_wholeread.json`): the chunk size is chosen per image
(8192 bits at this size, ~10 000 chunks), the table slots are kept in registers, the unstuffing has a fast path
and uses bit masks, and fewer launches run before the synchronisation passes (the last row of
[Decode speed-ups made in this repository](benchmarks.md#decode-speed-ups-made-in-this-repository)). The CPU
(libjpeg-turbo) spends most of its ~200 ms on the same work.

**Next:** [Documentation index](README.md)
