// exact-gpu-jpeg CUDA kernels. Compiled at run time with NVRTC (CuPy RawModule).
//
// THIS FILE IS A MODIFIED PORT, NOT ORIGINAL libjpeg-turbo CODE. It holds re-implementations for CUDA of routines
// from libjpeg-turbo 3.1.2, which is based on the Independent JPEG Group's software:
//   Copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
//   libjpeg-turbo modifications: Copyright (C) 2009-2025 D. R. Commander and the libjpeg-turbo contributors.
// The ported parts stay under the IJG License (licenses/README.ijg; see also licenses/libjpeg-turbo-LICENSE.md).
// THIRD_PARTY_NOTICES.md names each routine, its upstream file and what was changed.
//
// The pixel stage ports libjpeg-turbo 3.1.2 (jidctint.c jpeg_idct_islow, jdsample.c fancy upsampling,
// jdcolor.c ycc_rgb_convert), which are based on the Independent JPEG Group's software and covered by the IJG
// License (licenses/README.ijg); THIRD_PARTY_NOTICES.md lists what was changed. The Huffman stage follows
// Weissenberger & Schmidt, "Massively Parallel Huffman Decoding on GPUs" (ICPP 2018); see docs/huffman-sync.md.
//
// Every kernel bounds its reads and writes by sizes it is given; bad input sets bits in *err (see ERR_*),
// it never makes a kernel read or write outside its buffers or loop without bound.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

#define LB 10                          // Huffman lookup-table bits; longer codes take the canonical slow path
#define NT 6                           // table slots: up to 3 DC + 3 AC tables

#define ERR_HUFF 1                     // a code that is not in the table
#define ERR_SHORT 2                    // the data (or a restart interval) ends before its last block
#define ERR_COEF 4                     // AC run past coefficient 63
#define ERR_OVERRUN 8                  // a block needs bits past the end of its restart interval
#define ERR_RST 16                     // restart markers missing, extra or out of sequence
#define ERR_RANGE 32                   // coefficients outside the range where every libjpeg-turbo IDCT agrees
#define ERR_DC 64                      // DC value outside int16
#define WARN_EXTRA 128                 // not an error: whole bytes after an interval's last block (libjpeg-turbo warns)

// Huffman tables in dynamic shared memory, only the k slots this image uses (k <= NT), as 32-bit words:
// lut[k][1 << LB] (16-bit: (length << 8) | symbol, 0 = longer than LB bits or invalid), maxc[k][18], voff[k][18],
// hv[k][256] (bytes). Fewer slots, less shared memory per block, more blocks in flight.
#define TABW ((1 << LB) / 2 + 18 + 18 + 64)              // words per slot
extern __shared__ int htab[];
struct Tabs { int k; };

__device__ __forceinline__ u32 bswap(u32 x) { return __byte_perm(x, 0, 0x0123); }

__device__ __forceinline__ void load_tabs(Tabs* T, const int* g, int k) {
    T->k = k;
    for (int i = threadIdx.x; i < k * TABW; i += blockDim.x) htab[i] = g[i];
    __syncthreads();
}

// decoder state: bit position << 16 | block within the MCU << 8 | coefficient index (0 = DC next)
__device__ __forceinline__ u64 pk(u32 pos, int c, int z) { return ((u64)pos << 16) | ((u64)c << 8) | (u64)z; }

// Big-endian bit reader over the unstuffed stream (nw words). The buffer has PAD zero bytes past the data, more
// than any decode of valid data runs past a chunk's end; reads at or past word nw return zeros, so no input can
// make it read outside the buffer.
struct Rd {
    const u32* W; u32 nw; u32 wi; u32 a, b;
    __device__ __forceinline__ u32 word(u32 i) { return i < nw ? bswap(__ldg(W + i)) : 0u; }
    __device__ __forceinline__ void seek(u32 pos) { wi = pos >> 5; a = word(wi); b = word(wi + 1); }
    __device__ __forceinline__ u32 peek(u32 pos) {
        u32 w = pos >> 5;
        while (wi != w) { a = b; wi++; b = word(wi + 1); }
        return __funnelshift_l(b, a, pos & 31);
    }
};

// A code longer than LB bits: returns the symbol (-1 if invalid) and its length.
__device__ __forceinline__ int huff_slow(const Tabs* T, int t, u32 bits, int* len) {
    const int k = T->k;
    const int* maxc = htab + k * ((1 << LB) / 2) + t * 18;
    const int* voff = maxc + k * 18;
    const unsigned char* hv = (const unsigned char*)(htab + k * ((1 << LB) / 2 + 36)) + t * 256;
    int l = LB + 1;
    int code = (int)(bits >> (32 - l));
    while (l <= 16 && code > maxc[l]) { l++; code = (int)(bits >> (32 - l)); }
    if (l > 16) { *len = 16; return -1; }
    *len = l;
    return hv[code + voff[l]];
}

// A decoded symbol: (code length + value bits) | value bits s << 5 | z step << 9 (DC: 1; AC: run + 1 with a value,
// 16 for ZRL, 64 for EOB and other value-less runs), bit 16 = not a valid code (then: length 16, no value, and
// the step DC 1 / AC 64, as before). Codes up to LB bits come straight from the LUT (built by sym_entry() on the
// host); longer ones from the canonical tables.
__device__ __forceinline__ int sym_entry(int sym, int l, bool dc) {
    int s = sym & 15, r = sym >> 4;
    int za = dc ? 1 : s ? r + 1 : r == 15 ? 16 : 64;
    return (l + s) | (s << 5) | (za << 9);
}
__device__ __forceinline__ int hent(const Tabs* T, int t, u32 bits, bool dc) {
    int e = ((const unsigned short*)htab)[(t << LB) | (bits >> (32 - LB))];
    if (e) return e;
    int len;
    int sym = huff_slow(T, t, bits, &len);
    return sym < 0 ? (16 | ((dc ? 1 : 64) << 9) | (1 << 16)) : sym_entry(sym, len, dc);
}

__device__ __forceinline__ int extend(u32 v, int s) { return (int)v < (1 << (s - 1)) ? (int)v + (-1 << s) + 1 : (int)v; }

// Writes one block's coefficients in zig-zag order from registers: the 16 coefficients of zig-zag positions
// 16q..16q+15 are stored as one 32-byte piece when z moves past them, so the buffer needs no clearing. Pieces
// after the last non-zero one are not written; finish() returns how many were (0..4) and the IDCT reads only those.
struct Zw {
    uint4* dst; int q; bool has; u32 w0, w1, w2, w3, w4, w5, w6, w7;
    __device__ __forceinline__ void clear() { w0 = w1 = w2 = w3 = w4 = w5 = w6 = w7 = 0; }
    __device__ __forceinline__ void start(uint4* d) { dst = d; q = 0; has = false; clear(); }
    __device__ __forceinline__ void store() {
        dst[2 * q] = make_uint4(w0, w1, w2, w3);
        dst[2 * q + 1] = make_uint4(w4, w5, w6, w7);
    }
    __device__ __forceinline__ void flush_to(int qn) { while (q < qn) { store(); clear(); q++; } }
    __device__ __forceinline__ void put(int zi, int v) {        // 1 <= zi <= 63, increasing; v != 0
        if ((zi >> 4) != q) flush_to(zi >> 4);
        has = true;
        u32 x = (u32)(unsigned short)(short)v << ((zi & 1) * 16);
        int k = (zi >> 1) & 7;                                     // branch-free: no divergent switch
        w0 |= k == 0 ? x : 0u; w1 |= k == 1 ? x : 0u; w2 |= k == 2 ? x : 0u; w3 |= k == 3 ? x : 0u;
        w4 |= k == 4 ? x : 0u; w5 |= k == 5 ? x : 0u; w6 |= k == 6 ? x : 0u; w7 |= k == 7 ? x : 0u;
    }
    __device__ __forceinline__ int finish() { if (!has) return 0; store(); return q + 1; }
};

// One symbol and its value bits at state (pos, c, z). Returns an ERR_* mask (0 = fine). WRITE: AC coefficients
// go to the block writer, the DC difference to *dcv. The caller handles the end of a block (z >= 64).
// DC and AC share one path so the threads of a warp diverge as little as possible.
template <bool WRITE>
__device__ __forceinline__ int unit(const Tabs* T, Rd& r, u32& pos, int c, int& z, u32 dcp, u32 acp,
        Zw* row, int* dcv) {
    u32 bits = r.peek(pos);
    bool dc = z == 0;
    int h = hent(T, (int)(((dc ? dcp : acp) >> (3 * c)) & 7), bits, dc);
    int e = (h >> 16) ? ERR_HUFF : 0;
    int tot = h & 31, s = (h >> 5) & 15, za = (h >> 9) & 127;
    if (WRITE && s) {
        int v = extend((bits << (tot - s)) >> (32 - s), s);
        int zi = z + za - 1;
        if (dc) *dcv = v;
        else if (zi < 64) row->put(zi, v);
        else e |= ERR_COEF;               // libjpeg-turbo writes it to coefficient 63; we leave that to the CPU
    }
    pos += tot;
    z += za;                              // >= 64: end of block (the caller resets it)
    return e;
}

struct Hdr { int bpm; int dct[10]; int act[10]; };
// The table slot of each block of the MCU, 3 bits per block, in registers: no memory load per symbol.
struct HP { int bpm; u32 dcp, acp; };
__device__ __forceinline__ HP hpack(const Hdr* h) {
    HP p; p.bpm = h->bpm; p.dcp = 0; p.acp = 0;
    for (int k = 0; k < p.bpm; k++) { p.dcp |= (u32)h->dct[k] << (3 * k); p.acp |= (u32)h->act[k] << (3 * k); }
    return p;
}

// Decode from (pos, c, z) to `end`, recording the state at the first symbol boundary past each quarter mark
// (cps) and the blocks finished by then (cpc); n = blocks finished. With cmp, stops as soon as the state at a
// mark equals the one recorded by this chunk's previous decode: from there the two decodes are the same.
__device__ __forceinline__ bool run_cp(const Tabs* T, const HP* h, Rd& r, u32& pos, int& c, int& z, u32 base, int S,
        u32 end, u64* cps, int* cpc, int& n, bool cmp, int NCP) {
    int done = 0;
    const int bpm = h->bpm;
    for (int j = 0; j <= NCP; j++) {
        u32 mark = j < NCP ? min(base + (u32)((j + 1) * (S / (NCP + 1))), end) : end;
        while (pos < mark) {
            unit<false>(T, r, pos, c, z, h->dcp, h->acp, 0, 0);
            if (z >= 64) { z = 0; c = (c + 1 == bpm) ? 0 : c + 1; done++; }
        }
        if (j == NCP) break;
        u64 st = pk(pos, c, z);
        if (cmp && st == cps[j]) {
            int shift = done - cpc[j];
            for (int k = j; k < NCP; k++) cpc[k] += shift;
            n += shift;
            return true;
        }
        cps[j] = st; cpc[j] = done;
    }
    n = done;
    return false;
}

// ---- chunk table: chunks of S bits that never cross a restart interval ----
// seg[k] = first bit of restart interval k (k < nseg), seg[nseg] = total bits. cfirst = exclusive scan of the
// chunk counts per interval (cfirst[nseg] = number of chunks).
extern "C" __global__ void make_chunks(const u32* __restrict__ seg, int nseg, int S, const int* __restrict__ cfirst,
        u32* __restrict__ cstart, u32* __restrict__ cend, int* __restrict__ cseg, int nmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nch = cfirst[nseg];
    if (i >= nch || i >= nmax) return;
    int lo = 0, hi = nseg - 1;                       // last k with cfirst[k] <= i
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (cfirst[mid] <= i) lo = mid; else hi = mid - 1; }
    u32 a = seg[lo] + (u32)(i - cfirst[lo]) * (u32)S;
    cstart[i] = a;
    cend[i] = min(a + (u32)S, seg[lo + 1]);
    cseg[i] = lo;
}

// One synchronisation pass over all chunks (SB threads per CUDA block, one chunk each).
// First pass (prev_changed == 0 pointer): every chunk decodes from a guessed state (its first bit, first block of
// an MCU, DC next), which is the true state for a chunk that starts a restart interval. Later passes: if the
// previous pass changed nothing, exit at once (the GPU decides convergence; the host only checks the flags
// afterwards); otherwise the block's first chunk takes the previous block's last exit state from global memory.
// Then, inside the block, chunk i re-decodes from chunk i-1's exit state (shared memory) until no exit changes:
// most of the synchronisation happens here, without kernel launches. *changed = 1 if the block's last exit
// changed (the next block must see it), or on the first pass of a multi-block grid.
// At most `rounds` such rounds per launch (the host gives fewer to a large file): a real JPEG settles in a few,
// but a file can be made so that its chunks never synchronise, and how long one launch runs must not be the
// file's to decide. A block cut off before it settled sets *changed too, and the next pass carries on from the
// states it left (the fixed point, and so every decoded byte, is the same).
// The work of a launch (what the host's limit counts): a chunk is decoded once before the rounds on the first
// pass, and at most once a round. In round k only chunks k.. of the block can decode (k+1.. on the first pass): a
// chunk decodes in a round only if the chunk before it changed its exit in the round before, the block's first
// chunk reads the previous block once, before the rounds, and on the first pass it does not decode in a round at
// all. So a block decodes at most 128 + sum over k < rounds of (127 - k) chunks, each at most its own bits.
#define SB 128
#ifdef EGJ_COUNT_SYNC                 // test builds only (tests/test_sync_bound.py): the work a launch really did
__device__ u32 egj_sync_decodes;
__device__ u64 egj_sync_bits;
#define SYNC_COUNT(bits) { atomicAdd(&egj_sync_decodes, 1u); atomicAdd(&egj_sync_bits, (u64)(bits)); }
extern "C" __global__ void sync_count_read(u64* out) {
    out[0] = egj_sync_decodes; out[1] = egj_sync_bits; egj_sync_decodes = 0; egj_sync_bits = 0;
}
#else
#define SYNC_COUNT(bits)
#endif
extern "C" __global__ void __launch_bounds__(SB) sync_pass(const u32* __restrict__ W, u32 nw, const int* __restrict__ tabs,
        const Hdr* __restrict__ h, const int* __restrict__ nch_p, const u32* __restrict__ cstart,
        const u32* __restrict__ cend, const int* __restrict__ cseg, const u32* __restrict__ seg, int S,
        u64* __restrict__ st_in, volatile u64* st_out, int* __restrict__ cnt, u64* __restrict__ cps,
        int* __restrict__ cpc, const int* __restrict__ prev_changed, int* __restrict__ changed, int nmax, int ntab,
        int NCP, int rounds) {
    bool first = prev_changed == 0;
    if (!first && *prev_changed == 0) return;
    Tabs T;
    __shared__ u64 so[SB];
    load_tabs(&T, tabs, ntab);
    const HP hp = hpack(h);
    int tid = threadIdx.x;
    int i = blockIdx.x * SB + tid;
    int nch = *nch_p;
    bool valid = i < nch && i < nmax;
    bool fixed = !valid || i == 0 || cstart[i] == seg[cseg[i]];     // entry state known (or no chunk)
    u64 myin = 0, myout = 0, out0 = 0;
    int n = 0;
    Rd r; r.W = W; r.nw = nw;
    if (valid) {
        if (first) {
            u32 pos = cstart[i]; int c = 0, z = 0;
            myin = pk(pos, c, z);
            r.seek(pos);
            run_cp(&T, &hp, r, pos, c, z, cstart[i], S, cend[i], cps + (size_t)i * NCP, cpc + (size_t)i * NCP, n, false, NCP);
            SYNC_COUNT(pos - cstart[i])
            myout = pk(pos, c, z);
        } else {
            myin = st_in[i];
            myout = st_out[i];
            n = cnt[i];
        }
    }
    out0 = myout;
    u64 glob_prev = (!first && valid && tid == 0 && !fixed) ? st_out[i - 1] : 0;
    so[tid] = myout;
    __syncthreads();
    bool more = false;                                               // the block has not settled yet
    for (int it = 0; it < rounds; it++) {
        bool ch = false;
        if (valid && !fixed && (tid > 0 || !first)) {
            u64 want = tid == 0 ? glob_prev : so[tid - 1];
            if (want != myin) {
                myin = want;
                u32 pos = (u32)(want >> 16); int c = (int)((want >> 8) & 255), z = (int)(want & 255);
                r.seek(pos);
                bool same = run_cp(&T, &hp, r, pos, c, z, cstart[i], S, cend[i], cps + (size_t)i * NCP,
                                   cpc + (size_t)i * NCP, n, true, NCP);
                SYNC_COUNT(pos - (u32)(want >> 16))
                u64 e = same ? myout : pk(pos, c, z);
                if (e != myout) { myout = e; ch = true; }
            }
        }
        __syncthreads();
        so[tid] = myout;
        more = __syncthreads_or(ch);
        if (!more) break;
    }
    if (valid) { st_in[i] = myin; st_out[i] = myout; cnt[i] = n; }
    int last = min(SB, nch - blockIdx.x * SB) - 1;                   // the block's last valid chunk
    if (tid == last && (myout != out0 || more || (first && gridDim.x > 1))) *changed = 1;
}

// Write pass, from the true states, (NCP + 1) threads per chunk: after convergence each chunk's entry state and its
// NCP checkpoint states are all true decoder states (a re-decode that stopped at a matching checkpoint keeps the
// earlier, identical later ones), so each piece between checkpoints is decoded by its own thread. A block belongs
// to the piece its DC symbol starts in: a piece first skips the rest of a block begun before it, and finishes its
// own last block past its end. Block numbers: first[i] (blocks finished before chunk i's entry) plus the
// checkpoint's count. Interval k holds blocks [k * seglen, min((k+1) * seglen, nblk)).
extern "C" __global__ void decode_write(const u32* __restrict__ W, u32 nw, const int* __restrict__ tabs,
        const Hdr* __restrict__ h, const int* __restrict__ nch_p, const u32* __restrict__ cstart,
        const u32* __restrict__ cend, const int* __restrict__ cseg, const u32* __restrict__ seg,
        const int* __restrict__ cfirst, const u64* __restrict__ st_in, const u64* __restrict__ cps,
        const int* __restrict__ cpc, int S, const i64* __restrict__ first, i64 seglen, i64 nblk,
        short* __restrict__ coef, int* __restrict__ dcd, unsigned char* __restrict__ nq, int* __restrict__ err, int nmax,
        int ntab, int NCP) {
    Tabs T;
    load_tabs(&T, tabs, ntab);
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gi / (NCP + 1), piece = gi % (NCP + 1);
    if (i >= *nch_p || i >= nmax) return;
    int k = cseg[i];
    i64 lim = min((i64)(k + 1) * seglen, nblk);
    u32 send = seg[k + 1];
    u64 s = piece == 0 ? st_in[i] : cps[(size_t)i * NCP + piece - 1];
    u32 pos = (u32)(s >> 16); int c = (int)((s >> 8) & 255), z = (int)(s & 255);
    i64 B = first[i] + (piece == 0 ? 0 : cpc[(size_t)i * NCP + piece - 1]);
    u32 base = cstart[i];
    u32 end = piece < NCP ? min(base + (u32)((piece + 1) * (S / (NCP + 1))), cend[i]) : cend[i];
    Rd r; r.W = W; r.nw = nw; r.seek(pos);
    const HP hp = hpack(h);
    const int bpm = hp.bpm;
    int e = 0;
    if (z != 0) {                                          // tail of the previous piece's block
        while (z < 64) unit<false>(&T, r, pos, c, z, hp.dcp, hp.acp, 0, 0);
        z = 0; c = (c + 1 == bpm) ? 0 : c + 1; B++;
    }
    Zw row;
    while (pos < end && B < lim) {
        if (B < 0) { e |= ERR_SHORT; break; }
        row.start((uint4*)(coef + B * 64));
        int dcv = 0;
        do { e |= unit<true>(&T, r, pos, c, z, hp.dcp, hp.acp, &row, &dcv); } while (z < 64);
        nq[B] = (unsigned char)row.finish();
        z = 0; c = (c + 1 == bpm) ? 0 : c + 1;
        dcd[B] = dcv;
        if (pos > send) e |= ERR_OVERRUN;
        B++;
        // the interval's last block: a whole byte of its data left unused after it is what libjpeg-turbo reports
        // as extraneous bytes before the next marker (a warning; the image is the same). pos: the bits used.
        if (B == lim && ((pos + 7u) & ~7u) + 8u <= send) e |= WARN_EXTRA;
    }
    if (piece == NCP && i == cfirst[k + 1] - 1 && B < lim) e |= ERR_SHORT;   // last piece of the interval
    if (e) atomicOr(err, e);
}

// ---- unstuffing: drop the 0x00 after each 0xFF and (with restart intervals) the RSTn markers ----
#define TILE 4096
__device__ __forceinline__ bool is_rst(const unsigned char* b, int n, int i, int dri) {  // b[i] starts an RSTn
    return dri && i + 1 < n && b[i] == 0xFF && (b[i + 1] & 0xF8) == 0xD0;
}
__device__ __forceinline__ bool kept(const unsigned char* b, int n, int i, int dri) {
    if (i >= n) return false;
    unsigned char x = b[i];
    if (i > 0 && b[i - 1] == 0xFF && (x == 0 || (dri && (x & 0xF8) == 0xD0))) return false;
    return !is_rst(b, n, i, dri);
}
__device__ __forceinline__ int block_excl_scan(int v, int* tot) {     // 256 threads
    __shared__ int ws[8];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffff, x, o); if (lane >= o) x += y; }
    if (lane == 31) ws[w] = x;
    __syncthreads();
    if (threadIdx.x == 0) { int a = 0; for (int k = 0; k < 8; k++) { int t = ws[k]; ws[k] = a; a += t; } *tot = a; }
    __syncthreads();
    int res = ws[w] + x - v;
    __syncthreads();
    return res;
}
// A thread's 16 bytes (one 16-byte load when they are all inside the buffer) plus the byte before and after.
struct Bytes16 { unsigned char v[18]; };          // v[0] = b[i0 - 1] (0 if none), v[1..16] = b[i0..], v[17] = b[i0 + 16]
__device__ __forceinline__ void load16(const unsigned char* b, int n, int i0, Bytes16& x) {
    x.v[0] = i0 > 0 && i0 - 1 < n ? b[i0 - 1] : 0;
    if (i0 + 16 <= n) {
        uint4 q = *(const uint4*)(b + i0);
        const unsigned char* p = (const unsigned char*)&q;
        for (int k = 0; k < 16; k++) x.v[1 + k] = p[k];
    } else {
        for (int k = 0; k < 16; k++) x.v[1 + k] = i0 + k < n ? b[i0 + k] : 0;
    }
    x.v[17] = i0 + 16 < n ? b[i0 + 16] : 0;
}
// True when the thread's 16 bytes are all inside the buffer and neither they nor the byte before is 0xFF: then all
// 16 are kept, none starts an RST or another marker (the common case, a few 0xFF per 1000 bytes).
__device__ __forceinline__ bool plain16(const Bytes16& x, int n, int i0) {
    if (i0 + 16 > n) return false;
    bool ff = false;
    for (int k = 0; k <= 16; k++) ff |= x.v[k] == 0xFF;
    return !ff;
}
// same rules as kept() / is_rst() above, on the loaded bytes (k = 0..15 is byte i0 + k; valid if i0 + k < n)
__device__ __forceinline__ bool rst16(const Bytes16& x, int n, int i, int k, int dri) {
    return dri && i + 1 < n && x.v[k + 1] == 0xFF && (x.v[k + 2] & 0xF8) == 0xD0;
}
__device__ __forceinline__ bool kept16(const Bytes16& x, int n, int i, int k, int dri) {
    if (i >= n) return false;
    unsigned char c = x.v[k + 1];
    if (i > 0 && x.v[k] == 0xFF && (c == 0 || (dri && (c & 0xF8) == 0xD0))) return false;
    if (c == 0xFF && i + 1 >= n) return false;                 // a final 0xFF is a marker's first byte
    return !rst16(x, n, i, k, dri);
}

// A thread's 16 bytes classified at once (bit k = byte i0 + k, no per-byte branches), by the same rules as kept16(),
// rst16() and the marker test: km = kept bytes, rm = RSTn starts, mm = starts of any other marker (a 0xFF followed
// by anything but 0x00 or an RSTn, or a 0xFF that is the last byte of the data). Bytes past n are 0 in x.
__device__ __forceinline__ void classify16(const Bytes16& x, int n, int i0, int dri, unsigned& km, unsigned& rm, unsigned& mm) {
    u32 ff = 0, z = 0, rs = 0;
    for (int k = 0; k < 18; k++) {
        unsigned c = x.v[k];
        ff |= (u32)(c == 0xFF) << k;
        z |= (u32)(c == 0) << k;
        rs |= (u32)((c & 0xF8) == 0xD0) << k;
    }
    int left = n - i0;                                          // bytes of this thread's 16 inside the data
    u32 valid = left <= 0 ? 0u : left >= 16 ? 0xFFFFu : (1u << left) - 1u;
    u32 lastb = left >= 1 && left <= 16 ? 1u << (left - 1) : 0u;   // the data's last byte
    u32 prev_ff = ff & 0xFFFFu, cur_ff = (ff >> 1) & 0xFFFFu;
    u32 cur_z = (z >> 1) & 0xFFFFu, cur_r = (rs >> 1) & 0xFFFFu, nxt_z = (z >> 2) & 0xFFFFu, nxt_r = (rs >> 2) & 0xFFFFu;
    u32 drop = prev_ff & (cur_z | (dri ? cur_r : 0u));         // the 0x00 of 0xFF00, or the second byte of an RSTn
    u32 rst = dri ? cur_ff & nxt_r & ~lastb : 0u;              // an RSTn starts here
    km = valid & ~drop & ~rst & ~(cur_ff & lastb);
    rm = valid & rst;
    mm = valid & cur_ff & (~nxt_z | lastb) & ~rst;
}

// per tile: kept bytes and RST markers; end = offset of the first other marker's 0xFF (or n)
extern "C" __global__ void unstuff_count(const unsigned char* __restrict__ b, int n, int dri, int* __restrict__ tilecnt,
        int* __restrict__ tilerst, int* __restrict__ end) {
    __shared__ int tot;
    int i0 = blockIdx.x * TILE + threadIdx.x * 16, c = 0;
    Bytes16 x;
    load16(b, n, i0, x);
    // a 0xFF followed by anything but 0x00 (or RSTn with restart intervals) starts a marker; so does a 0xFF that
    // is the last byte of the file (libjpeg-turbo's source manager supplies a fake EOI after it)
    unsigned km, rm, mm;
    classify16(x, n, i0, dri, km, rm, mm);
    c = __popc(km) + (__popc(rm) << 16);
    if (mm) atomicMin(end, i0 + __ffs(mm) - 1);
    block_excl_scan(c, &tot);
    if (threadIdx.x == 0) { tilecnt[blockIdx.x] = tot & 0xFFFF; tilerst[blockIdx.x] = tot >> 16; }
}
// toff/roff = exclusive scans of tilecnt/tilerst. Writes the kept bytes before the end marker, nbits = 8 * their
// count, seg[r + 1] = bit offset of RST number r (r < nseg - 1), found = RSTs before the end marker.
// It also writes the zeros: the rest of its tile's output range (bytes after the end marker) and, in the last tile,
// everything from the last kept byte to nout, so `out` (nout bytes) needs no clearing beforehand.
extern "C" __global__ void unstuff_scatter(const unsigned char* __restrict__ b, int n, int dri,
        const int* __restrict__ toff, const int* __restrict__ roff, const int* __restrict__ end,
        unsigned char* __restrict__ out, u32* __restrict__ nbits, u32* __restrict__ seg, int nseg,
        int* __restrict__ found, int* __restrict__ err, int nout) {
    __shared__ int tot;
    int i0 = blockIdx.x * TILE + threadIdx.x * 16, c = 0;
    Bytes16 x;
    load16(b, n, i0, x);
    __shared__ unsigned char sb[TILE];             // the tile's kept bytes before the end marker, in order
    __shared__ int kw;                             // how many
    if (threadIdx.x == 0) kw = 0;
    unsigned kmask, rmask, mmask;
    classify16(x, n, i0, dri, kmask, rmask, mmask);
    c = __popc(kmask) + (__popc(rmask) << 16);
    int sx = block_excl_scan(c, &tot);             // (its __syncthreads also orders kw = 0 before the atomicMax)
    const int g0 = toff[blockIdx.x];
    int ol = sx & 0xFFFF;                          // offset within the tile
    int r = roff[blockIdx.x] + (sx >> 16);
    int e = *end;
    int last = 0;
    for (int k = 0; k < 16; k++) {
        int i = i0 + k;
        if (i == e) *nbits = (u32)(g0 + ol) * 8u;
        if (i >= n) break;
        if (rmask >> k & 1) {
            if (i < e) {
                if (r < nseg - 1) seg[r + 1] = (u32)(g0 + ol) * 8u;
                if ((x.v[k + 2] & 7) != (r & 7) || r >= nseg - 1) atomicOr(err, ERR_RST);
                atomicMax(found, r + 1);
            }
            r++;
        }
        if (kmask >> k & 1) { if (i < e) { sb[ol] = x.v[k + 1]; last = ol + 1; } ol++; }
    }
    if (last) atomicMax(&kw, last);               // kept bytes before the end marker are a prefix of the tile's
    __syncthreads();
    // write sb[0..kw) to out[g0..): single bytes up to a 4-byte boundary and after the last whole word, whole
    // words between (no other tile writes those bytes)
    const int K = kw;
    int head = min((int)((4 - (g0 & 3)) & 3), K);
    if ((int)threadIdx.x < head) out[g0 + threadIdx.x] = sb[threadIdx.x];
    int nw = (K - head) >> 2;
    u32* dw = (u32*)(out + g0 + head);
    for (int k = threadIdx.x; k < nw; k += blockDim.x) {
        const unsigned char* sp = sb + head + 4 * k;
        dw[k] = sp[0] | (sp[1] << 8) | (sp[2] << 16) | ((u32)sp[3] << 24);
    }
    int t0 = head + 4 * nw;
    if ((int)threadIdx.x < K - t0) out[g0 + t0 + threadIdx.x] = sb[t0 + threadIdx.x];
    int cnt = toff[blockIdx.x + 1] - g0;                // this tile's kept bytes (before and after the end marker)
    for (int k = K + threadIdx.x; k < cnt; k += blockDim.x) out[g0 + k] = 0;
    if (blockIdx.x == gridDim.x - 1)
        for (int k = g0 + cnt + threadIdx.x; k < nout; k += blockDim.x) out[k] = 0;
}

// One block of SCAN1 threads: inclusive prefix sums of a[0..n) and b[0..n) in place (the per-tile counts of
// kept bytes and RST markers: n = tiles + 1 <= 65537), in place of two multi-kernel scans. a[0] and b[0] are taken
// as 0 whatever they hold (and written as 0), so the host need not clear the arrays.
#define SCAN1 1024
__device__ __forceinline__ int block_incl_scan1(int v, int* wsum, int* total) {    // SCAN1 threads
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) { int u = __shfl_up_sync(0xffffffff, v, o); if (lane >= o) v += u; }
    if (lane == 31) wsum[w] = v;
    __syncthreads();
    if (w == 0) {
        int u = wsum[lane];
        for (int o = 1; o < 32; o <<= 1) { int x = __shfl_up_sync(0xffffffff, u, o); if (lane >= o) u += x; }
        wsum[lane] = u;
    }
    __syncthreads();
    int r = v + (w ? wsum[w - 1] : 0);
    *total = wsum[31];
    __syncthreads();
    return r;
}
extern "C" __global__ void __launch_bounds__(SCAN1) scan2_small(int* __restrict__ a, int* __restrict__ b, int n) {
    __shared__ int wa[32], wb[32];
    int ca = 0, cb = 0;
    for (int base = 0; base < n; base += SCAN1) {
        int i = base + threadIdx.x;
        int ta, tb;
        int x = block_incl_scan1(i > 0 && i < n ? a[i] : 0, wa, &ta);
        int y = block_incl_scan1(i > 0 && i < n ? b[i] : 0, wb, &tb);
        if (i < n) { a[i] = x + ca; b[i] = y + cb; }
        ca += ta; cb += tb;
    }
}
// The chunk table in one block of SCAN1 threads, for nseg and nmax small enough (the host decides): the same steps
// as fix_segs, seg_chunks (scan.cu), the scan of the chunk counts and make_chunks, plus cseg = 0 past the chunks
// (which the host otherwise clears first). cfirst: nseg + 1 ints. seg[0] = 0 is written here (seg needs no clearing:
// unstuff_scatter writes seg[1..found], this kernel the rest).
extern "C" __global__ void __launch_bounds__(SCAN1) chunk_table(u32* __restrict__ seg, int nseg,
        const int* __restrict__ found, const u32* __restrict__ nbits, int* __restrict__ err, int S,
        int* __restrict__ cfirst, u32* __restrict__ cstart, u32* __restrict__ cend, int* __restrict__ cseg, int nmax) {
    __shared__ int wsum[32];
    for (int k = threadIdx.x; k <= nseg; k += SCAN1) {               // fix_segs
        if (k == 0) { seg[0] = 0; continue; }
        if (k == nseg || k > *found) seg[k] = *nbits;
        if (k == 1 && *found != nseg - 1) atomicOr(err, ERR_RST);
    }
    __syncthreads();
    if (threadIdx.x == 0) cfirst[0] = 0;
    int run = 0;
    for (int base = 0; base < nseg; base += SCAN1) {                  // seg_chunks + inclusive scan into cfirst[1..]
        int i = base + threadIdx.x, c = 0;
        if (i < nseg) {
            i64 len = (i64)seg[i + 1] - (i64)seg[i];
            i64 q = (len + S - 1) / S;
            if ((len + S - 1) < 0 && (len + S - 1) % S) q -= 1;
            c = (int)(q < 1 ? 1 : q);
        }
        int tot;
        int x = block_incl_scan1(c, wsum, &tot);
        if (i < nseg) cfirst[i + 1] = x + run;
        run += tot;
    }
    __syncthreads();
    int nch = cfirst[nseg];
    for (int i = threadIdx.x; i < nmax; i += SCAN1) {                 // make_chunks; cseg = 0 past the chunks
        if (i >= nch) { cseg[i] = 0; continue; }
        int lo = 0, hi = nseg - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (cfirst[mid] <= i) lo = mid; else hi = mid - 1; }
        u32 a = seg[lo] + (u32)(i - cfirst[lo]) * (u32)S;
        cstart[i] = a;
        cend[i] = min(a + (u32)S, seg[lo + 1]);
        cseg[i] = lo;
    }
}

// The write pass's block numbers in one block of SCAN1 threads (chunk tables small enough for chunk_table):
// first[i] = blocks finished before chunk i's entry = cseg[i] * seglen + (blocks counted before chunk i) - (blocks
// counted before the first chunk of i's interval). cnt[0..nch) becomes its inclusive scan, in place; chunks past
// nch are not read (cnt needs no clearing). The counts fit 32 bits: a chunk of S <= 8192 bits finishes at most S
// blocks and there are at most 131072 chunks.
extern "C" __global__ void __launch_bounds__(SCAN1) first_blocks(int* __restrict__ cnt, const int* __restrict__ nch_p,
        const int* __restrict__ cseg, const int* __restrict__ cfirst, i64 seglen, i64* __restrict__ first, int nmax) {
    __shared__ int wsum[32];
    const int nch = min(*nch_p, nmax);
    int run = 0;
    for (int base = 0; base < nch; base += SCAN1) {
        int i = base + threadIdx.x, tot;
        int x = block_incl_scan1(i < nch ? cnt[i] : 0, wsum, &tot);
        if (i < nch) cnt[i] = x + run;
        run += tot;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nch; i += SCAN1) {
        int k = cseg[i], f = cfirst[k];
        first[i] = (i64)k * seglen + (i64)(i ? cnt[i - 1] : 0) - (i64)(f ? cnt[f - 1] : 0);
    }
}

// The state words' start values (in place of an upload: a second small copy to the GPU holds the stream up far
// longer than a launch): zeros, and the length of the data as the end.
extern "C" __global__ void state_init(int* __restrict__ st, int n, int ne) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) st[i] = i == 1 ? ne : 0;
}

// DC values from differences, in place: per component a running sum over its blocks in scan order, starting again
// at every restart interval (every dri MCUs; dri = 0: never); 32-bit sums that wrap, as a cumulative sum over the
// whole array. comp: the component of each block of the MCU, 2 bits each. Each thread takes a run of `per` MCUs
// and a CUDA block SCAN1 runs: the thread sums its run, a scan over the block's threads gives each thread its
// starting values, then each thread writes its run. A small image is one CUDA block and one launch (phase 1).
// Otherwise phase 0 only writes each CUDA block's sums to bs (4 words: the three sums since the block's last
// restart, or over the whole block, and whether it had a restart) and phase 1 starts each block from the blocks
// before it.
struct DcS { u32 a, b, c, r; };                            // sums since the last restart; r: there was a restart
__device__ __forceinline__ DcS dc_then(const DcS& left, DcS x) {    // the sums after left's MCUs, then x's
    if (!x.r) { x.a += left.a; x.b += left.b; x.c += left.c; x.r = left.r; }
    return x;
}
__device__ __forceinline__ DcS dc_shfl(const DcS& x, int o) {
    DcS y;
    y.a = __shfl_up_sync(0xffffffff, x.a, o); y.b = __shfl_up_sync(0xffffffff, x.b, o);
    y.c = __shfl_up_sync(0xffffffff, x.c, o); y.r = __shfl_up_sync(0xffffffff, x.r, o);
    return y;
}
extern "C" __global__ void __launch_bounds__(SCAN1) dc_scan(int* __restrict__ d, int nmcu, int bpm, u32 comp, int dri,
        int per, u32* __restrict__ bs, int phase) {
    __shared__ DcS wt[32];
    __shared__ DcS cin;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const i64 g = (i64)blockIdx.x * SCAN1 + t;
    const int m0 = (int)min(g * per, (i64)nmcu), m1 = (int)min((i64)m0 + per, (i64)nmcu);
    u32 s0 = 0, s1 = 0, s2 = 0, rs = 0;
    for (int m = m0; m < m1; m++) {
        if (dri && m % dri == 0) { s0 = s1 = s2 = 0; rs = 1; }
        const int* p = d + (size_t)m * bpm;
        for (int k = 0; k < bpm; k++) {
            u32 v = (u32)p[k]; int ci = (comp >> (2 * k)) & 3;
            s0 += ci == 0 ? v : 0u; s1 += ci == 1 ? v : 0u; s2 += ci == 2 ? v : 0u;
        }
    }
    DcS x = {s0, s1, s2, rs};
    for (int o = 1; o < 32; o <<= 1) { DcS y = dc_shfl(x, o); if (lane >= o) x = dc_then(y, x); }
    if (lane == 31) wt[w] = x;
    DcS e = dc_shfl(x, 1);                                 // the sums before this thread's run, within its warp
    if (lane == 0) e = DcS{0, 0, 0, 0};
    __syncthreads();
    if (w == 0) {
        DcS u = wt[lane];
        for (int o = 1; o < 32; o <<= 1) { DcS y = dc_shfl(u, o); if (lane >= o) u = dc_then(y, u); }
        wt[lane] = u;
        if (t == 0) {
            DcS c = {0, 0, 0, 0};
            if (phase)
                for (u32 j = 0; j < blockIdx.x; j++) {
                    DcS q = {bs[4 * j], bs[4 * j + 1], bs[4 * j + 2], bs[4 * j + 3]};
                    c = dc_then(c, q);
                }
            cin = c;
        }
    }
    __syncthreads();
    if (!phase) {
        if (t == 0) {
            u32* q = bs + 4 * (size_t)blockIdx.x;
            q[0] = wt[31].a; q[1] = wt[31].b; q[2] = wt[31].c; q[3] = wt[31].r;
        }
        return;
    }
    if (w) e = dc_then(wt[w - 1], e);
    e = dc_then(cin, e);
    s0 = e.a; s1 = e.b; s2 = e.c;
    for (int m = m0; m < m1; m++) {
        if (dri && m % dri == 0) s0 = s1 = s2 = 0;
        int* p = d + (size_t)m * bpm;
        for (int k = 0; k < bpm; k++) {
            u32 v = (u32)p[k]; int ci = (comp >> (2 * k)) & 3;
            s0 += ci == 0 ? v : 0u; s1 += ci == 1 ? v : 0u; s2 += ci == 2 ? v : 0u;
            p[k] = (int)(ci == 0 ? s0 : ci == 1 ? s1 : s2);
        }
    }
}

// intervals whose marker was not found get length 0 (then ERR_SHORT); seg[nseg] = nbits
extern "C" __global__ void fix_segs(u32* __restrict__ seg, int nseg, const int* __restrict__ found,
        const u32* __restrict__ nbits, int* __restrict__ err) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nseg || k == 0) return;
    if (k == nseg || k > *found) seg[k] = *nbits;
    if (k == 1 && *found != nseg - 1) atomicOr(err, ERR_RST);
}

// ---- pixels ----
#define CONST_BITS 13
#define PASS1_BITS 2
#define F_0_298 2446
#define F_0_390 3196
#define F_0_541 4433
#define F_0_765 6270
#define F_0_899 7373
#define F_1_175 9633
#define F_1_501 12299
#define F_1_847 15137
#define F_1_961 16069
#define F_2_053 16819
#define F_2_562 20995
#define F_3_072 25172
#define DESCALE(x, n) (((x) + (1 << ((n) - 1))) >> (n))
#define NOT16(v) ((u32)((v) + 32768) > 65535u)

// jidctint.c jpeg_idct_islow, one 8-point pass. Also flags the 16-bit sums that libjpeg-turbo's SIMD IDCTs form
// (in0 +- in4, in1 + in5, in3 + in7): when one of them, an input or an output leaves int16, the SIMD and C
// versions can differ, and we do not guess which one the caller has.
__device__ __forceinline__ bool idct8(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int* o, int shift) {
    bool bad = NOT16(i0 + i4) | NOT16(i0 - i4) | NOT16(i1 + i5) | NOT16(i3 + i7);
    // unsigned arithmetic: the intermediate sums may wrap, the results do not (inputs are within int16)
    u32 z2 = i2, z3 = i6;
    u32 z1 = (z2 + z3) * F_0_541;
    u32 tmp2 = z1 + z3 * (u32)(-F_1_847);
    u32 tmp3 = z1 + z2 * F_0_765;
    z2 = i0; z3 = i4;
    u32 tmp0 = (z2 + z3) << CONST_BITS;
    u32 tmp1 = (z2 - z3) << CONST_BITS;
    u32 tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    tmp0 = i7; tmp1 = i5; tmp2 = i3; tmp3 = i1;
    z1 = tmp0 + tmp3; z2 = tmp1 + tmp2; z3 = tmp0 + tmp2; u32 z4 = tmp1 + tmp3;
    u32 z5 = (z3 + z4) * F_1_175;
    tmp0 = tmp0 * F_0_298; tmp1 = tmp1 * F_2_053; tmp2 = tmp2 * F_3_072; tmp3 = tmp3 * F_1_501;
    z1 = z1 * (u32)(-F_0_899); z2 = z2 * (u32)(-F_2_562); z3 = z3 * (u32)(-F_1_961); z4 = z4 * (u32)(-F_0_390);
    z3 += z5; z4 += z5;
    tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4;
    u32 rnd = 1u << (shift - 1);
    o[0] = (int)(tmp10 + tmp3 + rnd) >> shift; o[7] = (int)(tmp10 - tmp3 + rnd) >> shift;
    o[1] = (int)(tmp11 + tmp2 + rnd) >> shift; o[6] = (int)(tmp11 - tmp2 + rnd) >> shift;
    o[2] = (int)(tmp12 + tmp1 + rnd) >> shift; o[5] = (int)(tmp12 - tmp1 + rnd) >> shift;
    o[3] = (int)(tmp13 + tmp0 + rnd) >> shift; o[4] = (int)(tmp13 - tmp0 + rnd) >> shift;
    return bad;
}

__constant__ unsigned char ZZ[64] = {@ZZ@};     // natural index -> zig-zag position

// blk: per block-in-MCU (component, dx, dy); hs/vs per component (blocks per MCU); planes: per component
// base offset and pitch; mcux = MCUs per row; q: [3][64] natural-order quantisers; coef in zig-zag order.
struct Pix { int bpm; int comp[10], dx[10], dy[10]; int hs[3], vs[3]; long long off[3]; int pitch[3]; int mcux; };

// one CUDA block = 32 JPEG blocks x 8 threads. Block numbers fit 32 bits (nblk < 2^31 even for 65535 x 65535 4:4:4).
// A JPEG block's 8 threads read its coefficients as one 16-byte word each (the 16-coefficient groups past nq are
// not read, as before), into shared memory, then each takes its column from there.
extern "C" __global__ void idct_islow(const short* __restrict__ coef, const unsigned char* __restrict__ nq,
        const int* __restrict__ dc, const int* __restrict__ q, const Pix* __restrict__ P, i64 nblk,
        unsigned char* __restrict__ planes, int* __restrict__ err) {
    __shared__ int ws[32][64];
    __shared__ __align__(16) short sc[32][64];
    int t = threadIdx.x & 7, lb = threadIdx.x >> 3;
    u32 g = blockIdx.x * 32u + (u32)lb;
    bool ok = (i64)g < nblk;
    u32 bpm = (u32)P->bpm;
    int j = ok ? (int)(g % bpm) : 0;
    u32 m = ok ? g / bpm : 0;
    int comp = P->comp[j];
    const int* qq = q + comp * 64;
    bool bad = false;
    int np = 0;
    if (ok) {
        np = nq[g];
        np = np > 4 ? 4 : np;
        if ((t >> 1) < np) *(uint4*)&sc[lb][8 * t] = *(const uint4*)(coef + (size_t)g * 64 + 8 * t);
    }
    __syncwarp();
    if (ok) {
        int in[8];
        for (int k = 0; k < 8; k++) { int zi = ZZ[k * 8 + t]; in[k] = (zi >> 4) < np ? (int)sc[lb][zi] * qq[k * 8 + t] : 0; }
        if (t == 0) { int d = dc[g]; bad |= NOT16(d); in[0] = d * qq[0]; }
        for (int k = 0; k < 8; k++) bad |= NOT16(in[k]);
        int o[8];
        if (in[1] == 0 && in[2] == 0 && in[3] == 0 && in[4] == 0 && in[5] == 0 && in[6] == 0 && in[7] == 0) {
            for (int k = 0; k < 8; k++) o[k] = in[0] * (1 << PASS1_BITS);
        } else if (!bad) {
            bad |= idct8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], o, CONST_BITS - PASS1_BITS);
        } else {
            for (int k = 0; k < 8; k++) o[k] = 0;
        }
        for (int k = 0; k < 8; k++) { bad |= NOT16(o[k]); ws[lb][k * 8 + t] = o[k]; }
    }
    __syncthreads();
    // all 32 lanes are still here; a JPEG block's 8 threads share a warp
    bool warp_bad = __any_sync(0xffffffff, bad);
    if (!ok) return;
    const int* w = ws[lb] + t * 8;
    int o[8];
    if (!warp_bad) {
        bad |= idct8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], o, CONST_BITS + PASS1_BITS + 3);
    } else {
        for (int k = 0; k < 8; k++) o[k] = 0;
    }
    unsigned char px[8];
    for (int k = 0; k < 8; k++) {
        bad |= o[k] < -512 || o[k] > 511;               // libjpeg-turbo's C range-limit table wraps outside this
        px[k] = (unsigned char)(min(max(o[k], -128), 127) + 128);
    }
    if (bad) atomicOr(err, ERR_RANGE);
    u32 mcux = (u32)P->mcux;
    i64 by = (i64)(m / mcux) * P->vs[comp] + P->dy[j], bx = (i64)(m % mcux) * P->hs[comp] + P->dx[j];
    unsigned char* dst = planes + P->off[comp] + (by * 8 + t) * P->pitch[comp] + bx * 8;
    uint2 v;
    v.x = px[0] | (px[1] << 8) | (px[2] << 16) | ((u32)px[3] << 24);
    v.y = px[4] | (px[5] << 8) | (px[6] << 16) | ((u32)px[7] << 24);
    *(uint2*)dst = v;
}

// ---- opt-in fast IDCT: jidctfst.c jpeg_idct_ifast (JDCT_IFAST, simplejpeg fastdct=True) ----
// The multiplier table q (host: _ifast_table) holds libjpeg-turbo's IFAST multipliers, (quant * aanscale + 2^11)
// >> 12 as int16, with IFAST_SCALE_BITS = PASS1_BITS = 2, so pass 1 needs no descaling shift. 8-bit constants,
// truncating shifts (no USE_ACCURATE_ROUNDING), no rounding before the final >> 5.
//
// x86-64 libjpeg-turbo runs jidctfst-sse2.asm when the CPU has SSE2, else (JSIMD_FORCENONE, JSIMD_FORCEAVX2: no
// AVX2 ifast exists) the C version. The SIMD one does every step in 16 bits: pmullw, paddw/psubw wrap mod 2^16,
// each multiply is pmulhw(x << 2, c << 6), which wraps unless x is within 14 bits, and the outputs saturate
// (packsswb). The C one wraps at each DCTELEM (short) store, but forms the multiply inputs and the outputs as int
// sums of shorts, and its range-limit table wraps outside [-512, 511]. Both equal exact integer arithmetic when
// (per 8-point pass) the multiply inputs are within 14 bits, the shorts summed into them (i2, i6, z11, z13) within
// int16, and the outputs within [-512, 511] after >> 5; everything else either version does is the same mod 2^16.
// We compute exactly and flag a block where one of those fails (ERR_RANGE: the caller declines).
#define IF_1_082 277
#define IF_1_414 362
#define IF_1_847 473
#define IF_2_613 669
#define NOT14(v) ((u32)((v) + 8192) > 16383u)

// one 8-point pass on inputs within int16 (pass 1) or the outputs of pass 1 (pass 2): no int32 overflow
__device__ __forceinline__ bool ifast8(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int* o) {
    int tmp10 = i0 + i4, tmp11 = i0 - i4;                          // even part
    int tmp13 = i2 + i6, d26 = i2 - i6;
    int tmp12 = ((d26 * IF_1_414) >> 8) - tmp13;
    int tmp0 = tmp10 + tmp13, tmp3 = tmp10 - tmp13, tmp1 = tmp11 + tmp12, tmp2 = tmp11 - tmp12;
    int z13 = i5 + i3, z10 = i5 - i3, z11 = i1 + i7, z12 = i1 - i7;   // odd part
    int tmp7 = z11 + z13, d1113 = z11 - z13, s1012 = z10 + z12;
    int t11 = (d1113 * IF_1_414) >> 8;
    int z5 = (s1012 * IF_1_847) >> 8;
    int t10 = ((z12 * IF_1_082) >> 8) - z5;
    int t12 = ((z10 * -IF_2_613) >> 8) + z5;
    int tmp6 = t12 - tmp7, tmp5 = t11 - tmp6, tmp4 = t10 + tmp5;
    o[0] = tmp0 + tmp7; o[7] = tmp0 - tmp7; o[1] = tmp1 + tmp6; o[6] = tmp1 - tmp6;
    o[2] = tmp2 + tmp5; o[5] = tmp2 - tmp5; o[4] = tmp3 + tmp4; o[3] = tmp3 - tmp4;
    return NOT16(i2) | NOT16(i6) | NOT16(z11) | NOT16(z13) |
           NOT14(d26) | NOT14(z10) | NOT14(z12) | NOT14(s1012) | NOT14(d1113);
}

// as idct_islow (same layout, threads and loads), with ifast8 and the IFAST multipliers in q
extern "C" __global__ void idct_ifast(const short* __restrict__ coef, const unsigned char* __restrict__ nq,
        const int* __restrict__ dc, const int* __restrict__ q, const Pix* __restrict__ P, i64 nblk,
        unsigned char* __restrict__ planes, int* __restrict__ err) {
    __shared__ int ws[32][64];
    __shared__ __align__(16) short sc[32][64];
    int t = threadIdx.x & 7, lb = threadIdx.x >> 3;
    u32 g = blockIdx.x * 32u + (u32)lb;
    bool ok = (i64)g < nblk;
    u32 bpm = (u32)P->bpm;
    int j = ok ? (int)(g % bpm) : 0;
    u32 m = ok ? g / bpm : 0;
    int comp = P->comp[j];
    const int* qq = q + comp * 64;
    bool bad = false;
    int np = 0;
    if (ok) {
        np = nq[g];
        np = np > 4 ? 4 : np;
        if ((t >> 1) < np) *(uint4*)&sc[lb][8 * t] = *(const uint4*)(coef + (size_t)g * 64 + 8 * t);
    }
    __syncwarp();
    if (ok) {
        int in[8];
        for (int k = 0; k < 8; k++) { int zi = ZZ[k * 8 + t]; in[k] = (zi >> 4) < np ? (int)sc[lb][zi] * qq[k * 8 + t] : 0; }
        if (t == 0) { int d = dc[g]; bad |= NOT16(d); in[0] = d * qq[0]; }
        for (int k = 0; k < 8; k++) bad |= NOT16(in[k]);
        int o[8];
        if (in[1] == 0 && in[2] == 0 && in[3] == 0 && in[4] == 0 && in[5] == 0 && in[6] == 0 && in[7] == 0) {
            for (int k = 0; k < 8; k++) o[k] = in[0];
        } else if (!bad) {
            bad |= ifast8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], o);
        } else {
            for (int k = 0; k < 8; k++) o[k] = 0;
        }
        for (int k = 0; k < 8; k++) ws[lb][k * 8 + t] = o[k];
    }
    __syncthreads();
    bool warp_bad = __any_sync(0xffffffff, bad);
    if (!ok) return;
    const int* w = ws[lb] + t * 8;
    int o[8];
    if (!warp_bad) {
        bad |= ifast8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], o);
    } else {
        for (int k = 0; k < 8; k++) o[k] = 0;
    }
    unsigned char px[8];
    for (int k = 0; k < 8; k++) {
        int v = o[k] >> (PASS1_BITS + 3);
        bad |= v < -512 || v > 511;
        px[k] = (unsigned char)(min(max(v, -128), 127) + 128);
    }
    if (bad) atomicOr(err, ERR_RANGE);
    u32 mcux = (u32)P->mcux;
    i64 by = (i64)(m / mcux) * P->vs[comp] + P->dy[j], bx = (i64)(m % mcux) * P->hs[comp] + P->dx[j];
    unsigned char* dst = planes + P->off[comp] + (by * 8 + t) * P->pitch[comp] + bx * 8;
    uint2 v;
    v.x = px[0] | (px[1] << 8) | (px[2] << 16) | ((u32)px[3] << 24);
    v.y = px[4] | (px[5] << 8) | (px[6] << 16) | ((u32)px[7] << 24);
    *(uint2*)dst = v;
}

// ---- upsampling (jdsample.c) + colour conversion (jdcolor.c) ----
// Per component: up = (fh, fv) expansion (1 or 2 each), dw/dh = downsampled size, fancy = libjpeg's choice.
struct Up { int fh[3], fv[3], dw[3], dh[3], fancy[3]; };

// Per block (one output row y): for each component, its kind and the one or two input rows it reads.
// kind: 0 full size, 1 h2v1 fancy, 2 h1v2 fancy, 3 h2v2 fancy, 4 box (h2v1/h2v2_upsample, int_upsample)
struct RowCtx { const unsigned char* r0; const unsigned char* r1; int kind, fh, dw, bias; };

__device__ __forceinline__ void row_ctx(RowCtx* R, const unsigned char* base, int pitch, const Up* U, int ci, int y) {
    int fh = U->fh[ci], fv = U->fv[ci], dh = U->dh[ci];
    R->fh = fh; R->dw = U->dw[ci];
    if (fh == 1 && fv == 1) { R->kind = 0; R->r0 = base + (i64)y * pitch; R->r1 = R->r0; return; }
    if (!U->fancy[ci]) { R->kind = 4; R->r0 = base + (i64)(y / fv) * pitch; R->r1 = R->r0; return; }
    if (fv == 1) { R->kind = 1; R->r0 = base + (i64)y * pitch; R->r1 = R->r0; return; }
    int k = y >> 1;                                                  // nearest input row
    int nb = (y & 1) ? min(k + 1, dh - 1) : max(k - 1, 0);          // next nearest (edges replicate, jdmainct.c)
    R->r0 = base + (i64)k * pitch; R->r1 = base + (i64)nb * pitch;
    R->bias = (y & 1) ? 2 : 1;
    R->kind = fh == 1 ? 2 : 3;
}

// values of a component at output columns x and x + 1 (x even; v1 only if x + 1 < W) (jdsample.c). Neighbour
// loads past the row's real samples are predicated off.
template <int K>
__device__ __forceinline__ void up2k(const RowCtx& R, int x, bool two, int& v0, int& v1) {
    const unsigned char* a = R.r0;
    const unsigned char* b = R.r1;
    v1 = 0;
    switch (K < 0 ? R.kind : K) {
    case 0: v0 = a[x]; if (two) v1 = a[x + 1]; return;
    case 4: {                                                        // box: a shift for the usual factor 2
        int f = R.fh;
        v0 = a[f == 2 ? x >> 1 : x / f];
        if (two) v1 = a[f == 2 ? (x + 1) >> 1 : (x + 1) / f];
        return;
    }
    case 2:                                                          // h1v2_fancy_upsample
        v0 = (3 * a[x] + b[x] + R.bias) >> 2;
        if (two) v1 = (3 * a[x + 1] + b[x + 1] + R.bias) >> 2;
        return;
    case 1: {                                                        // h2v1_fancy_upsample
        int j = x >> 1, c = a[j];
        int l = j > 0 ? a[j - 1] : 0, r = j < R.dw - 1 ? a[j + 1] : 0;
        v0 = j == 0 ? c : (3 * c + l + 1) >> 2;
        v1 = j == R.dw - 1 ? c : (3 * c + r + 2) >> 2;
        return;
    }
    default: {                                                       // h2v2_fancy_upsample
        int j = x >> 1;
        int cs = 3 * a[j] + b[j];
        int cl = j > 0 ? 3 * a[j - 1] + b[j - 1] : 0;
        int cn = j < R.dw - 1 ? 3 * a[j + 1] + b[j + 1] : 0;
        v0 = j == 0 ? (cs * 4 + 8) >> 4 : (cs * 3 + cl + 8) >> 4;
        v1 = j == R.dw - 1 ? (cs * 4 + 7) >> 4 : (cs * 3 + cn + 7) >> 4;
        return;
    }
    }
}
__device__ __forceinline__ void up2(const RowCtx& R, int x, bool two, int& v0, int& v1) { up2k<-1>(R, x, two, v0, v1); }

__device__ __forceinline__ unsigned char cl8(int v) { return (unsigned char)min(max(v, 0), 255); }

// mode: 0 RGB, 1 BGR, 2 GRAY (Y only), 3 gray JPEG to RGB (replicate). Output (H, W, channels) uint8.
// A block of 128 threads converts CPX consecutive pixels of one row, two adjacent pixels per thread and step
// (so odd and even columns never diverge), staged in shared memory so the global stores are contiguous bytes.
#define CPX 1024

__device__ __forceinline__ void put_px(unsigned char* o, int Y, int cb, int cr, int mode) {
    if (mode == 2) { o[0] = (unsigned char)Y; return; }
    if (mode == 3) { o[0] = o[1] = o[2] = (unsigned char)Y; return; }
    cb -= 128; cr -= 128;
    unsigned char r = cl8(Y + ((@CRR@ * cr + 32768) >> 16));
    unsigned char g = cl8(Y + ((-@CBG@ * cb + 32768 - @CRG@ * cr) >> 16));
    unsigned char b = cl8(Y + ((@CBB@ * cb + 32768) >> 16));
    if (mode == 1) { unsigned char t = r; r = b; b = t; }
    o[0] = r; o[1] = g; o[2] = b;
}

template <int K0, int K1, int K2, int NC, int MODE>
__device__ __forceinline__ void conv_row(const RowCtx& r0, const RowCtx& r1, const RowCtx& r2, int x0, int W, int rmode,
        int C, unsigned char* sh) {
    const int mode = MODE < 0 ? rmode : MODE;
    for (int k = 0; k < CPX / 256; k++) {
        int i = 2 * (threadIdx.x + 128 * k), x = x0 + i;
        if (x >= W) break;
        bool two = x + 1 < W;
        int y0, y1, b0 = 0, b1 = 0, c0 = 0, c1 = 0;
        up2k<K0>(r0, x, two, y0, y1);
        if (NC == 3) { up2k<K1>(r1, x, two, b0, b1); up2k<K2>(r2, x, two, c0, c1); }
        put_px(sh + i * C, y0, b0, c0, mode);
        if (two) put_px(sh + (i + 1) * C, y1, b1, c1, mode);
    }
}

extern "C" __global__ void color(const unsigned char* __restrict__ planes, const Pix* __restrict__ P,
        const Up* __restrict__ U, int W, int H, int mode, unsigned char* __restrict__ out) {
    __shared__ __align__(16) unsigned char sh[CPX * 3];
    __shared__ RowCtx R[3];
    int x0 = blockIdx.x * CPX, y = blockIdx.y;
    int nc = mode >= 2 ? 1 : 3;
    if (threadIdx.x < nc) row_ctx(&R[threadIdx.x], planes + P->off[threadIdx.x], P->pitch[threadIdx.x], U, threadIdx.x, y);
    __syncthreads();
    RowCtx r0 = R[0], r1 = R[0], r2 = R[0];
    if (nc == 3) { r1 = R[1]; r2 = R[2]; }
    int C = mode == 2 ? 1 : 3;
    // the same conversion with the upsampling kinds known at compile time for 4:4:4, 4:2:2 and 4:2:0 (fancy), and
    // 4:2:2 / 4:2:0 with box upsampling (the fast_upsample option)
    int ks = nc == 3 ? r0.kind * 100 + r1.kind * 10 + r2.kind : -1;
    if (mode == 0 && ks == 0) conv_row<0, 0, 0, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);
    else if (mode == 0 && ks == 11) conv_row<0, 1, 1, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);
    else if (mode == 0 && ks == 33) conv_row<0, 3, 3, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);
    else if (mode == 0 && ks == 44) conv_row<0, 4, 4, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);   // box (fast upsample)
    else if (ks == 0) conv_row<0, 0, 0, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else if (ks == 11) conv_row<0, 1, 1, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else if (ks == 33) conv_row<0, 3, 3, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else if (nc == 3) conv_row<-1, -1, -1, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else conv_row<-1, -1, -1, 1, -1>(r0, r1, r2, x0, W, mode, C, sh);
    __syncthreads();
    int nb = min(CPX, W - x0) * C;
    unsigned char* dst = out + ((i64)y * W + x0) * C;
    // the row segment leaves shared memory in 16- or 4-byte words when its address allows (same bytes)
    int done = 0;
    if ((((size_t)dst) & 15) == 0) {
        int n16 = nb >> 4;
        for (int k = threadIdx.x; k < n16; k += 128) ((uint4*)dst)[k] = ((const uint4*)sh)[k];
        done = n16 << 4;
    } else if ((((size_t)dst) & 3) == 0) {
        int n4 = nb >> 2;
        for (int k = threadIdx.x; k < n4; k += 128) ((unsigned int*)dst)[k] = ((const unsigned int*)sh)[k];
        done = n4 << 2;
    }
    for (int k = done + threadIdx.x; k < nb; k += 128) dst[k] = sh[k];
}

// Epilogue variant (decode(..., dtype="float32" / "float16", layout=..., scale=..., bias=...)): the same bytes as
// color() from the same code, then each channel value v leaves as v * scale[c] + bias[c], rounded after the multiply
// and after the add (as NumPy / CuPy's img.astype(float32) * scale + bias), stored as float32 or float16 (round to
// nearest even), in HWC or CHW (channel planes) layout. One pass instead of decode + a conversion kernel.
struct Epi { float scale[3], bias[3]; int f16, chw; };
extern "C" __global__ void color_ep(const unsigned char* __restrict__ planes, const Pix* __restrict__ P,
        const Up* __restrict__ U, int W, int H, int mode, void* __restrict__ out, const Epi* __restrict__ E) {
    __shared__ __align__(16) unsigned char sh[CPX * 3];
    __shared__ RowCtx R[3];
    int x0 = blockIdx.x * CPX, y = blockIdx.y;
    int nc = mode >= 2 ? 1 : 3;
    if (threadIdx.x < nc) row_ctx(&R[threadIdx.x], planes + P->off[threadIdx.x], P->pitch[threadIdx.x], U, threadIdx.x, y);
    __syncthreads();
    RowCtx r0 = R[0], r1 = R[0], r2 = R[0];
    if (nc == 3) { r1 = R[1]; r2 = R[2]; }
    int C = mode == 2 ? 1 : 3;
    // the same conversion with the upsampling kinds known at compile time for 4:4:4, 4:2:2 and 4:2:0 (fancy)
    int ks = nc == 3 ? r0.kind * 100 + r1.kind * 10 + r2.kind : -1;
    if (mode == 0 && ks == 0) conv_row<0, 0, 0, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);
    else if (mode == 0 && ks == 11) conv_row<0, 1, 1, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);
    else if (mode == 0 && ks == 33) conv_row<0, 3, 3, 3, 0>(r0, r1, r2, x0, W, mode, 3, sh);
    else if (ks == 0) conv_row<0, 0, 0, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else if (ks == 11) conv_row<0, 1, 1, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else if (ks == 33) conv_row<0, 3, 3, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else if (nc == 3) conv_row<-1, -1, -1, 3, -1>(r0, r1, r2, x0, W, mode, C, sh);
    else conv_row<-1, -1, -1, 1, -1>(r0, r1, r2, x0, W, mode, C, sh);
    __syncthreads();
    int nb = min(CPX, W - x0) * C;
    float sc[3], bi[3];
    for (int c = 0; c < 3; c++) { sc[c] = E->scale[c]; bi[c] = E->bias[c]; }
    const int f16 = E->f16, chw = E->chw;
    for (int k = threadIdx.x; k < nb; k += 128) {
        int p = C == 3 ? k / 3 : k, c = k - p * C;
        float v = __fadd_rn(__fmul_rn((float)sh[k], sc[c]), bi[c]);
        i64 x = x0 + p;
        i64 idx = chw ? ((i64)c * H + y) * W + x : ((i64)y * W + x) * C + c;
        if (f16) {
            unsigned short hv;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(v));
            ((unsigned short*)out)[idx] = hv;
        } else {
            ((float*)out)[idx] = v;
        }
    }
}
