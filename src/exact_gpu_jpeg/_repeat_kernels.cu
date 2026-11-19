// exact-gpu-jpeg: kernels for the packed coefficient form (decoding the same file again). Compiled at run time
// with NVRTC, and only when a packed copy is first made or used.
//
// The packed form holds a decoded JPEG's quantised coefficients with no entropy coding, so that every block, and
// every quad of 16 coefficients in it, is unpacked by its own thread: no bit stream, no synchronisation, no DC
// prediction. It refills exactly what the IDCT reads: coef (zig-zag, in quads of 16; only the first nq[b] quads of
// a block), nq (valid quads, 0..4) and the absolute DC.
//
// idx[b] (u32): byte offset of the block's data (bits 0..27) | valid quads << 28 (0..4) | wide << 31.
// Block data, at an even offset: the absolute DC (i16); one u16 mask per valid quad (bit i: zig-zag position
// 16 q + i is not zero); the non-zero values in zig-zag order, i8 each, or i16 each if the block is wide (some
// value outside -128..127); one zero pad byte if the bytes so far are odd in number.
//
// No index word or data byte can make these kernels read or write outside their buffers, or make a misaligned
// load (which ends the whole CUDA context): an index word whose offset is past the data (dlen bytes), or odd, or
// that claims more than 4 quads is not read through at all (ERR_PK_RANGE). With offset <= dlen a block's reads
// stay within 160 bytes of its offset, and the buffer holds at least that many zero bytes after the data. Writes
// go only to the thread's own block. No field sizes a loop: a launch's run time depends on the block count alone.
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef long long i64;

#define ERR_PK_RANGE 1024
#define ERR_PK_LAYOUT 2048
#define PK_OFF 0x0fffffffu

// value r of a narrow quad, from the 16 bytes at its first value (lo, hi)
__device__ __forceinline__ u32 nval(u64 lo, u64 hi, int r) {
    u64 w = r < 8 ? lo >> (8 * r) : hi >> (8 * (r - 8));
    return (u32)(u16)(short)(signed char)(w & 255);
}

// value r of a wide quad
__device__ __forceinline__ u32 wval(const unsigned char* __restrict__ v, int r) {
    return (u32)v[2 * r] | ((u32)v[2 * r + 1] << 8);
}

// Packed copy -> coef, dc, nq. One thread per (block, quad).
extern "C" __global__ void unpack(const u32* __restrict__ idx, const unsigned char* __restrict__ data, int nblk,
        u32 dlen, short* __restrict__ coef, int* __restrict__ dc, unsigned char* __restrict__ nq, int* __restrict__ err) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    int b = (int)(t >> 2), q = (int)(t & 3);
    if (b >= nblk) return;
    u32 h = idx[b];
    int n = (int)((h >> 28) & 7);
    bool wide = (h >> 31) != 0;
    u32 off = h & PK_OFF;
    if (off > dlen || (off & 1u) || n > 4) {
        if (q == 0) { dc[b] = 0; nq[b] = 0; atomicOr(err, ERR_PK_RANGE); }
        return;
    }
    const unsigned char* p = data + off;
    const u16* p16 = (const u16*)p;
    if (q == 0) { dc[b] = (int)(short)p16[0]; nq[b] = (unsigned char)n; }
    if (q >= n) return;
    int base = 0;                                           // values before this quad
    for (int j = 0; j < q; j++) base += __popc((u32)p16[1 + j]);
    u32 m = p16[1 + q];                                     // this quad's mask
    const unsigned char* v = p + 2 + 2 * n + (wide ? 2 * base : base);
    u32 w[8];
    // a narrow quad's values (at most 16 bytes) through three 8-byte loads at the address rounded down to a
    // multiple of 8: aligned whatever the copy says
    u64 vlo = 0, vhi = 0;
    if (!wide) {
        const u64* a8 = (const u64*)((size_t)v & ~(size_t)7);
        int sh = (int)((size_t)v & 7) * 8;
        u64 w0 = a8[0], w1 = a8[1], w2 = a8[2];
        vlo = sh ? (w0 >> sh) | (w1 << (64 - sh)) : w0;
        vhi = sh ? (w1 >> sh) | (w2 << (64 - sh)) : w1;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int i0 = 2 * k, i1 = 2 * k + 1;
        u32 lo = 0, hi = 0;
        if ((m >> i0) & 1) { int r = __popc(m & ((1u << i0) - 1)); lo = wide ? wval(v, r) : nval(vlo, vhi, r); }
        if ((m >> i1) & 1) { int r = __popc(m & ((1u << i1) - 1)); hi = wide ? wval(v, r) : nval(vlo, vhi, r); }
        w[k] = lo | (hi << 16);
    }
    uint4* dst = (uint4*)(coef + (i64)b * 64 + q * 16);
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// The layout check, for a copy that was not made by this process: one thread per block. The form the packer
// makes (ERR_PK_LAYOUT otherwise): block 0 starts at 0 and each block's size (from its quads, masks and wide flag)
// ends where the next block starts, the last at dlen; the last valid quad's mask is not 0; bit 0 of the first mask
// is 0 (position 0 is the DC's place); a block without values is not wide; a wide block holds a value outside
// -128..127; a pad byte is 0.
extern "C" __global__ void pk_check(const u32* __restrict__ idx, const unsigned char* __restrict__ data, int nblk,
        u32 dlen, int* __restrict__ err) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    u32 h = idx[b];
    int n = (int)((h >> 28) & 7);
    u32 off = h & PK_OFF;
    if (off > dlen || (off & 1u) || n > 4) { atomicOr(err, ERR_PK_RANGE); return; }
    const u16* p16 = (const u16*)(data + off);
    int cnt = 0;
    for (int j = 0; j < n; j++) cnt += __popc((u32)p16[1 + j]);
    bool wide = (h >> 31) != 0;
    u32 used = (u32)(2 + 2 * n + cnt * (wide ? 2 : 1));
    u32 size = (used + 1) & ~1u;
    u32 next = b + 1 < nblk ? (idx[b + 1] & PK_OFF) : dlen;
    bool canon = n == 0 ? !wide : (p16[n] != 0 && (p16[1] & 1u) == 0);
    if (used & 1u) canon = canon && data[off + used] == 0;
    if (wide && n > 0) {
        const unsigned char* v = data + off + 2 + 2 * n;
        bool big = false;
        for (int r = 0; r < cnt; r++) {
            int x = (int)(short)((u32)v[2 * r] | ((u32)v[2 * r + 1] << 8));
            big = big || x < -128 || x > 127;
        }
        canon = canon && big;
    }
    if (off + size != next || (b == 0 && off != 0) || !canon) atomicOr(err, ERR_PK_LAYOUT);
}

// ---- the packer: the arrays a decode leaves on the GPU -> a packed copy. Two stages around one prefix sum of the
// block sizes; both take a block range. ----

// sz[b]: the data bytes of block b (even). meta[b]: valid quads | wide << 3 | the non-zero count of quad j
// << (4 + 5 j). *err is set when a DC does not fit 16 bits (the image is then not packed).
extern "C" __global__ void pk_size(const short* __restrict__ coef, const unsigned char* __restrict__ nq,
        const int* __restrict__ dc, int b0, int b1, int* __restrict__ sz, u32* __restrict__ meta, int* __restrict__ err) {
    int b = b0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= b1) return;
    int n = min((int)nq[b], 4);
    const u32* c = (const u32*)(coef + (i64)b * 64);
    u32 m = (u32)n;
    int tot = 0, wide = 0;
    for (int q = 0; q < n; q++) {
        int cnt = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            u32 w = c[q * 8 + i];
            int lo = (short)(w & 0xffffu), hi = (short)(w >> 16);
            cnt += (lo != 0) + (hi != 0);
            wide |= (lo < -128) | (lo > 127) | (hi < -128) | (hi > 127);
        }
        m |= (u32)cnt << (4 + 5 * q);
        tot += cnt;
    }
    m |= (u32)wide << 3;
    sz[b] = (2 + 2 * n + tot * (1 + wide) + 1) & ~1;
    meta[b] = m;
    int d = dc[b];
    if (d < -32768 || d > 32767) atomicOr(err, 1);
}

// off: nblk + 1 byte offsets (off[0] = 0). One thread per (block, quad).
extern "C" __global__ void pk_write(const short* __restrict__ coef, const int* __restrict__ dc,
        const u32* __restrict__ off, const u32* __restrict__ meta, int b0, int b1, u32* __restrict__ idx,
        unsigned char* __restrict__ data) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    int b = b0 + (int)(t >> 2), q = (int)(t & 3);
    if (b >= b1) return;
    u32 m = meta[b];
    int n = (int)(m & 7);
    bool wide = ((m >> 3) & 1) != 0;
    u32 o = off[b];
    unsigned char* p = data + o;
    if (q == 0) { idx[b] = o | ((u32)n << 28) | ((u32)wide << 31); *(short*)p = (short)dc[b]; }
    if (q >= n) return;
    int base = 0, total = 0;
    for (int j = 0; j < n; j++) { int cj = (int)((m >> (4 + 5 * j)) & 31); base += j < q ? cj : 0; total += cj; }
    const short* c = coef + (i64)b * 64 + q * 16;
    unsigned char* v = p + 2 + 2 * n + (wide ? 2 * base : base);
    u32 mask = 0;
    int k = 0;
    for (int i = 0; i < 16; i++) {
        int x = c[i];
        if (x != 0) {
            mask |= 1u << i;
            if (wide) { v[2 * k] = (unsigned char)(x & 255); v[2 * k + 1] = (unsigned char)((x >> 8) & 255); }
            else v[k] = (unsigned char)x;
            k++;
        }
    }
    *(u16*)(p + 2 + 2 * q) = (u16)mask;
    if (q == n - 1) { int bytes = 2 + 2 * n + total * (wide ? 2 : 1); if (bytes & 1) p[bytes] = 0; }
}
