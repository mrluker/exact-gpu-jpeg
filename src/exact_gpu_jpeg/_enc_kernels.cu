// exact-gpu-jpeg encoder kernels (CUDA, compiled at run time with NVRTC).
//
// THIS FILE IS A MODIFIED PORT, NOT ORIGINAL libjpeg-turbo CODE. It holds re-implementations for CUDA of routines
// from libjpeg-turbo 3.1.2, which is based on the Independent JPEG Group's software:
//   Copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
//   libjpeg-turbo modifications: Copyright (C) 2009-2025 D. R. Commander and the libjpeg-turbo contributors.
// The ported parts stay under the IJG License (licenses/README.ijg; see also licenses/libjpeg-turbo-LICENSE.md).
// THIRD_PARTY_NOTICES.md names each routine, its upstream file and what was changed.
//
// Byte-identical to libjpeg-turbo's baseline compressor as Pillow drives it (islow DCT, no smoothing, standard
// Huffman tables). Ported from libjpeg-turbo 3.1.2 (jccolor.c rgb_ycc_convert, jcsample.c fullsize/h2v1/h2v2
// downsampling and edge expansion, jcprepct.c bottom-edge padding, jfdctint.c jpeg_fdct_islow, jcdctmgr.c
// quantisation by reciprocals with 16-bit DCTELEM as in SIMD builds, jccoefct.c dummy blocks, jchuff.c
// encode_one_block); these are based on the Independent JPEG Group's software (IJG License, licenses/README.ijg).
// THIRD_PARTY_NOTICES.md lists what was changed.

typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;

#define ERR_INPUT 1

// layout (int32 words): bpm, comp[10], dx[10], dy[10], hs[3], vs[3], hmax, vmax, mcux, wib[3], hib[3], W, H,
// firstj[3], lastj[3], tq[3]
struct EL {
    int bpm; int comp[10], dx[10], dy[10]; int hs[3], vs[3]; int hmax, vmax, mcux; int wib[3], hib[3]; int W, H;
    int firstj[3], lastj[3], tq[3];
};

__constant__ unsigned char ZZ[64] = {@ZZ@};    // natural index -> zig-zag position

// jccolor.c rgb_ycc_convert (fixed point, SCALEBITS 16)
__device__ __forceinline__ int conv(const unsigned char* rgb, int W, int H, int x, int y, int ci) {
    x = min(x, W - 1); y = min(y, H - 1);                       // jcsample/jcprepct edge expansion
    const unsigned char* p = rgb + ((i64)y * W + x) * 3;
    int r = p[0], g = p[1], b = p[2];
    if (ci == 0) return (@FY_R@ * r + @FY_G@ * g + @FY_B@ * b + 32768) >> 16;
    if (ci == 1) return (-@FCB_R@ * r - @FCB_G@ * g + @F_HALF@ * b + (128 << 16) + 32767) >> 16;
    return (@F_HALF@ * r - @FCR_G@ * g - @FCR_B@ * b + (128 << 16) + 32767) >> 16;
}

// component ci's sample (sx, sy) after downsampling (jcsample.c, no smoothing) and edge padding
__device__ __forceinline__ int comp_sample(const unsigned char* rgb, const EL* L, int ci, int sx, int sy) {
    int fh = L->hmax / L->hs[ci], fv = L->vmax / L->vs[ci];
    int W = L->W, H = L->H;
    if (fh == 1 && fv == 1) return conv(rgb, W, H, sx, sy, ci);
    if (fh == 2 && fv == 1) {                                     // h2v1_downsample: bias 0, 1, 0, 1, ...
        return (conv(rgb, W, H, 2 * sx, sy, ci) + conv(rgb, W, H, 2 * sx + 1, sy, ci) + (sx & 1)) >> 1;
    }
    // h2v2_downsample: bias 1, 2, 1, 2, ...; rows past the downsampled height repeat its last row
    int cy = min(sy, (H + 1) / 2 - 1);
    return (conv(rgb, W, H, 2 * sx, 2 * cy, ci) + conv(rgb, W, H, 2 * sx + 1, 2 * cy, ci) +
            conv(rgb, W, H, 2 * sx, 2 * cy + 1, ci) + conv(rgb, W, H, 2 * sx + 1, 2 * cy + 1, ci) +
            ((sx & 1) ? 2 : 1)) >> 2;
}

// Huffman tables: code[4][256] (u16), size[4][256] (u8): 0 DC luminance, 1 AC luminance, 2 DC chroma, 3 AC chroma
struct HT { unsigned short code[4][256]; unsigned char size[4][256]; };

__device__ __forceinline__ int nbits(int v) { v = v < 0 ? -v : v; return v ? 32 - __clz(v) : 0; }

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

// jfdctint.c jpeg_fdct_islow, one 8-point pass; pass 1: rows (shift1 = 0 for the even outputs' left shift)
__device__ __forceinline__ void fdct8(const int* d, int* o, bool pass1) {
    int tmp0 = d[0] + d[7], tmp7 = d[0] - d[7], tmp1 = d[1] + d[6], tmp6 = d[1] - d[6];
    int tmp2 = d[2] + d[5], tmp5 = d[2] - d[5], tmp3 = d[3] + d[4], tmp4 = d[3] - d[4];
    int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    int sh = pass1 ? CONST_BITS - PASS1_BITS : CONST_BITS + PASS1_BITS;
    if (pass1) { o[0] = (tmp10 + tmp11) * (1 << PASS1_BITS); o[4] = (tmp10 - tmp11) * (1 << PASS1_BITS); }
    else { o[0] = DESCALE(tmp10 + tmp11, PASS1_BITS); o[4] = DESCALE(tmp10 - tmp11, PASS1_BITS); }
    int z1 = (tmp12 + tmp13) * F_0_541;
    o[2] = DESCALE(z1 + tmp13 * F_0_765, sh);
    o[6] = DESCALE(z1 + tmp12 * (-F_1_847), sh);
    z1 = tmp4 + tmp7; int z2 = tmp5 + tmp6, z3 = tmp4 + tmp6, z4 = tmp5 + tmp7;
    int z5 = (z3 + z4) * F_1_175;
    tmp4 *= F_0_298; tmp5 *= F_2_053; tmp6 *= F_3_072; tmp7 *= F_1_501;
    z1 *= -F_0_899; z2 *= -F_2_562; z3 *= -F_1_961; z4 *= -F_0_390;
    z3 += z5; z4 += z5;
    o[7] = DESCALE(tmp4 + z1 + z3, sh);
    o[5] = DESCALE(tmp5 + z2 + z4, sh);
    o[3] = DESCALE(tmp6 + z2 + z3, sh);
    o[1] = DESCALE(tmp7 + z1 + z4, sh);
}

// jcdctmgr.c quantize() with divisors from compute_reciprocal(q << 3) for a 16-bit DCTELEM (SIMD builds)
__device__ __forceinline__ int quant(int v, const int* dv, int i) {
    short t = (short)v;
    u32 recip = (u32)dv[i], corr = (u32)dv[64 + i];
    int shift = dv[128 + i];
    // one path for both signs (the magnitude, then the sign back): the same operations as jcdctmgr.c's two
    unsigned short a = t < 0 ? (unsigned short)(-t) : (unsigned short)t;
    u32 p = (u32)(unsigned short)(a + corr) * recip;
    int q = (int)(short)(p >> (shift + 16));
    return t < 0 ? -q : q;
}

// colour conversion + downsampling + forward DCT + quantisation. One CUDA block = 32 JPEG blocks x 8 threads.
// div: per quantisation table (0, 1): recip[64], corr[64], shift[64] in natural order. Dummy blocks (past the
// component's real blocks, in partial MCUs) are zero; enc_dummy_dc sets their DC afterwards.
// nb[g] = the bits of block g's AC part as jchuff.c encode_one_block writes them (run/size codes, ZRLs, value bits,
// EOB); enc_dc_bits adds the DC part once the DC values are final.
extern "C" __global__ void enc_dct(const unsigned char* __restrict__ rgb, const EL* __restrict__ L,
        const int* __restrict__ div, i64 nblk, short* __restrict__ coef, const HT* __restrict__ T,
        int* __restrict__ nb) {
    __shared__ int ws[32][64];
    __shared__ unsigned char asz[2][256];            // AC code lengths: luminance, chroma
    for (int i = threadIdx.x; i < 512; i += blockDim.x) asz[i >> 8][i & 255] = T->size[(i >> 8) ? 3 : 1][i & 255];
    int t = threadIdx.x & 7, lb = threadIdx.x >> 3;
    u32 g = blockIdx.x * 32u + (u32)lb;                 // block numbers fit 32 bits at any JPEG size
    bool ok = (i64)g < nblk;
    u32 bpm = (u32)L->bpm, mcux = (u32)L->mcux;
    int j = ok ? (int)(g % bpm) : 0;
    u32 m = ok ? g / bpm : 0;
    int ci = L->comp[j];
    int by = (int)(m / mcux) * L->vs[ci] + L->dy[j], bx = (int)(m % mcux) * L->hs[ci] + L->dx[j];
    bool real = ok && bx < L->wib[ci] && by < L->hib[ci];
    if (real) {
        int d[8], o[8];
        if (L->hs[ci] == L->hmax && L->vs[ci] == L->vmax) {
            // a full-size component: conv() with the component's coefficients picked per thread instead of
            // branched on (the threads of a warp convert different components); the same integer sum
            int W = L->W, H = L->H;
            int cr = ci == 0 ? @FY_R@ : ci == 1 ? -@FCB_R@ : @F_HALF@;
            int cg = ci == 0 ? @FY_G@ : ci == 1 ? -@FCB_G@ : -@FCR_G@;
            int cb = ci == 0 ? @FY_B@ : ci == 1 ? @F_HALF@ : -@FCR_B@;
            int c0 = ci == 0 ? 32768 : (128 << 16) + 32767;
            const unsigned char* row = rgb + (i64)min(by * 8 + t, H - 1) * W * 3;
            for (int k = 0; k < 8; k++) {
                const unsigned char* q = row + min(bx * 8 + k, W - 1) * 3;
                d[k] = ((cr * q[0] + cg * q[1] + cb * q[2] + c0) >> 16) - 128;
            }
        } else {
            for (int k = 0; k < 8; k++) d[k] = comp_sample(rgb, L, ci, bx * 8 + k, by * 8 + t) - 128;
        }
        fdct8(d, o, true);
        for (int k = 0; k < 8; k++) ws[lb][t * 8 + k] = (int)(short)o[k];    // DCTELEM is 16-bit
    }
    __syncthreads();
    int qv[8];
    if (real) {
        int d[8], o[8];
        for (int k = 0; k < 8; k++) d[k] = ws[lb][k * 8 + t];
        fdct8(d, o, false);
        const int* dv = div + L->tq[ci] * 192;
        for (int k = 0; k < 8; k++) qv[k] = (short)quant(o[k], dv, k * 8 + t);
    } else {
        for (int k = 0; k < 8; k++) qv[k] = 0;
    }
    __syncwarp();                                    // a block's 8 threads are in one warp: ws[lb] is free again
    short* cf = coef + (size_t)g * 64;
    if (ok) for (int k = 0; k < 8; k++) { int z = ZZ[k * 8 + t]; ws[lb][z] = qv[k]; }
    __syncwarp();
    // the block's coefficients leave in zig-zag order as one 16-byte word per thread (positions 8t..8t+7)
    if (ok) {
        const int* w = ws[lb] + 8 * t;
        uint4 v;
        v.x = (u32)(unsigned short)w[0] | ((u32)(unsigned short)w[1] << 16);
        v.y = (u32)(unsigned short)w[2] | ((u32)(unsigned short)w[3] << 16);
        v.z = (u32)(unsigned short)w[4] | ((u32)(unsigned short)w[5] << 16);
        v.w = (u32)(unsigned short)w[6] | ((u32)(unsigned short)w[7] << 16);
        *(uint4*)(cf + 8 * t) = v;
    }
    // AC bits: lane t takes zig-zag positions 8t..8t+7; the 64-bit mask of non-zero AC coefficients gives each one's
    // run (positions since the previous non-zero one).
    u32 m8 = 0;
    for (int k = 0; k < 8; k++) { int p = 8 * t + k; if (p && ws[lb][p]) m8 |= 1u << k; }
    u64 mask = (u64)m8 << (8 * t);
    for (int o = 1; o < 8; o <<= 1) mask |= __shfl_xor_sync(0xffffffff, mask, o);
    const unsigned char* sz = asz[ci ? 1 : 0];
    int bits = 0;
    for (int k = 0; k < 8; k++) {                     // every position computed, the zero ones add nothing (no branch)
        int p = 8 * t + k;
        u64 below = mask & ((1ull << p) - 1);
        int r = p - (below ? 63 - __clzll((long long)below) : 0) - 1;
        int sv = nbits(ws[lb][p]);
        int add = (r >> 4) * sz[0xF0] + sz[((r & 15) << 4) | sv] + sv;
        bits += (m8 >> k & 1) ? add : 0;
    }
    if (t == 0 && !(mask >> 63 & 1)) bits += sz[0];  // EOB unless coefficient 63 is non-zero
    for (int o = 1; o < 8; o <<= 1) bits += __shfl_xor_sync(0xffffffff, bits, o);
    if (ok && t == 0) nb[g] = bits;
}

// adds each block's DC bits (size code + value bits of the difference to the previous block of its component)
extern "C" __global__ void enc_dc_bits(const short* __restrict__ coef, const EL* __restrict__ L, const HT* __restrict__ T,
        i64 nblk, int* __restrict__ nb);

// jccoefct.c compress_data: a dummy block at the right edge takes the DC of the block to its left, a row of dummy
// blocks at the bottom the DC of the MCU's previous block. One thread per MCU, blocks in MCU order.
extern "C" __global__ void enc_dummy_dc(const EL* __restrict__ L, i64 nmcu, short* __restrict__ coef) {
    i64 m = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= nmcu) return;
    for (int j = 0; j < L->bpm; j++) {
        int ci = L->comp[j];
        int by = (int)(m / L->mcux) * L->vs[ci] + L->dy[j], bx = (int)(m % L->mcux) * L->hs[ci] + L->dx[j];
        i64 g = m * L->bpm + j;
        if (by >= L->hib[ci]) coef[g * 64] = coef[(g - L->dx[j] - 1) * 64];     // MCU_buffer[blkn - 1]
        else if (bx >= L->wib[ci]) coef[g * 64] = coef[(g - 1) * 64];           // block to the left
    }
}


__device__ __forceinline__ i64 prev_block(const EL* L, i64 g) {     // previous block of the same component, or -1
    int j = (int)(g % L->bpm), ci = L->comp[j];
    if (j > L->firstj[ci]) return g - 1;
    i64 m = g / L->bpm;
    return m > 0 ? (m - 1) * L->bpm + L->lastj[ci] : -1;
}

// jchuff.c encode_one_block: calls put(code, length) for each code + value bits (at most 31 bits each)
template <class F>
__device__ __forceinline__ void encode_block(const short* cf, int dcdiff, const HT* T, int tdc, int tac, F put) {
    int s = nbits(dcdiff);
    int v = dcdiff < 0 ? dcdiff - 1 : dcdiff;
    put(((u32)T->code[tdc][s] << s) | ((u32)v & ((1u << s) - 1)), T->size[tdc][s] + s);
    int r = 0;
    for (int k = 1; k < 64; k++) {
        int c = cf[k];
        if (c == 0) { r++; continue; }
        while (r > 15) { put(T->code[tac][0xF0], T->size[tac][0xF0]); r -= 16; }
        s = nbits(c);
        v = c < 0 ? c - 1 : c;
        int sym = (r << 4) + s;
        put(((u32)T->code[tac][sym] << s) | ((u32)v & ((1u << s) - 1)), T->size[tac][sym] + s);
        r = 0;
    }
    if (r > 0) put(T->code[tac][0], T->size[tac][0]);
}

extern "C" __global__ void enc_dc_bits(const short* __restrict__ coef, const EL* __restrict__ L, const HT* __restrict__ T,
        i64 nblk, int* __restrict__ nb) {
    i64 g = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nblk) return;
    int ci = L->comp[g % L->bpm];
    i64 p = prev_block(L, g);
    int s = nbits(coef[g * 64] - (p >= 0 ? coef[p * 64] : 0));
    nb[g] += T->size[ci ? 2 : 0][s] + s;
}

// writes each block's bits at its offset (bit 31 of word 0 is the first bit). A block's bits are gathered in a
// 64-bit accumulator and written a word at a time: words wholly inside the block with plain stores (no other block
// writes them), its first and last words, which it may share with its neighbours, with atomicOr (buf starts zeroed).
// The coefficients go through shared memory (one 128-byte row per thread, 16-byte loads) and only the non-zero ones
// are visited. Same bits as encode_block (jchuff.c encode_one_block).
#define EW 128
#define CSW 33                                         // words per row (64 coefficients + 1 word against bank conflicts)
extern "C" __global__ void __launch_bounds__(EW) enc_write(const short* __restrict__ coef, const EL* __restrict__ L,
        const HT* __restrict__ T, i64 nblk, const i64* __restrict__ off, u32* __restrict__ buf, i64 cap_bits,
        int* __restrict__ overflow) {
    __shared__ u32 cs[EW * CSW];
    __shared__ HT tb;
    for (int i = threadIdx.x; i < (int)(sizeof(HT) / 4); i += EW) ((u32*)&tb)[i] = ((const u32*)T)[i];
    __syncthreads();
    i64 g = (i64)blockIdx.x * EW + threadIdx.x;
    if (g >= nblk) return;
    const i64 total_bits = off[nblk];
    if (total_bits + 8 > cap_bits) { if (g == 0) *overflow = 1; return; }    // buffer too small: the caller retries
    u32* row = cs + threadIdx.x * CSW;
    const uint4* src = (const uint4*)(coef + g * 64);
    u64 mask = 0;
    for (int q = 0; q < 8; q++) {
        uint4 v = src[q];
        row[4 * q] = v.x; row[4 * q + 1] = v.y; row[4 * q + 2] = v.z; row[4 * q + 3] = v.w;
        u32 w[4] = {v.x, v.y, v.z, v.w};
        for (int h = 0; h < 4; h++) {
            if (w[h] & 0xFFFF) mask |= 1ull << (8 * q + 2 * h);
            if (w[h] >> 16) mask |= 1ull << (8 * q + 2 * h + 1);
        }
    }
    mask &= ~1ull;                                     // AC only
    const short* cv = (const short*)row;
    int ci = L->comp[g % L->bpm];
    const int tdc = ci ? 2 : 0, tac = ci ? 3 : 1;
    i64 pb = prev_block(L, g);
    int dcdiff = cv[0] - (pb >= 0 ? coef[pb * 64] : 0);
    i64 pos = off[g];
    i64 w = pos >> 5;
    int fill = (int)(pos & 31);                        // bits in acc (left-aligned); the first `fill` are the previous block's (zero here)
    bool shared_first = fill != 0;
    u64 acc = 0;
    auto put = [&](u32 c, int len) {                   // len <= 31
        if (len == 0) return;
        acc |= (u64)(c & ((1u << len) - 1)) << (64 - fill - len);
        fill += len;
        if (fill >= 32) {
            u32 x = (u32)(acc >> 32);
            if (shared_first) { atomicOr(buf + w, x); shared_first = false; } else buf[w] = x;
            acc <<= 32; fill -= 32; w++;
        }
    };
    int s = nbits(dcdiff);
    int v = dcdiff < 0 ? dcdiff - 1 : dcdiff;
    put(((u32)tb.code[tdc][s] << s) | ((u32)v & ((1u << s) - 1)), tb.size[tdc][s] + s);
    int prev = 0;
    while (mask) {
        int p = __ffsll((long long)mask) - 1;
        mask &= mask - 1;
        int r = p - prev - 1;
        while (r > 15) { put(tb.code[tac][0xF0], tb.size[tac][0xF0]); r -= 16; }
        int c = cv[p];
        s = nbits(c);
        v = c < 0 ? c - 1 : c;
        int sym = (r << 4) + s;
        put(((u32)tb.code[tac][sym] << s) | ((u32)v & ((1u << s) - 1)), tb.size[tac][sym] + s);
        prev = p;
    }
    if (prev < 63) put(tb.code[tac][0], tb.size[tac][0]);
    if (fill > 0) atomicOr(buf + w, (u32)(acc >> 32)); // the last, partial word (shared with the next block)
    if (g == nblk - 1) {                                                   // jchuff.c flush_bits: pad with ones
        int pad = (int)((8 - (total_bits & 7)) & 7);
        if (pad) {
            i64 w2 = total_bits >> 5;
            int sh = (int)(total_bits & 31);
            u64 x = (((1ull << pad) - 1) << (64 - pad)) >> sh;
            if ((u32)(x >> 32)) atomicOr(buf + w2, (u32)(x >> 32));
            if ((u32)x) atomicOr(buf + w2 + 1, (u32)x);
        }
    }
}

// ---- byte stuffing: a 0x00 after every 0xFF ----
#define TILE 4096
__device__ __forceinline__ int byte_at(const u32* buf, i64 i) { return (buf[i >> 2] >> (24 - 8 * (int)(i & 3))) & 0xFF; }
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
// n (bytes of entropy-coded data) is read from the device: (*total_bits + 7) / 8, at most cap
__device__ __forceinline__ i64 nbytes_of(const i64* total_bits, i64 cap) { i64 n = (*total_bits + 7) / 8; return n < cap ? n : cap; }
// A thread's 16 bytes (i0 is a multiple of 16: four whole big-endian words) as four words in byte order.
__device__ __forceinline__ void load16w(const u32* buf, i64 i0, u32 v[4]) {
    uint4 q = *(const uint4*)(buf + (i0 >> 2));
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
}
__device__ __forceinline__ bool has_ff(const u32 v[4]) {
    return (__vcmpeq4(v[0], 0xFFFFFFFFu) | __vcmpeq4(v[1], 0xFFFFFFFFu) | __vcmpeq4(v[2], 0xFFFFFFFFu) |
            __vcmpeq4(v[3], 0xFFFFFFFFu)) != 0;
}
extern "C" __global__ void stuff_count(const u32* __restrict__ buf, const i64* __restrict__ total_bits, i64 cap,
        int* __restrict__ tilecnt) {
    __shared__ int tot;
    i64 n = nbytes_of(total_bits, cap);
    if ((i64)blockIdx.x * TILE >= n) { if (threadIdx.x == 0) tilecnt[blockIdx.x] = 0; return; }
    i64 i0 = (i64)blockIdx.x * TILE + threadIdx.x * 16;
    int c = 0;
    if (i0 + 16 <= n) {
        u32 v[4];
        load16w(buf, i0, v);
        c = 16;
        if (has_ff(v)) for (int k = 0; k < 16; k++) c += ((v[k >> 2] >> (24 - 8 * (k & 3))) & 0xFF) == 0xFF;
    } else {
        for (int k = 0; k < 16; k++) if (i0 + k < n) c += 1 + (byte_at(buf, i0 + k) == 0xFF);
    }
    block_excl_scan(c, &tot);
    if (threadIdx.x == 0) tilecnt[blockIdx.x] = tot;
}
extern "C" __global__ void stuff_scatter(const u32* __restrict__ buf, const i64* __restrict__ total_bits, i64 cap,
        const i64* __restrict__ toff, unsigned char* __restrict__ out, i64 out_cap, int* __restrict__ overflow) {
    __shared__ int tot;
    __shared__ unsigned char sb[2 * TILE];            // the tile's output bytes
    i64 n = nbytes_of(total_bits, cap);
    if ((i64)blockIdx.x * TILE >= n) return;           // the same for every thread of the block
    i64 i0 = (i64)blockIdx.x * TILE + threadIdx.x * 16;
    u32 v[4] = {0, 0, 0, 0};
    bool whole = i0 + 16 <= n, ff = true;
    int c = 0;
    if (whole) {
        load16w(buf, i0, v);
        ff = has_ff(v);
        c = 16;
        if (ff) for (int k = 0; k < 16; k++) c += ((v[k >> 2] >> (24 - 8 * (k & 3))) & 0xFF) == 0xFF;
    } else {
        for (int k = 0; k < 16; k++) if (i0 + k < n) c += 1 + (byte_at(buf, i0 + k) == 0xFF);
    }
    int ol = block_excl_scan(c, &tot);
    const i64 g0 = toff[blockIdx.x];
    if (g0 + tot + 1 > out_cap) { if (threadIdx.x == 0) *overflow = 1; return; }  // the same for every thread
    for (int k = 0; k < 16; k++) {
        if (i0 + k >= n) break;
        int b = whole ? (v[k >> 2] >> (24 - 8 * (k & 3))) & 0xFF : byte_at(buf, i0 + k);
        sb[ol++] = (unsigned char)b;
        if (ff && b == 0xFF) sb[ol++] = 0;
    }
    __syncthreads();
    const int K = tot;
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
}

// One block of SCAN1 threads: off[0] = 0, off[i + 1] = v[0] + ... + v[i] (i < n), in place of a cumulative sum made
// of several launches. Each thread adds up a run of ceil(n / SCAN1) values, thread 0 turns the run sums into each
// run's starting value, then each thread writes its run.
#define SCAN1 1024
extern "C" __global__ void __launch_bounds__(SCAN1) enc_offsets(const int* __restrict__ v, int n, i64* __restrict__ off) {
    __shared__ i64 rs[SCAN1];
    const int per = (n + SCAN1 - 1) / SCAN1, a = min(n, (int)threadIdx.x * per), b = min(n, a + per);
    i64 t = 0;
    for (int i = a; i < b; i++) t += v[i];
    rs[threadIdx.x] = t;
    __syncthreads();
    if (threadIdx.x == 0) {
        i64 c = 0;
        for (int k = 0; k < SCAN1; k++) { i64 x = rs[k]; rs[k] = c; c += x; }
        off[0] = 0;
    }
    __syncthreads();
    t = rs[threadIdx.x];
    for (int i = a; i < b; i++) { t += v[i]; off[i + 1] = t; }
}
// The four words the host reads back, gathered by one launch: bytes after stuffing, bits before, the two flags.
extern "C" __global__ void enc_info(const i64* __restrict__ nout, const i64* __restrict__ total_bits,
        const int* __restrict__ flags, i64* __restrict__ info) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { info[0] = *nout; info[1] = *total_bits; info[2] = flags[0]; info[3] = flags[1]; }
}

// ---- small images: fewer launches ----
// One block of SCAN1 threads in place of enc_dc_bits, the offsets' scan and the two clearing launches. It adds each
// block's DC bits to nb (as enc_dc_bits), writes off (off[0] = 0, off[i + 1] = the bits of blocks 0..i), clears the
// two flags and clears the words of buf that the bits go into (enc_write ORs into them; nothing reads past them).
extern "C" __global__ void __launch_bounds__(SCAN1) enc_plan(const short* __restrict__ coef, const EL* __restrict__ L,
        const HT* __restrict__ T, int nblk, int* __restrict__ nb, i64* __restrict__ off, u32* __restrict__ buf,
        i64 buf_words, int* __restrict__ flags) {
    __shared__ i64 rs[SCAN1];
    __shared__ i64 total;
    const int per = (nblk + SCAN1 - 1) / SCAN1, a = min(nblk, (int)threadIdx.x * per), b = min(nblk, a + per);
    i64 t = 0;
    for (int g = a; g < b; g++) {
        int ci = L->comp[g % L->bpm];
        i64 p = prev_block(L, g);
        int s = nbits(coef[(i64)g * 64] - (p >= 0 ? coef[p * 64] : 0));
        int v = nb[g] + T->size[ci ? 2 : 0][s] + s;
        nb[g] = v;
        t += v;
    }
    rs[threadIdx.x] = t;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int used = min(SCAN1, (nblk + per - 1) / per);       // threads that have blocks
        i64 c = 0;
        for (int k = 0; k < used; k++) { i64 x = rs[k]; rs[k] = c; c += x; }
        total = c;
        off[0] = 0; flags[0] = 0; flags[1] = 0;
    }
    __syncthreads();
    i64 words = total / 32 + 2;
    if (words > buf_words) words = buf_words;
    for (i64 k = threadIdx.x; k < words; k += SCAN1) buf[k] = 0;
    t = rs[threadIdx.x];
    for (int g = a; g < b; g++) { t += nb[g]; off[g + 1] = t; }
}

// One block of 256 threads in place of stuff_count, the scan of the tile counts, stuff_scatter and enc_info: the
// tiles one after the other (a small image has one or a few), the output offset carried along, then the four
// words the host reads back (bytes out, bits, the write overflow flag, the output overflow flag). Each tile is
// handled exactly as stuff_scatter handles it.
extern "C" __global__ void stuff_small(const u32* __restrict__ buf, const i64* __restrict__ total_bits, i64 cap,
        unsigned char* __restrict__ out, i64 out_cap, const int* __restrict__ flags, i64* __restrict__ info) {
    __shared__ int tot;
    __shared__ unsigned char sb[2 * TILE];
    const i64 n = nbytes_of(total_bits, cap);
    i64 g0 = 0;                                        // the same in every thread
    int over = 0;
    for (i64 base = 0; base < n && !over; base += TILE) {
        i64 i0 = base + threadIdx.x * 16;
        u32 v[4] = {0, 0, 0, 0};
        bool whole = i0 + 16 <= n, ff = true;
        int c = 0;
        if (whole) {
            load16w(buf, i0, v);
            ff = has_ff(v);
            c = 16;
            if (ff) for (int k = 0; k < 16; k++) c += ((v[k >> 2] >> (24 - 8 * (k & 3))) & 0xFF) == 0xFF;
        } else {
            for (int k = 0; k < 16; k++) if (i0 + k < n) c += 1 + (byte_at(buf, i0 + k) == 0xFF);
        }
        int ol = block_excl_scan(c, &tot);
        const int K = tot;
        if (g0 + K + 1 > out_cap) { over = 1; continue; }
        for (int k = 0; k < 16; k++) {
            if (i0 + k >= n) break;
            int b = whole ? (v[k >> 2] >> (24 - 8 * (k & 3))) & 0xFF : byte_at(buf, i0 + k);
            sb[ol++] = (unsigned char)b;
            if (ff && b == 0xFF) sb[ol++] = 0;
        }
        __syncthreads();
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
        __syncthreads();                               // sb is free for the next tile
        g0 += K;
    }
    if (threadIdx.x == 0) { info[0] = g0; info[1] = *total_bits; info[2] = flags[0]; info[3] = over; }
}
