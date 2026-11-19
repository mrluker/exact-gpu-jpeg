// exact-gpu-jpeg batch kernels: many JPEGs decoded by one set of launches. Compiled after _kernels.cu (same module
// source), whose device functions and structs they use; each kernel here is the single-image kernel of the same
// name with the image looked up first, so the arithmetic (and every output byte) is the same.
//
// THIS FILE IS A MODIFIED PORT, NOT ORIGINAL libjpeg-turbo CODE. It holds re-implementations for CUDA of routines
// from libjpeg-turbo 3.1.2, which is based on the Independent JPEG Group's software:
//   Copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
//   libjpeg-turbo modifications: Copyright (C) 2009-2025 D. R. Commander and the libjpeg-turbo contributors.
// The ported parts stay under the IJG License (licenses/README.ijg; see also licenses/libjpeg-turbo-LICENSE.md).
// THIRD_PARTY_NOTICES.md names each routine, its upstream file and what was changed.
//
// idct_islow_b and color_b carry the arithmetic ported from libjpeg-turbo 3.1.2 (jidctint.c, jdsample.c, jdmerge.c,
// jdcolor.c; IJG License, licenses/README.ijg; THIRD_PARTY_NOTICES.md lists what was changed).
//
// A batch holds images that share their Huffman tables and MCU layout (Hdr). It is decoded as ONE stream:
//  - entropy-coded data: image i's bytes at a TILE-aligned offset of one buffer, so every tile (one CUDA block of
//    the unstuffing kernels) belongs to one image;
//  - the unstuffed bytes of all images, one after the other, form one bit stream with global bit positions;
//  - segments: image i has its nseg restart intervals (1 without restart markers) at seg[segbase ..], then one
//    empty "gap" segment covering the bits between its end and the next image's first bit. A chunk never crosses
//    a segment, every segment starts in a known decoder state, so sync_pass runs unchanged and images cannot
//    influence each other;
//  - JPEG blocks: image i's blocks at bbase (a multiple of 32 and of the blocks per MCU);
//  - errors are per image (err[image]): a bad image is declined, the others are not affected.
// Bounds: every index is bounded by the host-built tables below (sizes from the headers); a block number outside
// its own segment's range is an error and is not written.

struct BImg {
    int rb;                 // byte offset of the image's entropy-coded data in the batch buffer (TILE-aligned)
    int ne;                 // its length
    int dri;                // restart interval in MCUs (0: none)
    int tile0;              // first tile
    int segbase;            // first segment
    int nseg;               // restart intervals (the gap segment is segbase + nseg)
    int W, H;
    u32 bbase;              // first JPEG block
    u32 nblk;               // JPEG blocks
    u32 ubase;              // first colour unit (CPX pixels of one row)
    int xb;                 // colour units per row = ceil(W / CPX)
    i64 pbase;              // offset of its planes
    i64 obase;              // offset of its output (bytes for uint8, elements for float output)
};
#define SEG_GAP (1 << 30)   // flag in simg[]: the empty segment after an image
#define SEG_IMG(v) ((v) & (SEG_GAP - 1))

// per tile: kept bytes and RST markers; end[image] = offset of the first other marker's 0xFF (starts at ne)
extern "C" __global__ void unstuff_count_b(const unsigned char* __restrict__ raw, const BImg* __restrict__ D,
        const int* __restrict__ timg, int* __restrict__ tilecnt, int* __restrict__ tilerst, int* __restrict__ end) {
    __shared__ int tot;
    const int im = timg[blockIdx.x];
    const unsigned char* b = raw + D[im].rb;
    const int n = D[im].ne, dri = D[im].dri;
    int i0 = ((int)blockIdx.x - D[im].tile0) * TILE + threadIdx.x * 16, c = 0;
    Bytes16 x;
    load16(b, n, i0, x);
    unsigned km, rm, mm;
    classify16(x, n, i0, dri, km, rm, mm);
    c = __popc(km) + (__popc(rm) << 16);
    if (mm) atomicMin(end + im, i0 + __ffs(mm) - 1);
    block_excl_scan(c, &tot);
    if (threadIdx.x == 0) { tilecnt[blockIdx.x] = tot & 0xFFFF; tilerst[blockIdx.x] = tot >> 16; }
}

// toff/roff = exclusive scans of tilecnt/tilerst over the whole batch. As unstuff_scatter, per image: the kept
// bytes before its end marker go to out[] at the batch-wide offset, nbits[image] = the bit position of its end,
// seg[segbase] = its first bit, seg[segbase + r + 1] = the bit offset of its RST number r, found[image] = RSTs
// before its end marker. Zeros: the rest of each tile's output range, and from the last kept byte to nout.
extern "C" __global__ void unstuff_scatter_b(const unsigned char* __restrict__ raw, const BImg* __restrict__ D,
        const int* __restrict__ timg, const int* __restrict__ toff, const int* __restrict__ roff,
        const int* __restrict__ end, unsigned char* __restrict__ out, u32* __restrict__ nbits, u32* __restrict__ seg,
        int* __restrict__ found, int* __restrict__ err, int nout) {
    __shared__ int tot;
    const int im = timg[blockIdx.x];
    const unsigned char* b = raw + D[im].rb;
    const int n = D[im].ne, dri = D[im].dri, nseg = D[im].nseg, tile0 = D[im].tile0;
    u32* sg = seg + D[im].segbase;
    int i0 = ((int)blockIdx.x - tile0) * TILE + threadIdx.x * 16, c = 0;
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
    int r = roff[blockIdx.x] - roff[tile0] + (sx >> 16);        // RSTs of this image before this thread's bytes
    int e = end[im];
    if ((int)blockIdx.x == tile0 && threadIdx.x == 0) sg[0] = (u32)g0 * 8u;      // the image's first bit
    int last = 0;
    for (int k = 0; k < 16; k++) {
        int i = i0 + k;
        if (i == e) nbits[im] = (u32)(g0 + ol) * 8u;
        if (i >= n) break;
        if (rmask >> k & 1) {
            if (i < e) {
                if (r < nseg - 1) sg[r + 1] = (u32)(g0 + ol) * 8u;
                if ((x.v[k + 2] & 7) != (r & 7) || r >= nseg - 1) atomicOr(err + im, ERR_RST);
                atomicMax(found + im, r + 1);
            }
            r++;
        }
        if (kmask >> k & 1) { if (i < e) { sb[ol] = x.v[k + 1]; last = ol + 1; } ol++; }
    }
    if (last) atomicMax(&kw, last);               // kept bytes before the end marker are a prefix of the tile's
    __syncthreads();
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

// The chunk table of the batch in one block of SCAN1 threads (as chunk_table). NS segments in all (gaps included),
// simg[k] = image of segment k (| SEG_GAP), total = kept bytes of the whole batch. A gap segment gets one empty
// chunk. cfirst: NS + 1 ints.
extern "C" __global__ void __launch_bounds__(SCAN1) chunk_table_b(u32* __restrict__ seg, const int* __restrict__ simg,
        int NS, const BImg* __restrict__ D, const int* __restrict__ found, const u32* __restrict__ nbits,
        int* __restrict__ err, const int* __restrict__ total, int S, int* __restrict__ cfirst,
        u32* __restrict__ cstart, u32* __restrict__ cend, int* __restrict__ cseg, int nmax) {
    __shared__ int wsum[32];
    for (int k = threadIdx.x; k <= NS; k += SCAN1) {                 // fix_segs, per image
        if (k == NS) { seg[k] = (u32)(*total) * 8u; continue; }
        int im = SEG_IMG(simg[k]);
        int l = k - D[im].segbase, ns = D[im].nseg;                  // l == ns: the gap, which starts at the image's end
        if (l == 0) continue;
        if (l == ns || l > found[im]) seg[k] = nbits[im];
        if (l == 1 && found[im] != ns - 1) atomicOr(err + im, ERR_RST);
    }
    __syncthreads();
    if (threadIdx.x == 0) cfirst[0] = 0;
    int run = 0;
    for (int base = 0; base < NS; base += SCAN1) {                    // chunks per segment + inclusive scan
        int i = base + threadIdx.x, c = 0;
        if (i < NS) {
            i64 len = (i64)seg[i + 1] - (i64)seg[i];
            i64 q = (len + S - 1) / S;
            if ((len + S - 1) < 0 && (len + S - 1) % S) q -= 1;
            c = (int)(q < 1 ? 1 : q);
            if (simg[i] & SEG_GAP) c = 1;
        }
        int tot;
        int x = block_incl_scan1(c, wsum, &tot);
        if (i < NS) cfirst[i + 1] = x + run;
        run += tot;
    }
    __syncthreads();
    int nch = cfirst[NS];
    for (int i = threadIdx.x; i < nmax; i += SCAN1) {                 // make_chunks; cseg = 0 past the chunks
        if (i >= nch) { cseg[i] = 0; continue; }
        int lo = 0, hi = NS - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (cfirst[mid] <= i) lo = mid; else hi = mid - 1; }
        u32 a = seg[lo] + (u32)(i - cfirst[lo]) * (u32)S;
        cstart[i] = a;
        cend[i] = (simg[lo] & SEG_GAP) ? a : min(a + (u32)S, seg[lo + 1]);
        cseg[i] = lo;
    }
}

// decode_write for a batch: segment k holds blocks [segblk[k], seglim[k]) (a gap: none); errors go to the
// segment's image. A block number outside the segment's own range is never written.
extern "C" __global__ void decode_write_b(const u32* __restrict__ W, u32 nw, const int* __restrict__ tabs,
        const Hdr* __restrict__ h, const int* __restrict__ nch_p, const u32* __restrict__ cstart,
        const u32* __restrict__ cend, const int* __restrict__ cseg, const u32* __restrict__ seg,
        const int* __restrict__ cfirst, const u64* __restrict__ st_in, const u64* __restrict__ cps,
        const int* __restrict__ cpc, int S, const i64* __restrict__ first, const i64* __restrict__ segblk,
        const i64* __restrict__ seglim, const int* __restrict__ simg,
        short* __restrict__ coef, int* __restrict__ dcd, unsigned char* __restrict__ nq, int* __restrict__ err, int nmax,
        int ntab, int NCP) {
    Tabs T;
    load_tabs(&T, tabs, ntab);
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gi / (NCP + 1), piece = gi % (NCP + 1);
    if (i >= *nch_p || i >= nmax) return;
    int k = cseg[i];
    const i64 lo = segblk[k], lim = seglim[k];
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
        if (B < lo) { e |= ERR_SHORT; break; }
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
    if (e) atomicOr(err + SEG_IMG(simg[k]), e);
}

// DC values from running sums. cs = inclusive running sum (int32, wrapping) over one component's DC differences
// of the whole batch in block order (w blocks per MCU; padding blocks between images included: whatever they
// hold cancels in the difference). DC = cs[e] - cs[just before the block's restart interval (or image)].
// a = the component's first block within the MCU.
extern "C" __global__ void dc_apply_b(const int* __restrict__ cs, int w, int a, int bpm, const BImg* __restrict__ D,
        int nimg, int* __restrict__ dcd, u32 nel) {
    u32 e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nel) return;
    u32 m = e / (u32)w;
    u32 gb = m * (u32)bpm;                                 // the MCU's first block
    int lo = 0, hi = nimg - 1;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (D[mid].bbase <= gb) lo = mid; else hi = mid - 1; }
    u32 lb = gb - D[lo].bbase;
    if (lb >= D[lo].nblk) return;                          // padding
    u32 ml = lb / (u32)bpm, m0 = D[lo].bbase / (u32)bpm;
    u32 dri = (u32)D[lo].dri;
    u32 ms = m0 + (dri ? (ml / dri) * dri : 0u);           // first MCU of the restart interval
    u32 eb = ms * (u32)w;
    int base = eb ? cs[eb - 1] : 0;
    dcd[(size_t)gb + a + (e - m * (u32)w)] = (int)((u32)cs[e] - (u32)base);
}

// idct_islow for a batch: a CUDA block's 32 JPEG blocks belong to one image (bbase is a multiple of 32).
// q: [nimg][3][64], P: [nimg].
extern "C" __global__ void idct_islow_b(const short* __restrict__ coef, const unsigned char* __restrict__ nq,
        const int* __restrict__ dc, const int* __restrict__ q, const Pix* __restrict__ Pa, const BImg* __restrict__ D,
        int nimg, unsigned char* __restrict__ planes, int* __restrict__ err) {
    __shared__ int ws[32][64];
    __shared__ __align__(16) short sc[32][64];
    __shared__ int s_im;
    if (threadIdx.x == 0) {
        u32 g0 = blockIdx.x * 32u;
        int lo = 0, hi = nimg - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (D[mid].bbase <= g0) lo = mid; else hi = mid - 1; }
        s_im = lo;
    }
    __syncthreads();
    const int im = s_im;
    const Pix* P = Pa + im;
    int t = threadIdx.x & 7, lb = threadIdx.x >> 3;
    u32 g = blockIdx.x * 32u + (u32)lb;                    // block number in the batch
    u32 gl = g - D[im].bbase;                              // ... in its image
    bool ok = gl < D[im].nblk;
    u32 bpm = (u32)P->bpm;
    int j = ok ? (int)(gl % bpm) : 0;
    u32 m = ok ? gl / bpm : 0;
    int comp = P->comp[j];
    const int* qq = q + (size_t)im * 192 + comp * 64;
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
    if (bad) atomicOr(err + im, ERR_RANGE);
    u32 mcux = (u32)P->mcux;
    i64 by = (i64)(m / mcux) * P->vs[comp] + P->dy[j], bx = (i64)(m % mcux) * P->hs[comp] + P->dx[j];
    unsigned char* dst = planes + D[im].pbase + P->off[comp] + (by * 8 + t) * P->pitch[comp] + bx * 8;
    uint2 v;
    v.x = px[0] | (px[1] << 8) | (px[2] << 16) | ((u32)px[3] << 24);
    v.y = px[4] | (px[5] << 8) | (px[6] << 16) | ((u32)px[7] << 24);
    *(uint2*)dst = v;
}

// color / color_ep for a batch: one CUDA block = one colour unit (CPX pixels of one row of one image). E == 0:
// uint8 (H, W, channels) per image at out + obase (bytes); else the epilogue's float output at obase (elements).
extern "C" __global__ void color_b(const unsigned char* __restrict__ planes, const Pix* __restrict__ Pa,
        const Up* __restrict__ Ua, const BImg* __restrict__ D, int nimg, int mode, void* __restrict__ out,
        const Epi* __restrict__ E) {
    __shared__ __align__(16) unsigned char sh[CPX * 3];
    __shared__ RowCtx R[3];
    __shared__ int s_im;
    if (threadIdx.x == 0) {
        u32 u0 = blockIdx.x;
        int lo = 0, hi = nimg - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (D[mid].ubase <= u0) lo = mid; else hi = mid - 1; }
        s_im = lo;
    }
    __syncthreads();
    const int im = s_im;
    const Pix* P = Pa + im;
    const Up* U = Ua + im;
    const int W = D[im].W, H = D[im].H, xb = D[im].xb;
    u32 u = blockIdx.x - D[im].ubase;
    if (u >= (u32)xb * (u32)H) return;                       // (the whole block: nothing after the last unit)
    int y = (int)(u / (u32)xb), x0 = (int)(u % (u32)xb) * CPX;
    const unsigned char* pl = planes + D[im].pbase;
    int nc = mode >= 2 ? 1 : 3;
    if (threadIdx.x < nc) row_ctx(&R[threadIdx.x], pl + P->off[threadIdx.x], P->pitch[threadIdx.x], U, threadIdx.x, y);
    __syncthreads();
    RowCtx r0 = R[0], r1 = R[0], r2 = R[0];
    if (nc == 3) { r1 = R[1]; r2 = R[2]; }
    int C = mode == 2 ? 1 : 3;
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
    if (!E) {
        unsigned char* dst = (unsigned char*)out + D[im].obase + ((i64)y * W + x0) * C;
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
        return;
    }
    float sc[3], bi[3];
    for (int c = 0; c < 3; c++) { sc[c] = E->scale[c]; bi[c] = E->bias[c]; }
    const int f16 = E->f16, chw = E->chw;
    const i64 ob = D[im].obase;
    for (int k = threadIdx.x; k < nb; k += 128) {
        int p = C == 3 ? k / 3 : k, c = k - p * C;
        float v = __fadd_rn(__fmul_rn((float)sh[k], sc[c]), bi[c]);
        i64 x = x0 + p;
        i64 idx = ob + (chw ? ((i64)c * H + y) * W + x : ((i64)y * W + x) * C + c);
        if (f16) {
            unsigned short hv;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(v));
            ((unsigned short*)out)[idx] = hv;
        } else {
            ((float*)out)[idx] = v;
        }
    }
}
