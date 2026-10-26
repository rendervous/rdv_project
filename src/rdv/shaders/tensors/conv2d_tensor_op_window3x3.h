/*
'same'-padded (zero border), stride-1 2D convolution, fixed to a 3x3
kernel/padding=1. One workgroup per 8x8 *tile* of output pixels (WIN=8,
64 threads = 2 subgroups, local id 0..63, local_r = id/8, local_c = id%8),
cooperatively fetching the 10x10 window that tile's own receptive field
needs (FETCH=WIN+2) into shared memory, then sequentially sweeping every
out-channel block r=0..OUT_TILES-1 - the same overall shape as
conv2d_tensor_op_simple3x3.h, but with the window shared/fetched once per
tile instead of gathered redundantly per lane (see this file's own
earlier revisions for that comparison). Output goes through a shared
_SH_Y (coopMatStore'd there, then copied out per lane into result) since
a subgroup's own 32 output pixels span a 4x8 block of the *tile*, not 32
flat-buffer-contiguous rows (H*W is row-major over the whole image, not
the 8x8 tile), so there's no single stride a direct coopMatStore into
result could use.

Two phases (see Conv2DTensorOpWindow3x3.evaluate()/__init__, and
deref(node).phase, read once up front below):

  - phase 0 ("pad"): one thread per *padded* pixel ((H+2)x(W+2) of them),
    run once, up front, over the *whole* image - copies x_buf into temp,
    a (H+2)x(W+2)xCIN scratch buffer (not part of this op's own public
    result) with an explicit, explicitly-zeroed 1-pixel border, instead
    of ever reading x_buf directly. This can't be folded into each
    tile's own phase-1 workgroup the way the original cooperative fetch
    was: adjacent tiles' own windows overlap into each other's padding,
    so the padding has to be fully written, image-wide, before *any*
    tile starts fetching from it.
  - phase 1 ("compute"): the actual convolution, one workgroup per tile,
    as described above - fetches its own 10x10 window straight out of
    temp with plain offset arithmetic only, no bounds check/clamp()
    needed at all (every index it could ever read is already guaranteed
    valid - either real data or the pre-baked zero border).

CIN is assumed <= 64 so _SH_WIN/_SH_A/_SH_XC/_SH_Y together always fit
well under 48KB (see Conv2DTensorOpWindow3x3.__init__ for the exact
check).

H/W are generics (bound once, at construction), not runtime struct
fields - so tiles_w (used as the divisor decoding wgTile into
tile_row/tile_col) is a compile-time constant too, turning that division/
modulo into a cheap strength-reduced multiply instead of a genuine
runtime-variable division - see Conv2DTensorOp's own docstring for why.
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _WIN 8
#define _FETCH 10  // _WIN + 2*pad
#define _WG_THREADS 64  // _WIN * _WIN
#define _TAPK (9 * CIN)
#define _WIN_ELEMS (_FETCH * _FETCH * CIN)
#define _PADDED_W (W + 2)

#define _SH_WIN RDV_CONCAT(RDV_NODE, _sh_win)
#define _SH_A RDV_CONCAT(RDV_NODE, _sh_a)
#define _SH_XC RDV_CONCAT(RDV_NODE, _sh_xc)
#define _SH_Y RDV_CONCAT(RDV_NODE, _sh_y)
shared float16_t _SH_WIN[_WIN_ELEMS];
shared float16_t _SH_A[16 * _TAPK];
shared float16_t _SH_XC[_WG_THREADS * 16];
shared float _SH_Y[_WG_THREADS * 16];
#endif

#define IN_TILES (_TAPK / 16)
#define OUT_TILES (COUT / 16)

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint local = gl_LocalInvocationID.x;

    if (deref(node).phase == 0)
    {
        // phase 0: one thread per padded pixel - copy x_buf's own real
        // data into temp's interior, zero-fill temp's own 1-pixel
        // border, once, for the whole image.
        int idx = int(gl_GlobalInvocationID.x);
        int total_padded = (H + 2) * _PADDED_W;
        if (idx >= total_padded)
            return;

        int prow = idx / _PADDED_W;
        int pcol = idx % _PADDED_W;
        int gh = prow - 1;
        int gw = pcol - 1;
        bool interior = (gh >= 0 && gh < H && gw >= 0 && gw < W);

        uint64_t temp_base = address(deref(node).temp) + uint64_t(idx * CIN) * 4;
        FBUFFER dst = cast(FBUFFER, temp_base);
        if (interior)
        {
            uint64_t x_base = address(deref(node).x_buf) + uint64_t((gh * W + gw) * CIN) * 4;
            FBUFFER src = cast(FBUFFER, x_base);
            TENSOR_OP_UNROLL
            for (int c = 0; c < CIN; c++)
                deref(dst)[c] = deref(src)[c];
        }
        else
        {
            TENSOR_OP_UNROLL
            for (int c = 0; c < CIN; c++)
                deref(dst)[c] = 0.0;
        }
        return;
    }

    // phase 1: one workgroup per tile.
    uint lane = local % 32;
    uint sub = local / 32;
    int local_r = int(local) / _WIN;
    int local_c = int(local) % _WIN;

    int tiles_w = W / _WIN;
    int tile_idx = int(gl_WorkGroupID.x);
    int tile_row = tile_idx / tiles_w;
    int tile_col = tile_idx % tiles_w;
    int h0 = tile_row * _WIN;
    int w0 = tile_col * _WIN;
    int oh = h0 + local_r;
    int ow = w0 + local_c;

    // cooperative fetch: this tile's own 10x10 receptive field, straight
    // out of temp (already padded, already zero-filled at the image
    // border - see this file's own header comment) - no bounds check,
    // no clamp(), no index readjustment of any kind. One row of the
    // window (_FETCH pixels x CIN) is contiguous in temp's own
    // [(H+2)*(W+2), CIN] layout - alias it directly at its own row base
    // pointer (computed once per wr, not per element).
    uint64_t temp_base = address(deref(node).temp);

    TENSOR_OP_UNROLL
    for (int wr = 0; wr < _FETCH; wr++)
    {
        int prow = h0 + wr;
        uint64_t row_base = temp_base + uint64_t((prow * _PADDED_W + w0) * CIN) * 4;
        FBUFFER row_buf = cast(FBUFFER, row_base);

        TENSOR_OP_UNROLL
        for (int j = int(local); j < _FETCH * CIN; j += _WG_THREADS)
            _SH_WIN[wr * _FETCH * CIN + j] = float16_t(deref(row_buf)[j]);
    }
    barrier();

    uint64_t bias_base = address(deref(node).bias);
    uint64_t result_base = address(deref(node).result);

    TENSOR_OP_UNROLL
    for (int r = 0; r < OUT_TILES; r++)
    {
        // cache this out-channel block's own row-tile (16 out-channels x
        // all 9*CIN) into shared memory, once per r
        TENSOR_OP_UNROLL
        for (int i = int(local); i < 16 * _TAPK; i += _WG_THREADS)
        {
            int n = i / _TAPK;
            int k = i % _TAPK;
            _SH_A[i] = float16_t(deref(deref(node).weight)[k * COUT + (r * 16 + n)]);
        }
        barrier();

        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> mat_acc;
        coopMatLoad(mat_acc, deref(cast(F16, bias_base + uint64_t(r * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);

        TENSOR_OP_UNROLL
        for (int c = 0; c < IN_TILES; c++)
        {
            // ensure every lane's own read of the *previous* chunk's
            // _SH_XC is done before this chunk starts overwriting it
            barrier();

            // gather this lane's own 16-element K-chunk straight out of
            // the already-resident window - no global memory traffic.
            TENSOR_OP_UNROLL
            for (int k = 0; k < 16; k++)
            {
                int flat_k = c * 16 + k;
                int tap = flat_k / CIN;
                int cin = flat_k % CIN;
                int wr = local_r + (tap / 3);
                int wc = local_c + (tap % 3);
                _SH_XC[int(local) * 16 + k] = _SH_WIN[(wr * _FETCH + wc) * CIN + cin];
            }
            barrier();

            coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> mat_a;
            coopMatLoad(mat_a, _SH_XC, int(sub) * 32 * 16, 16, gl_CooperativeMatrixLayoutRowMajor);

            // weight's own row-tile is laid out [16 out-channels][9*CIN]
            // (its natural staging order above, not transposed) -
            // ColumnMajor addressing here reads it as the [K, N] matB
            // needs without physically transposing anything (see
            // linear_tensor_op_simple.h's own header comment).
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
            coopMatLoad(mat_b, _SH_A, c * 16, _TAPK, gl_CooperativeMatrixLayoutColumnMajor);

            mat_acc = coopMatMulAdd(mat_a, mat_b, mat_acc);
        }

        // can't coopMatStore straight into result (see this file's own
        // header comment) - store into _SH_Y, then copy out per lane
        // using this lane's own real (oh, ow) flat index. This lane's
        // own 16 out-channels (n=0..15) are contiguous in result - alias
        // it once, at this lane's own row/r base, instead of
        // recomputing (oh*W+ow)*COUT+(r*16+n) from scratch per n.
        coopMatStore(mat_acc, _SH_Y, int(sub) * 32 * 16, 16, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        FBUFFER out_row = cast(FBUFFER, result_base + uint64_t((oh * W + ow) * COUT + r * 16) * 4);
        TENSOR_OP_UNROLL
        for (int n = 0; n < 16; n++)
            deref(out_row)[n] = _SH_Y[int(local) * 16 + n];
        barrier(); // before the next r overwrites _SH_A/_SH_XC/_SH_Y
    }
#endif
}
