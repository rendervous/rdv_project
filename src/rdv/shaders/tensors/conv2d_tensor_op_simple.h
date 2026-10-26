/*
'same'-padded (zero border), stride-1 2D convolution - same "simple" idea
as linear_tensor_op_simple.h (weight row-tile staged once per out-channel
block, reused across every K-chunk; the input tile staged once per
workgroup, reused across every out-channel block) with the reduction
dimension flattened to KH*KW*CIN (every tap's own CIN slice, back to
back) instead of just IN.

x: (H*W)xCIN (channels-last, row-major)
weight: KHxKWxCINxCOUT (channels-last - see Conv2DTensorOp's own docstring)
y: (H*W)xCOUT
b: COUT

Unlike linear_tensor_op_simple.h, KH*KW*CIN ("_TAPK", this op's own "K")
isn't staged all at once - _SH_X/_SH_A are capped to a fixed _CHUNK_K-wide
slice of it (256, chosen so 32*_CHUNK_K + 16*_CHUNK_K float16 elements +
the fixed 32x16 float32 _SH_Y always stays well under 48KB, regardless of
CIN/KH/KW - see this file's own shared array declarations) rather than
scaling with _TAPK, which for a wide in_channels count or a large kernel
was measured to overrun this hardware's per-workgroup shared memory
budget and crash the whole Vulkan device outright (not a catchable
error). When _TAPK <= _CHUNK_K (the common case), _NUM_CHUNKS collapses
to 1 and this behaves exactly like the original single-chunk design (x
staged once, reused across every out-channel block); when it doesn't,
the outer _NUM_CHUNKS loop re-stages x's own _CHUNK_K-wide slice once per
chunk (still reused across every out-channel block *within* that chunk),
and each out-channel block's own running partial sum is carried across
chunks by reading it back from deref(resolve(deref(node).result)) (its
accumulator seeded from bias, broadcast via a stride-0 coopMatLoad, only
on the first chunk) instead of a shared coopmat array (coopmat types
can't be arrayed on this driver - see linear_tensor_op_fast.h).

H/W are generics (bound once, at construction), not runtime struct
fields - see Conv2DTensorOp's own docstring for why.
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _CHUNK_K 256
#define _TAPK (KH * KW * CIN)
#define _NUM_CHUNKS ((_TAPK + _CHUNK_K - 1) / _CHUNK_K)
#define OUT_TILES (COUT / 16)

#define _SH_X RDV_CONCAT(RDV_NODE, _sh_x)
#define _SH_A RDV_CONCAT(RDV_NODE, _sh_a)
shared float16_t _SH_X[32 * _CHUNK_K];
shared float16_t _SH_A[16 * _CHUNK_K];
#endif

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint lane = gl_LocalInvocationID.x;
    int wgTile = int(gl_WorkGroupID.x);
    int start_row = wgTile * 32;

    const int pad_h = KH / 2;
    const int pad_w = KW / 2;

    int flat_idx = start_row + int(lane);
    int h = flat_idx / W;
    int w = flat_idx % W;

    uint64_t bias_base = address(deref(node).bias);
    uint64_t result_base = address(resolve(deref(node).result));

    for (int kc = 0; kc < _NUM_CHUNKS; kc++)
    {
        int k0 = kc * _CHUNK_K;
        int k_width = min(_CHUNK_K, _TAPK - k0);
        bool first_chunk = (kc == 0);
        int k_tiles = k_width / 16;

        // stage this lane's own [k0, k0+k_width) slice of the flattened
        // tap x CIN dimension - (kh, kw, cin) re-derived from the flat k
        // index per element, zero-filled where that tap falls outside
        // the image for this lane's own (h, w).
        for (int lk = 0; lk < k_width; lk++)
        {
            int k = k0 + lk;
            int tap = k / CIN;
            int cin = k % CIN;
            int ih = h + (tap / KW) - pad_h;
            int iw = w + (tap % KW) - pad_w;
            bool valid = (ih >= 0 && ih < H && iw >= 0 && iw < W);
            _SH_X[int(lane) * _CHUNK_K + lk] = valid
                ? float16_t(deref(deref(node).x_buf)[(ih * W + iw) * CIN + cin])
                : float16_t(0.0);
        }
        barrier();

        for (int r = 0; r < OUT_TILES; r++)
        {
            // cache this out-channel block's own [k0, k0+k_width) weight
            // slice into shared memory, once per (chunk, r)
            for (int i = int(lane); i < 16 * k_width; i += 32)
            {
                int n = i / k_width;
                int lk = i % k_width;
                int k = k0 + lk;
                int tap = k / CIN;
                int cin = k % CIN;
                _SH_A[n * _CHUNK_K + lk] = float16_t(deref(deref(node).weight)[(tap * CIN + cin) * COUT + (r * 16 + n)]);
            }
            barrier();

            // seed the accumulator with bias (broadcast across all 32
            // rows via a stride-0 coopMatLoad) on the first chunk, or
            // with this out-block's own running partial sum (already
            // includes bias, written by the previous chunk) otherwise -
            // carries the accumulation across chunks without ever
            // needing a persistent per-r coopmat array.
            coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> mat_acc;
            uint64_t out_ptr = result_base + uint64_t(start_row * COUT + r * 16) * 4;
            if (first_chunk)
                coopMatLoad(mat_acc, deref(cast(F16, bias_base + uint64_t(r * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
            else
                coopMatLoad(mat_acc, deref(cast(F512, out_ptr)), 0, COUT, gl_CooperativeMatrixLayoutRowMajor);

            for (int c = 0; c < k_tiles; c++)
            {
                coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> mat_a;
                coopMatLoad(mat_a, _SH_X, c * 16, _CHUNK_K, gl_CooperativeMatrixLayoutRowMajor);

                // weight's own row-tile is laid out [16 out-channels][_CHUNK_K]
                // (its natural staging order above, not transposed) -
                // ColumnMajor addressing here reads it as the [K, N]
                // matB needs without physically transposing anything
                // (see linear_tensor_op_simple.h's own header comment).
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
                coopMatLoad(mat_b, _SH_A, c * 16, _CHUNK_K, gl_CooperativeMatrixLayoutColumnMajor);

                mat_acc = coopMatMulAdd(mat_a, mat_b, mat_acc);
            }

            coopMatStore(mat_acc, deref(cast(F512, out_ptr)), 0, COUT, gl_CooperativeMatrixLayoutRowMajor);
            memoryBarrierBuffer();
            barrier(); // before the next r (re-staging _SH_A) or chunk (possibly reading this r's own partial sum back) proceeds
        }
    }
#endif
}
