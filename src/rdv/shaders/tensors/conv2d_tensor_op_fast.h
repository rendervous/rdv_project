// 'same'-padded (zero border), stride-1 2D convolution via
// GL_KHR_cooperative_matrix (tensor cores) - see linear_tensor_op_fast.h
// for the same idea applied to a plain matmul. Unlike Conv2DTensorOp (one
// 4x8 tile per thread), one 32x16 (M x N) tile per *workgroup*: all 32
// lanes collectively compute this tile, tiled over 16-wide K-chunks of
// in_channels *and* every (kh, kw) tap - each tap/K-chunk pair
// contributes one more coopMatMulAdd into the same running accumulator,
// so the "K" dimension coopMatMulAdd tiles over here is really
// KH*KW*in_channels. Each of the 32 lanes owns exactly one (flattened)
// row throughout (its own (h, w) decoded once, up front), so a tap that
// falls outside the image for a given lane's own (h, w) is handled by
// zero-filling *that lane's own* row of the staged input tile (not
// skipping the tap globally - different lanes can disagree on validity
// for the same tap) before the coopMatLoad, rather than Conv2DTensorOp's
// own per-lane skip (coopMatLoad always reads the *whole* staged tile,
// there's no per-row early-out once it's issued).
//
// x/weight staged as float16 (this class of tensor core doesn't support
// float32 operands at all - see LinearFast's own writeup); the
// accumulator stays float32. No occupancy fallback needed - shape is
// required to be an exact multiple of the 32x16 tile, so every
// dispatched workgroup is always fully, uniformly occupied. H/W are
// generics (bound once, at construction), not runtime struct fields -
// see Conv2DTensorOp's own docstring for why.
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _SH_X RDV_CONCAT(RDV_NODE, _sh_x)
#define _SH_W RDV_CONCAT(RDV_NODE, _sh_w)
#define _SH_Y RDV_CONCAT(RDV_NODE, _sh_y)
shared float16_t _SH_X[32 * 16];
shared float16_t _SH_W[16 * 16];
shared float _SH_Y[32 * 16];
#endif

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint lane = gl_LocalInvocationID.x;
    int wg_tile = int(gl_WorkGroupID.x);

    int n_tiles = COUT / 16;
    int row_tile = wg_tile / n_tiles;
    int n_tile = wg_tile % n_tiles;
    int r0 = row_tile * 32;
    int c0 = n_tile * 16;
    int k_tiles = CIN / 16;
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;

    int flat_idx = r0 + int(lane);
    int h = flat_idx / W;
    int w = flat_idx % W;

    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);

    for (int kh = 0; kh < KH; kh++)
    {
        int ih = h + kh - pad_h;
        for (int kw = 0; kw < KW; kw++)
        {
            int iw = w + kw - pad_w;
            bool valid = (ih >= 0 && ih < H && iw >= 0 && iw < W);
            int in_row = valid ? (ih * W + iw) : 0;

            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k0 = kt * 16;

                for (int k = 0; k < 16; k++)
                {
                    _SH_X[lane * 16 + k] = valid
                        ? float16_t(deref(resolve(deref(node).x_buf))[in_row * CIN + (k0 + k)])
                        : float16_t(0.0);
                }
                barrier();
                for (int idx = int(lane); idx < 16 * 16; idx += 32)
                {
                    int kk = idx / 16;
                    int nn = idx % 16;
                    _SH_W[idx] = float16_t(deref(resolve(deref(node).weight))[((kh * KW + kw) * CIN + (k0 + kk)) * COUT + (c0 + nn)]);
                }
                barrier();

                coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> matA;
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
                coopMatLoad(matA, _SH_X, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(matB, _SH_W, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
                acc = coopMatMulAdd(matA, matB, acc);
                barrier();
            }
        }
    }

    coopMatStore(acc, _SH_Y, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    for (int n = 0; n < 16; n++)
    {
        deref(resolve(deref(node).result))[(r0 + int(lane)) * COUT + (c0 + n)] =
            _SH_Y[lane * 16 + n] + deref(resolve(deref(node).bias))[c0 + n];
    }
#endif
}
