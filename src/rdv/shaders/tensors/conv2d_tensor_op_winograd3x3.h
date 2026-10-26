/*
'same'-padded (zero border), stride-1 2D convolution, fixed to a 3x3
kernel/padding=1, via Winograd F(2x2, 3x3): instead of the KHxKWxCIN=9*CIN
direct-GEMM reduction every other Conv2D op here does, a 2x2 output tile
is produced from a 4x4 input tile by transforming both into a 4x4
"Winograd domain" (16 positions), doing one CIN-reduction GEMM per
domain position (coopMatMulAdd, same as every other op here - the actual
tensor-core work), then inverse-transforming the 16 per-position results
back into the 2x2 output - 16 GEMMs of reduction-width CIN instead of a
single GEMM of reduction-width 9*CIN, at the cost of the (cheap, pure
add/sub, no multiplies) forward/inverse transforms themselves. See
https://arxiv.org/abs/1509.09308 (Lavin & Gray) for the general algorithm;
this file's own transform formulas are verified against a direct (non-
Winograd) reference - see the class docstring - not hand-derived from the
"fast" literature formulas, to keep sign/index mistakes out of the shader
itself.

Weight is pretransformed once, at construction (see
Conv2DTensorOpWinograd3x3._winograd_transform_weight): U = G @ g @ G^T
per (cin, cout) 3x3 filter, flattened to a [16][CIN][COUT] float16
buffer (domain-position-major) - read directly (RowMajor, no transpose
trick needed, since this layout is already [K, N] the way matB needs).

One workgroup covers 32 *tiles* (not pixels) - one lane per 2x2 output
tile, flat-indexed over (H/2)*(W/2) tiles the same way every non-windowed
Conv2D op here flat-indexes pixels. For each domain position p (the
OUTER loop, not CIN - see this file's own shared array declarations),
every lane re-reads its own 4x4xCIN input patch from global memory
(redundant across the 16 domain positions - the alternative, caching the
whole patch once, doesn't fit shared memory alongside the 16-domain-
position M scratch this algorithm also needs resident - see
Conv2DTensorOpWinograd3x3's own docstring) and keeps only that p's own
transformed CIN-vector (_SH_V), then accumulates over CIN in 16-wide
chunks against U's own (p, chunk) slice (_SH_U) into a per-domain-
position running sum (_SH_M[p] - float32, all 16 positions resident at
once, since the final inverse transform needs all 16 simultaneously per
(tile, out-channel)).

H/W are generics (bound once, at construction), not runtime struct
fields - so W_tiles (used as the divisor decoding tile_idx into
tile_row/tile_col) is a compile-time constant too - see Conv2DTensorOp's
own docstring for why.
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _SH_V RDV_CONCAT(RDV_NODE, _sh_v)
#define _SH_U RDV_CONCAT(RDV_NODE, _sh_u)
#define _SH_M RDV_CONCAT(RDV_NODE, _sh_m)
shared float16_t _SH_V[32 * CIN];
shared float16_t _SH_U[16 * 16];
shared float _SH_M[16 * 32 * 16];
#endif

#define CIN_TILES (CIN / 16)
#define OUT_TILES (COUT / 16)

void _winograd_fwd(in float d[16], out float V[16])
{
    float m[16];
    for (int j = 0; j < 4; j++)
    {
        float d0 = d[0 * 4 + j], d1 = d[1 * 4 + j], d2 = d[2 * 4 + j], d3 = d[3 * 4 + j];
        m[0 * 4 + j] = d0 - d2;
        m[1 * 4 + j] = d1 + d2;
        m[2 * 4 + j] = d2 - d1;
        m[3 * 4 + j] = d1 - d3;
    }
    for (int i = 0; i < 4; i++)
    {
        float m0 = m[i * 4 + 0], m1 = m[i * 4 + 1], m2 = m[i * 4 + 2], m3 = m[i * 4 + 3];
        V[i * 4 + 0] = m0 - m2;
        V[i * 4 + 1] = m1 + m2;
        V[i * 4 + 2] = m2 - m1;
        V[i * 4 + 3] = m1 - m3;
    }
}

void _winograd_inv(in float M[16], out float Y[4])
{
    float t[8];
    for (int j = 0; j < 4; j++)
    {
        t[0 * 4 + j] = M[0 * 4 + j] + M[1 * 4 + j] + M[2 * 4 + j];
        t[1 * 4 + j] = M[1 * 4 + j] - M[2 * 4 + j] - M[3 * 4 + j];
    }
    for (int i = 0; i < 2; i++)
    {
        Y[i * 2 + 0] = t[i * 4 + 0] + t[i * 4 + 1] + t[i * 4 + 2];
        Y[i * 2 + 1] = t[i * 4 + 1] - t[i * 4 + 2] - t[i * 4 + 3];
    }
}

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint lane = gl_LocalInvocationID.x;
    int wgTile = int(gl_WorkGroupID.x);
    int W_tiles = W / 2;

    int tile_idx = wgTile * 32 + int(lane);
    int tile_row = tile_idx / W_tiles;
    int tile_col = tile_idx % W_tiles;
    int oh0 = tile_row * 2;
    int ow0 = tile_col * 2;
    float d[16];
    float M[16];
    float Y[4];
    float V[16];
    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> mat_acc;

    for (int r = 0; r < OUT_TILES; r++)
    {
        for (int p = 0; p < 16; p++)
        {
            barrier(); // previous p's own coopMatLoad(matA, _SH_V, ...) reads must finish before we overwrite it

            // this lane's own transformed value at domain position p,
            // for every input channel - re-derives the raw 4x4 patch
            // fresh from global memory (see this file's own header
            // comment for why).
            for (int c = 0; c < CIN; c++)
            {
                for (int dr = 0; dr < 4; dr++)
                {
                    int gh = oh0 - 1 + dr;
                    for (int dc = 0; dc < 4; dc++)
                    {
                        int gw = ow0 - 1 + dc;
                        bool valid = (gh >= 0 && gh < H && gw >= 0 && gw < W);
                        d[dr * 4 + dc] = valid ? deref(deref(node).x_buf)[(gh * W + gw) * CIN + c] : 0.0;
                    }
                }
                _winograd_fwd(d, V);
                _SH_V[int(lane) * CIN + c] = float16_t(V[p]);
            }
            barrier();

            mat_acc =
                coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);
            for (int c = 0; c < CIN_TILES; c++)
            {
                barrier(); // previous chunk's own coopMatLoad(matB, _SH_U, ...) read must finish before we overwrite it

                for (int i = int(lane); i < 16 * 16; i += 32)
                {
                    int kk = i / 16;
                    int nn = i % 16;
                    _SH_U[i] = deref(deref(node).weight_u)[(p * CIN + (c * 16 + kk)) * COUT + (r * 16 + nn)];
                }
                barrier();

                coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> mat_a;
                coopMatLoad(mat_a, _SH_V, c * 16, CIN, gl_CooperativeMatrixLayoutRowMajor);
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
                coopMatLoad(mat_b, _SH_U, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
                mat_acc = coopMatMulAdd(mat_a, mat_b, mat_acc);
            }

            coopMatStore(mat_acc, _SH_M, p * 32 * 16, 16, gl_CooperativeMatrixLayoutRowMajor);
        }
        barrier(); // every _SH_M[p] must be fully written before the inverse transform below reads them all
        for (int n = 0; n < 16; n++)
        {
            for (int p = 0; p < 16; p++)
                M[p] = _SH_M[p * 32 * 16 + int(lane) * 16 + n];
            _winograd_inv(M, Y);
            float bias_val = deref(deref(node).bias)[r * 16 + n];
            deref(resolve(deref(node).result))[(oh0 * W + ow0) * COUT + (r * 16 + n)] = Y[0] + bias_val;
            deref(resolve(deref(node).result))[(oh0 * W + ow0 + 1) * COUT + (r * 16 + n)] = Y[1] + bias_val;
            deref(resolve(deref(node).result))[((oh0 + 1) * W + ow0) * COUT + (r * 16 + n)] = Y[2] + bias_val;
            deref(resolve(deref(node).result))[((oh0 + 1) * W + ow0 + 1) * COUT + (r * 16 + n)] = Y[3] + bias_val;
        }
        barrier(); // before the next r overwrites _SH_M
    }
#endif
}
