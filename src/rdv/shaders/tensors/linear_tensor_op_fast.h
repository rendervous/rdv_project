// y = W @ x + b via GL_KHR_cooperative_matrix (tensor cores), register-
// tiled 2x2 accumulators per workgroup (each workgroup call covers 64
// rows x 32 out_features - hand-unrolled instead of arrayed, since this
// driver's glslang rejects arrays of coopmat types). weight is
// rearranged once, at op construction time (see LinearTensorOpFast's own
// __init__/_tile_weight_fp16), into a contiguous, pre-converted-to-
// float16 [n_tile][k_tile][16][16] buffer - matB is coopMatLoad'd
// directly from it in a single call (address(...) + a byte offset,
// cast to a fixed-size F256 = FixArray[FLOAT16,256] view of exactly one
// tile), no shared memory/staging/barrier for weight at all. x still
// needs float16 staging in shared memory (it isn't pre-tiled - it's the
// actual runtime input, arbitrary row layout, changing every dispatch),
// so it keeps its own barriers.
//
// x_buf/weight_tiled/bias are plain (non-Deferred) Buffer fields - see
// LinearTensorOpFast's own docstring - so every access below is a direct
// deref(), no resolve() indirection.
//
// This class of tensor core only supports float16 A/B operands (float32
// compiles and runs but silently produces wrong results - see
// LinearFast's own writeup) - the accumulator stays float32. No
// occupancy fallback needed - shape is required to be an exact multiple
// of the 64x32 tile, so every dispatched workgroup is always fully,
// uniformly occupied.
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _SH_X0 RDV_CONCAT(RDV_NODE, _sh_x0)
#define _SH_X1 RDV_CONCAT(RDV_NODE, _sh_x1)
#define _SH_Y RDV_CONCAT(RDV_NODE, _sh_y)
shared float16_t _SH_X0[32 * 16];
shared float16_t _SH_X1[32 * 16];
shared float _SH_Y[32 * 16];
#endif

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint lane = gl_LocalInvocationID.x;
    int wg_tile = int(gl_WorkGroupID.x);

    int out_cols = deref(node).shape[1];
    int n_tiles = out_cols / 16;
    int n_super_tiles = n_tiles / 2;
    int row_tile = wg_tile / n_super_tiles;
    int n_super = wg_tile % n_super_tiles;
    int r0 = row_tile * 64;
    int c0 = n_super * 32;
    int k_tiles = INF / 16;
    int n_tile0 = c0 / 16;
    int n_tile1 = n_tile0 + 1;

    uint64_t weight_base = address(deref(node).weight_tiled);

    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> acc00 =
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> acc01 =
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> acc10 =
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> acc11 =
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);

    for (int kt = 0; kt < k_tiles; kt++)
    {
        int k0 = kt * 16;

        for (int k = 0; k < 16; k++)
        {
            _SH_X0[lane * 16 + k] = float16_t(deref(deref(node).x_buf)[(r0 + int(lane)) * INF + (k0 + k)]);
            _SH_X1[lane * 16 + k] = float16_t(deref(deref(node).x_buf)[(r0 + 32 + int(lane)) * INF + (k0 + k)]);
        }
        barrier();

        coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> matA0;
        coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> matA1;
        coopMatLoad(matA0, _SH_X0, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matA1, _SH_X1, 0, 16, gl_CooperativeMatrixLayoutRowMajor);

        uint64_t tile_ptr0 = weight_base + uint64_t((n_tile0 * k_tiles + kt) * 256) * 2;
        uint64_t tile_ptr1 = weight_base + uint64_t((n_tile1 * k_tiles + kt) * 256) * 2;
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB0;
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB1;
        coopMatLoad(matB0, deref(cast(F256, tile_ptr0)), 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matB1, deref(cast(F256, tile_ptr1)), 0, 16, gl_CooperativeMatrixLayoutRowMajor);

        acc00 = coopMatMulAdd(matA0, matB0, acc00);
        acc10 = coopMatMulAdd(matA1, matB0, acc10);
        acc01 = coopMatMulAdd(matA0, matB1, acc01);
        acc11 = coopMatMulAdd(matA1, matB1, acc11);
        barrier();
    }

    coopMatStore(acc00, _SH_Y, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (int n = 0; n < 16; n++)
        deref(resolve(deref(node).result))[(r0 + int(lane)) * out_cols + (c0 + n)] =
            _SH_Y[lane * 16 + n] + deref(deref(node).bias)[c0 + n];
    barrier();

    coopMatStore(acc01, _SH_Y, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (int n = 0; n < 16; n++)
        deref(resolve(deref(node).result))[(r0 + int(lane)) * out_cols + (c0 + 16 + n)] =
            _SH_Y[lane * 16 + n] + deref(deref(node).bias)[c0 + 16 + n];
    barrier();

    coopMatStore(acc10, _SH_Y, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (int n = 0; n < 16; n++)
        deref(resolve(deref(node).result))[(r0 + 32 + int(lane)) * out_cols + (c0 + n)] =
            _SH_Y[lane * 16 + n] + deref(deref(node).bias)[c0 + n];
    barrier();

    coopMatStore(acc11, _SH_Y, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (int n = 0; n < 16; n++)
        deref(resolve(deref(node).result))[(r0 + 32 + int(lane)) * out_cols + (c0 + 16 + n)] =
            _SH_Y[lane * 16 + n] + deref(deref(node).bias)[c0 + 16 + n];
#endif
}
