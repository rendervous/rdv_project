/*
Resolves y = x * A^T + b
y: Ynm, A^T: Akm, x: Xnk  b: Bk

y tensor is split in tiles 32x16

Use cooperative matrices of a 32x16 b 16x16 c 32x16

tiles_m = ceil(m/32)
tiles_n = ceil(n/16)

Adapted from jeffbolznv/vk_cooperative_matrix_perf's own shmem.comp
(https://github.com/jeffbolznv/vk_cooperative_matrix_perf/blob/master/shaders/shmem.comp)
- one subgroup per workgroup here (lane == A's own row), not the
reference's own multi-subgroup-per-workgroup tiling (this project's own
convention throughout is 32 threads = 1 workgroup = 1 subgroup), and a
single M x N tile per workgroup (no C_ROWS/C_COLS register tiling - see
linear_tensor_op_fast.h for that, separately). Keeps the reference's
double-buffered prefetch (each K-chunk's global reads for the *next*
iteration are issued right after this iteration's own values, already
fetched last iteration, are published to shared memory, so those
next-chunk loads can run in the background while this iteration's
coopMatMulAdd executes, instead of stalling on them up front) - but NOT
its row-padded shared memory (the reference's own bank-conflict-avoidance
trick): padding _SH_A/_SH_B's own row stride by even one float16_t
element (_PAD=1) was tried and measured to silently produce wrong
results here (values only correct for K=16, i.e. a single K-chunk with
no re-staging - anything requiring 2+ K-chunks came out wrong) - an odd
element stride for a float16_t coopMatLoad source apparently isn't
something this driver/hardware handles correctly, unlike a plain (even)
stride. Left at _PAD=0 (no padding) - correct, unlike the reference's own
choice for 16-bit types.

gl_WorkGroupID.x is the actual (shared, workgroup-wide) tile index,
unflattened against OUTF/16 (a compile-time constant - INF/OUTF are
generics, bound once at construction, not runtime struct fields).
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _PAD 0
#define _SH_A RDV_CONCAT(RDV_NODE, _sh_a)
#define _SH_B RDV_CONCAT(RDV_NODE, _sh_b)
#define _SH_Y RDV_CONCAT(RDV_NODE, _sh_y)
shared float16_t _SH_A[32 * (16 + _PAD)];
shared float16_t _SH_B[16 * (16 + _PAD)];
shared float _SH_Y[32 * 16];
#endif

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint lane = gl_LocalInvocationID.x;
    int wgTile = int(gl_WorkGroupID.x);
    int tileRow = wgTile / (OUTF / 16);
    int tileCol = wgTile % (OUTF / 16);
    int r0 = tileRow * 32;
    int c0 = tileCol * 16;
    int K = INF;
    int k_tiles = K / 16;

    // Fetch chunk 0 of A (this lane's own row of 16) and B (this lane's
    // own 8-element slice - 32 lanes x 8 = 256 = 16x16) into registers,
    // ahead of the K-loop, so the loop body always starts with an
    // already-fetched chunk ready to publish.
    float tempA[16];
    for (int k = 0; k < 16; k++)
        tempA[k] = deref(deref(node).x_buf)[(r0 + int(lane)) * K + k];

    float tempB[8];
    for (int e = 0; e < 8; e++)
    {
        int idx = int(lane) + e * 32;
        int kk = idx / 16;
        int nn = idx % 16;
        tempB[e] = deref(deref(node).weight)[(c0 + nn) * K + kk];
    }

    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>(0.0);

    for (int kt = 0; kt < k_tiles; kt++)
    {
        bool last = (kt + 1 == k_tiles);

        // ensure every lane finished reading last iteration's shared data
        // before this iteration overwrites it
        barrier();

        for (int k = 0; k < 16; k++)
            _SH_A[int(lane) * (16 + _PAD) + k] = float16_t(tempA[k]);
        for (int e = 0; e < 8; e++)
        {
            int idx = int(lane) + e * 32;
            int kk = idx / 16;
            int nn = idx % 16;
            _SH_B[kk * (16 + _PAD) + nn] = float16_t(tempB[e]);
        }

        // ensure the publish above is visible before coopMatLoad reads it
        barrier();

        // prefetch the *next* chunk now, so its global loads can run
        // while the coopMatMulAdd below is executing
        if (!last)
        {
            int k0n = (kt + 1) * 16;
            for (int k = 0; k < 16; k++)
                tempA[k] = deref(deref(node).x_buf)[(r0 + int(lane)) * K + (k0n + k)];
            for (int e = 0; e < 8; e++)
            {
                int idx = int(lane) + e * 32;
                int kk = idx / 16;
                int nn = idx % 16;
                tempB[e] = deref(deref(node).weight)[(c0 + nn) * K + (k0n + kk)];
            }
        }

        coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> matA;
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
        coopMatLoad(matA, _SH_A, 0, 16 + _PAD, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matB, _SH_B, 0, 16 + _PAD, gl_CooperativeMatrixLayoutRowMajor);
        acc = coopMatMulAdd(matA, matB, acc);
    }

    coopMatStore(acc, _SH_Y, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (int n = 0; n < 16; n++)
        deref(resolve(deref(node).result))[(r0 + int(lane)) * OUTF + (c0 + n)] =
            _SH_Y[int(lane) * 16 + n] + deref(deref(node).bias)[c0 + n];
#endif
}
