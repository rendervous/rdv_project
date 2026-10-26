/*

y = x * A + b   (A stored PyTorch nn.Linear-style, [out_features, in_features] -
                  read via coopMatLoad's own ColumnMajor addressing below, so
                  it's used as if x * A^T without ever physically transposing it)

x: BxIN
A: OUTxIN (precast to float16 once, at construction - see LinearTensorOpSimple._cast_fp16)
y: BxOUT
b: OUT

each work tile (one workgroup) is a 32-row batch of x - the whole batch's
own IN columns are staged (converted to float16) into shared memory once,
since every one of the OUT_TILES row-blocks below reads the exact same 32
rows of x, just against a different 16-wide slice of A.

_SUBGROUPS_PER_TILE (32/64/128-thread workgroups, 1/2/4 subgroups) lets
several 16-out-feature blocks be processed *concurrently*: each subgroup
owns one block at a time (r = r_base + subgroup id), stages that block's
own row-tile (16 x IN, all of IN) into its own private slice of _SH_A, and
stores its own result independently - only the outer loop over r_base (OUT_TILES
stepped by _SUBGROUPS_PER_TILE) is still sequential. _SH_A's own total size
is therefore _SUBGROUPS_PER_TILE*16*IN (16xIN/32xIN/64xIN for 1/2/4).

Each subgroup's own accumulator is seeded with its bias slice (broadcast
across all 32 rows via a stride-0 coopMatLoad - legal per
GL_KHR_cooperative_matrix, stride is just "elements between rows") instead
of zero, and coopMatStore writes straight into
deref(resolve(deref(node).result)) (address()+cast to a fixed-size view,
same trick as LinearTensorOpFast's own weight_tiled reads) - no
intermediate shared output buffer/copy-out loop.
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _SUBGROUPS_PER_TILE SPT  // SPT is a real generic (LinearTensorOpSimple._max_subgroups), not a fixed literal - auto-#define'd by codegen

#define _SH_X RDV_CONCAT(RDV_NODE, _sh_x)
#define _SH_A RDV_CONCAT(RDV_NODE, _sh_a)
shared float16_t _SH_X[32 * INF];
shared float16_t _SH_A[_SUBGROUPS_PER_TILE * 16 * INF];
#endif

#define IN_TILES (INF / 16)
#define OUT_TILES (OUTF / 16)
#define _WG_THREADS (32 * _SUBGROUPS_PER_TILE)

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint local = gl_LocalInvocationID.x;
    uint lane = local % 32;
    uint sub = local / 32;
    int wgTile = int(gl_WorkGroupID.x);
    int start_subgroup_row = wgTile * 32;

    // Stage this workgroup's own 32 rows (all of IN), converted to
    // float16, cooperatively across every thread of the workgroup (not
    // just one subgroup's own 32 lanes) - every r-block below reads the
    // same rows.
    TENSOR_OP_UNROLL
    for (int i = int(local); i < 32 * INF; i += _WG_THREADS)
        _SH_X[i] = float16_t(deref(deref(node).x_buf)[(start_subgroup_row + i / INF) * INF + (i % INF)]);
    barrier();

    uint64_t bias_base = address(deref(node).bias);
    uint64_t result_base = address(deref(node).result);

    TENSOR_OP_UNROLL
    for (int r_base = 0; r_base < OUT_TILES; r_base += _SUBGROUPS_PER_TILE)
    {
        int r = r_base + int(sub);

        // cache this subgroup's own A row-tile (16 out-features x all of
        // IN) into its own private slice of shared memory, once per r
        int a_off = int(sub) * 16 * INF;
        for (int i = int(lane); i < 16 * INF; i += 32)
            _SH_A[a_off + i] = deref(deref(node).weight)[r * 16 * INF + i]; // already float16 - plain copy
        barrier();

        // seed the accumulator with bias, broadcast across all 32 rows
        // (stride 0 - every row reads the same 16 elements)
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> mat_acc;
        coopMatLoad(mat_acc, deref(cast(F16, bias_base + uint64_t(r * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);

        TENSOR_OP_UNROLL
        for (int c = 0; c < IN_TILES; c++) // channels are split into 16-sized chunks
        {
            coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> mat_a;
            coopMatLoad(mat_a, _SH_X, c * 16, INF, gl_CooperativeMatrixLayoutRowMajor);

            // A's own row-tile is laid out [16 out-features][IN] (its
            // native storage order, not transposed) - ColumnMajor
            // addressing here reads it as the [K, N] matB needs without
            // physically transposing anything (see this file's own
            // header comment).
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
            coopMatLoad(mat_b, _SH_A, a_off + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);

            mat_acc = coopMatMulAdd(mat_a, mat_b, mat_acc);
        }

        // store straight into result - element offset picks this tile's
        // own (row, out-feature) origin, stride OUTF is the real row
        // pitch (F512 is just a fixed-size view wide enough for
        // coopMatStore's own type; the actual addressing/bounds are
        // element+row*stride+col, same as every other coopMatLoad/Store
        // here reading directly off a buffer).
        uint64_t out_off = uint64_t(start_subgroup_row * OUTF + r * 16) * 4;
        coopMatStore(mat_acc, deref(cast(F512, result_base + out_off)), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);

        barrier(); // before the next r_base iteration overwrites _SH_A
    }
#endif
}
