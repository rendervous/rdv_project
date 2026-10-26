/*
y = x * A + b, for A ([out_features, in_features], PyTorch nn.Linear
layout) small enough (both dims <= 128) to be staged into shared memory
*entirely*, once, as float16 - unlike linear_tensor_op_simple.h, which
only ever keeps one 16-row slice of A resident at a time (re-staged per
out-feature block), every out-feature block here reads directly off the
same, already-resident copy - no re-staging, and no barrier at all
between out-feature blocks (nothing shared is overwritten once staged).

x: BxIN, A: OUTxIN, y: BxOUT, b: OUT

_SUBGROUPS_PER_TILE subgroups share one workgroup, each processing its
own, *different* 32-row slice of x concurrently (not, as an earlier
revision of this file did, different out-feature blocks of the *same*
32 rows) - since A is already fully resident and shared by the whole
workgroup regardless, the natural way to spend extra subgroups here is
more rows per dispatch, exactly one thread per output row (dispatch
thread count is simply output.shape[0] - see
LinearTensorOpSmall._dispatch_thread_count), not extra out-feature
parallelism gated by out_features being a multiple of
16*_SUBGROUPS_PER_TILE (that constraint doesn't even apply here: only
batch needs to be a multiple of 32*_SUBGROUPS_PER_TILE). A is staged
once, cooperatively, by every thread of the workgroup; x is staged once
too, each subgroup into its own private _SUBGROUPS_PER_TILE*32*IN-wide
slice of _SH_X. Every out-feature block's own accumulator is seeded with
bias (broadcast across all 32 rows via a stride-0 coopMatLoad) and,
once accumulated over all IN/16 K-chunks, stored directly into
deref(deref(node).result) (address()+cast to a fixed-size view,
same trick as LinearTensorOpFast's own weight_tiled reads) - one store,
no intermediate shared output buffer/copy-out loop, and no barrier at
all once staging is done (neither _SH_X nor _SH_W is ever written again).
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _SUBGROUPS_PER_TILE SPT  // SPT is a real generic (LinearTensorOpSmall._max_subgroups), not a fixed literal - auto-#define'd by codegen
#define _WG_THREADS (32 * _SUBGROUPS_PER_TILE)
#define _ROWS_PER_WG (32 * _SUBGROUPS_PER_TILE)

#define _SH_X RDV_CONCAT(RDV_NODE, _sh_x)
#define _SH_W RDV_CONCAT(RDV_NODE, _sh_w)
shared float16_t _SH_X[_ROWS_PER_WG * INF];
shared float16_t _SH_W[OUTF * INF];
#endif

#define IN_TILES (INF / 16)
#define OUT_TILES (OUTF / 16)

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    // stage this workgroup's own _ROWS_PER_WG rows of x, converted to
    // float16, cooperatively across every thread of the workgroup - read
    // directly off the buffer once; each subgroup will only ever read
    // back its own 32-row slice of this below.
//    TENSOR_OP_UNROLL
//    for (int i = int(local); i < _ROWS_PER_WG * INF; i += _WG_THREADS)
//        _SH_X[i] = float16_t(deref(deref(node).x_buf)[(start_row + i / INF) * INF + (i % INF)]);
    FBUFFER b = cast(FBUFFER, address(deref(node).x_buf) + (int(gl_WorkGroupID.x) * _ROWS_PER_WG * INF) * 4);

    // 3. El bucle avanza de forma coalescente: todos los hilos leen posiciones contiguas simultáneamente
    TENSOR_OP_UNROLL
    for (int linear_idx = int(gl_LocalInvocationID.x); linear_idx < _ROWS_PER_WG * INF; linear_idx += _WG_THREADS) {
        _SH_X[linear_idx] = float16_t(deref(b)[linear_idx]);
    }
    barrier();

    // stage the *entire* weight matrix once - small enough (<=128x128)
    // to fit alongside x - shared by every subgroup below, no re-staging.
    TENSOR_OP_UNROLL
    for (int i = int(gl_LocalInvocationID.x); i < OUTF * INF; i += _WG_THREADS)
        _SH_W[i] = float16_t(deref(deref(node).weight)[i]);
    barrier();

    uint64_t bias_base = address(deref(node).bias);
    uint64_t result_base = address(deref(node).result);

    int my_row0 = int(gl_WorkGroupID.x) * _ROWS_PER_WG + int(gl_LocalInvocationID.x / 32) * 32;
    int x_off = int(gl_LocalInvocationID.x / 32) * 32 * INF;

    // this subgroup's own 32 rows sweep every out-feature block, against
    // the one, shared (already fully resident, read-only from here on)
    // _SH_W - no barrier needed anywhere in this loop.
    coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> mat_acc;
    coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> mat_a;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;

#if INF < OUTF
    // OUT_TILES outer, IN_TILES (K-reduction) inner, one coopmat
    // accumulator per r, reset from bias and fully reduced before its
    // one and only store - the straightforward nesting, empirically
    // fastest when IN < OUT (see LinearTensorOpSmall's own docstring).
    TENSOR_OP_UNROLL
    for (int r = 0; r < OUT_TILES; r++)
    {
        // seed the accumulator with bias, broadcast across all 32 rows
        // (stride 0 - every row reads the same 16 elements)
        coopMatLoad(mat_acc, deref(cast(F16, bias_base + uint64_t(r * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);

        TENSOR_OP_UNROLL
        for (int c = 0; c < IN_TILES; c++)
        {
            coopMatLoad(mat_a, _SH_X, x_off + c * 16, INF, gl_CooperativeMatrixLayoutRowMajor);

            // A's own row-tile is laid out [OUT][IN] (its natural,
            // already-resident staging order, not transposed) -
            // ColumnMajor addressing here reads it as the [K, N] matB
            // needs without physically transposing anything (see
            // linear_tensor_op_simple.h's own header comment).
            coopMatLoad(mat_b, _SH_W, r * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
            mat_acc = coopMatMulAdd(mat_a, mat_b, mat_acc);
        }

        // one store, straight into result - no intermediate shared
        // output buffer, no barrier needed before the next r (neither
        // _SH_X nor _SH_W is ever overwritten once staged).
        uint64_t out_off = uint64_t(my_row0 * OUTF + r * 16) * 4;
        coopMatStore(mat_acc, deref(cast(FOUT, result_base + out_off)), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
    }
#else
    // Swapped: IN_TILES outer, OUT_TILES inner - mat_a (x's own K-chunk)
    // is now loaded once per c and reused across every r, instead of
    // reloaded per (r, c) pair. One accumulator per r is kept resident
    // in registers across the whole outer c loop instead of round-
    // tripping through result every (c, r) pair - arrays of coopmat are
    // rejected by this driver (see linear_tensor_op_fast.h), so hand-
    // unrolled up to OUT_TILES<=8 (OUTF<=128's own max) - each stored
    // only once, after the loop.
    #define _ACC_T coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator>
    #define _OUT_OFF(i) (uint64_t(my_row0 * OUTF + (i) * 16) * 4)
    _ACC_T acc0;
#if OUT_TILES > 1
    _ACC_T acc1;
#endif
#if OUT_TILES > 2
    _ACC_T acc2;
#endif
#if OUT_TILES > 3
    _ACC_T acc3;
#endif
#if OUT_TILES > 4
    _ACC_T acc4;
#endif
#if OUT_TILES > 5
    _ACC_T acc5;
#endif
#if OUT_TILES > 6
    _ACC_T acc6;
#endif
#if OUT_TILES > 7
    _ACC_T acc7;
#endif

    // seed every accumulator with its own bias slice, broadcast across
    // all 32 rows (stride 0 - every row reads the same 16 elements)
    coopMatLoad(acc0, deref(cast(F16, bias_base + uint64_t(0 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#if OUT_TILES > 1
    coopMatLoad(acc1, deref(cast(F16, bias_base + uint64_t(1 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 2
    coopMatLoad(acc2, deref(cast(F16, bias_base + uint64_t(2 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 3
    coopMatLoad(acc3, deref(cast(F16, bias_base + uint64_t(3 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 4
    coopMatLoad(acc4, deref(cast(F16, bias_base + uint64_t(4 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 5
    coopMatLoad(acc5, deref(cast(F16, bias_base + uint64_t(5 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 6
    coopMatLoad(acc6, deref(cast(F16, bias_base + uint64_t(6 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 7
    coopMatLoad(acc7, deref(cast(F16, bias_base + uint64_t(7 * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);
#endif

    TENSOR_OP_UNROLL
    for (int c = 0; c < IN_TILES; c++)
    {
        coopMatLoad(mat_a, _SH_X, x_off + c * 16, INF, gl_CooperativeMatrixLayoutRowMajor);

        // A's own row-tile is laid out [OUT][IN] (its natural, already-
        // resident staging order, not transposed) - ColumnMajor
        // addressing here reads it as the [K, N] matB needs without
        // physically transposing anything (see linear_tensor_op_simple.h's
        // own header comment).
        coopMatLoad(mat_b, _SH_W, 0 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc0 = coopMatMulAdd(mat_a, mat_b, acc0);
#if OUT_TILES > 1
        coopMatLoad(mat_b, _SH_W, 1 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc1 = coopMatMulAdd(mat_a, mat_b, acc1);
#endif
#if OUT_TILES > 2
        coopMatLoad(mat_b, _SH_W, 2 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc2 = coopMatMulAdd(mat_a, mat_b, acc2);
#endif
#if OUT_TILES > 3
        coopMatLoad(mat_b, _SH_W, 3 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc3 = coopMatMulAdd(mat_a, mat_b, acc3);
#endif
#if OUT_TILES > 4
        coopMatLoad(mat_b, _SH_W, 4 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc4 = coopMatMulAdd(mat_a, mat_b, acc4);
#endif
#if OUT_TILES > 5
        coopMatLoad(mat_b, _SH_W, 5 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc5 = coopMatMulAdd(mat_a, mat_b, acc5);
#endif
#if OUT_TILES > 6
        coopMatLoad(mat_b, _SH_W, 6 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc6 = coopMatMulAdd(mat_a, mat_b, acc6);
#endif
#if OUT_TILES > 7
        coopMatLoad(mat_b, _SH_W, 7 * 16 * INF + c * 16, INF, gl_CooperativeMatrixLayoutColumnMajor);
        acc7 = coopMatMulAdd(mat_a, mat_b, acc7);
#endif
    }

    // store every accumulator exactly once, after the whole K-reduction
    coopMatStore(acc0, deref(cast(FOUT, result_base + _OUT_OFF(0))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#if OUT_TILES > 1
    coopMatStore(acc1, deref(cast(FOUT, result_base + _OUT_OFF(1))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 2
    coopMatStore(acc2, deref(cast(FOUT, result_base + _OUT_OFF(2))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 3
    coopMatStore(acc3, deref(cast(FOUT, result_base + _OUT_OFF(3))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 4
    coopMatStore(acc4, deref(cast(FOUT, result_base + _OUT_OFF(4))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 5
    coopMatStore(acc5, deref(cast(FOUT, result_base + _OUT_OFF(5))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 6
    coopMatStore(acc6, deref(cast(FOUT, result_base + _OUT_OFF(6))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#if OUT_TILES > 7
    coopMatStore(acc7, deref(cast(FOUT, result_base + _OUT_OFF(7))), 0, OUTF, gl_CooperativeMatrixLayoutRowMajor);
#endif
#endif
#endif
}
