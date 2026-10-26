/*
'same'-padded (zero border), stride-1 2D convolution, fixed to a 3x3
kernel/padding=1 - same "simple" idea as linear_tensor_op_simple.h
(weight row-tile staged once per out-channel block, reused across every
K-chunk; the input tile staged once per workgroup, reused across every
out-channel block), with the reduction dimension flattened to 9*CIN
(9 fixed taps x CIN, back to back) instead of just IN. KH/KW are literal
constants here (not generics), so the tap loop is a plain -1..1 x -1..1
unroll instead of Conv2DTensorOpSimple's own general KHxKW loop, and
there's no _CHUNK_K-style chunking (see Conv2DTensorOpSimple's own
docstring for why that exists) - shared memory scales directly with CIN
and SPT, checked up front in Conv2DTensorOpSimple3x3.__init__ (matches
LinearTensorOpSmall's own safety-assert approach).

x: (H*W)xCIN (channels-last, row-major)
weight: 3x3xCINxCOUT (channels-last - see Conv2DTensorOp's own docstring)
y: (H*W)xCOUT
b: COUT

SPT subgroups share one workgroup, each processing a *different*
out-channel block concurrently (r = r_base + subgroup id), against the
one, shared (already staged, read-only from here on) 32-row _SH_X - see
LinearTensorOpSimple's own docstring for the identical idea applied to a
plain matmul; SPT is computed the same way (_max_subgroups), just with
K = 9*CIN instead of IN.

H/W are generics (bound once, at construction), not runtime struct
fields - see Conv2DTensorOp's own docstring for why.
*/
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define _SUBGROUPS_PER_TILE SPT  // SPT is a real generic (Conv2DTensorOpSimple3x3._max_subgroups), not a fixed literal - auto-#define'd by codegen
#define _WG_THREADS (32 * _SUBGROUPS_PER_TILE)
#define _TAPK (9 * CIN)

#define _SH_X RDV_CONCAT(RDV_NODE, _sh_x)
#define _SH_A RDV_CONCAT(RDV_NODE, _sh_a)
shared float16_t _SH_X[32 * _TAPK];
shared float16_t _SH_A[_SUBGROUPS_PER_TILE * 16 * _TAPK];
#endif

#define IN_TILES (_TAPK / 16)
#define OUT_TILES (COUT / 16)

void forward(RDV_NODE node)
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint local = gl_LocalInvocationID.x;
    uint lane = local % 32;
    uint sub = local / 32;
    int wgTile = int(gl_WorkGroupID.x);
    int start_row = wgTile * 32;

    // stage this workgroup's own 32 rows - 9 fixed (3x3, pad=1) taps x
    // CIN, zero-filled where a tap falls outside the image - every
    // out-block below (across every subgroup) reads the same staged
    // taps, so this loop runs once, cooperatively, across all
    // _WG_THREADS threads (not just one subgroup's own 32 lanes).
    for (int i = int(local); i < 32 * _TAPK; i += _WG_THREADS)
    {
        int row = i / _TAPK;
        int k = i % _TAPK;
        int tap = k / CIN;
        int c = k % CIN;
        int flat_idx = start_row + row;
        int h = flat_idx / W;
        int w = flat_idx % W;
        int ih = h + (tap / 3) - 1;
        int iw = w + (tap % 3) - 1;
        bool valid = (ih >= 0 && ih < H && iw >= 0 && iw < W);
        _SH_X[i] = valid
            ? float16_t(deref(deref(node).x_buf)[(ih * W + iw) * CIN + c])
            : float16_t(0.0);
    }
    barrier();

    uint64_t bias_base = address(deref(node).bias);
    uint64_t result_base = address(resolve(deref(node).result));

    for (int r_base = 0; r_base < OUT_TILES; r_base += _SUBGROUPS_PER_TILE)
    {
        int r = r_base + int(sub);

        // cache this subgroup's own out-channel block's row-tile (16
        // out-channels x all 9*CIN) into its own private slice of
        // shared memory, once per r
        int a_off = int(sub) * 16 * _TAPK;
        for (int i = int(lane); i < 16 * _TAPK; i += 32)
        {
            int n = i / _TAPK;
            int k = i % _TAPK;
            _SH_A[a_off + i] = float16_t(deref(deref(node).weight)[k * COUT + (r * 16 + n)]);
        }
        barrier();

        // seed the accumulator with bias, broadcast across all 32 rows
        // (stride 0 - every row reads the same 16 elements)
        coopmat<float, gl_ScopeSubgroup, 32, 16, gl_MatrixUseAccumulator> mat_acc;
        coopMatLoad(mat_acc, deref(cast(F16, bias_base + uint64_t(r * 16) * 4)), 0, 0, gl_CooperativeMatrixLayoutRowMajor);

        for (int c = 0; c < IN_TILES; c++)
        {
            coopmat<float16_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseA> mat_a;
            coopMatLoad(mat_a, _SH_X, c * 16, _TAPK, gl_CooperativeMatrixLayoutRowMajor);

            // this out-block's own row-tile is laid out [16 out-channels][9*CIN]
            // (its natural staging order above, not transposed) -
            // ColumnMajor addressing here reads it as the [K, N] matB
            // needs without physically transposing anything (see
            // linear_tensor_op_simple.h's own header comment).
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
            coopMatLoad(mat_b, _SH_A, a_off + c * 16, _TAPK, gl_CooperativeMatrixLayoutColumnMajor);

            mat_acc = coopMatMulAdd(mat_a, mat_b, mat_acc);
        }

        // one store, straight into result - no intermediate shared
        // output buffer/copy-out loop.
        uint64_t out_off = uint64_t(start_row * COUT + r * 16) * 4;
        coopMatStore(mat_acc, deref(cast(FOUT, result_base + out_off)), 0, COUT, gl_CooperativeMatrixLayoutRowMajor);

        barrier(); // before the next r_base iteration overwrites _SH_A
    }
#endif
}
