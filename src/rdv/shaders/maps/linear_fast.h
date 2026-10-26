// y = W @ x + b via GL_KHR_cooperative_matrix (Vulkan's analogue of
// CUDA's WMMA/tensor cores) - only usable, and only actually faster,
// when the current device supports cooperative matrices
// (RDV_COOPERATIVE_MATRIX_SUPPORT, defined by rdv's own megakernel
// header - see _core.py's _build_code) *and* the whole 32-row dispatch
// workgroup is evaluating this exact LinearFast instance (checked via
// CooperativeLibrary's own all_thread_sync - see
// shaders/libraries/cooperative_library.h). Otherwise every thread falls
// back to a plain, fully independent fp32 read-and-multiply - no
// shared-memory tiling tier of its own (see Linear for that); LinearFast
// is cooperative-matrix-or-bust by design.
//
// This class of tensor core only supports float16 A/B operands (a
// float32 matA/matB compiled and ran with no error, but its behavior for
// an unsupported configuration is undefined per spec, and was reliably
// wrong when measured) - weight/input are staged into shared memory as
// float16_t, trading precision (~1e-2 max abs error on randn-scale
// values) for hardware matrix-unit throughput; the accumulator stays
// float32.
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// IN/OUT rounded up to the next multiple of 16 - lets every tile of the
// GEMM below (including a final, partial one whenever IN or OUT isn't
// itself a multiple of 16) be a full, uniform 16x16x16 coopmat tile, its
// out-of-range rows/columns simply zero-filled once at staging time
// instead of ever being a special case in the tile loop itself.
#define _CM_PIN (((IN + 15) / 16) * 16)
#define _CM_POUT (((OUT + 15) / 16) * 16)
#define _SH_X RDV_CONCAT(RDV_NODE, _sh_x)
#define _SH_Y RDV_CONCAT(RDV_NODE, _sh_y)
#define _SH_WTILE RDV_CONCAT(RDV_NODE, _sh_wtile)
shared float16_t _SH_X[32 * _CM_PIN];
shared float _SH_Y[32 * _CM_POUT];
shared float16_t _SH_WTILE[16 * 16];
#endif

void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
    uint lane = gl_LocalInvocationID.x;

    if (all_thread_sync(uint(deref(node).id)))
    {
        // Stage this workgroup's 32 input rows into shared memory once
        // (zero-padded out to _CM_PIN columns), then run a tiled GEMM
        // (Y[32,OUT] = X[32,IN] @ W^T[IN,OUT]) as two 16-row halves of
        // the 32-row workgroup tile (m0 = 0, 16), each a sequence of
        // fixed 16x16x16 coopmat tiles against the hardware's own matrix
        // units, staging the 32 (real-width) result rows back out
        // through shared memory. W itself ([OUT, IN] row-major, PyTorch
        // nn.Linear layout, un-padded) is never coopMatLoad'd from
        // directly - a raw 16-wide load near a K/N boundary would spill
        // into the next row's own data - so each (k0, n0) weight tile is
        // staged into a small, padded shared scratch buffer first
        // (out-of-range k/n zero-filled the same way as _SH_X), and only
        // that scratch is ever coopMatLoad'd from; W^T falls out for
        // free since staging reads weight[(n0+nn)*IN + (k0+kk)] into
        // _SH_WTILE[kk][nn] (row=k, col=n), no transposed load needed.
        for (int k = 0; k < _CM_PIN; k++)
            _SH_X[lane * _CM_PIN + k] = float16_t((k < IN) ? _input[k] : 0.0);
        barrier();

        for (int m0 = 0; m0 < 32; m0 += 16)
        {
            for (int n0 = 0; n0 < _CM_POUT; n0 += 16)
            {
                int n_len = min(16, OUT - n0);
                coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
                    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
                for (int k0 = 0; k0 < _CM_PIN; k0 += 16)
                {
                    int k_len = min(16, IN - k0);

                    barrier();
                    for (int idx = int(lane); idx < 16 * 16; idx += 32)
                    {
                        int kk = idx / 16;
                        int nn = idx % 16;
                        _SH_WTILE[idx] = float16_t((kk < k_len && nn < n_len)
                            ? deref(resolve(deref(node).weight))[(n0 + nn) * IN + (k0 + kk)] : 0.0);
                    }
                    barrier();

                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
                    coopMatLoad(matA, _SH_X, m0 * _CM_PIN + k0, _CM_PIN, gl_CooperativeMatrixLayoutRowMajor);
                    coopMatLoad(matB, _SH_WTILE, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
                    acc = coopMatMulAdd(matA, matB, acc);
                }
                coopMatStore(acc, _SH_Y, m0 * _CM_POUT + n0, _CM_POUT, gl_CooperativeMatrixLayoutRowMajor);
            }
        }
        barrier();

        for (int o = 0; o < OUT; o++)
            _output[o] = _SH_Y[lane * _CM_POUT + o] + deref(resolve(deref(node).bias))[o];
        return;
    }
#endif
    for (int o = 0; o < OUT; o++)
    {
        float acc = deref(resolve(deref(node).bias))[o];
        for (int k = 0; k < IN; k++)
            acc += _input[k] * deref(resolve(deref(node).weight))[o * IN + k];
        _output[o] = acc;
    }
}
#ifdef RDV_COOPERATIVE_MATRIX_SUPPORT
#undef _CM_PIN
#undef _CM_POUT
#undef _SH_X
#undef _SH_Y
#undef _SH_WTILE
#endif
