// y = W @ x + b - one input row per thread, full fp32 precision.
//
// Whenever the whole 32-row dispatch workgroup happens to be evaluating
// this exact Linear instance (checked via CooperativeLibrary's own
// all_thread_sync - see shaders/libraries/cooperative_library.h, which
// also covers the case where rdv's own dispatch bounds check left a
// partial tail workgroup with fewer than 32 threads actually here), each
// output row's weight values are cooperatively staged into shared memory
// once per workgroup instead of independently re-read by every thread.
// Otherwise every thread reads/multiplies fully independently - always
// correct, just without the shared-memory reuse.
#define _SH_ROW RDV_CONCAT(RDV_NODE, _sh_row)
shared float _SH_ROW[IN];

void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    uint lane = gl_LocalInvocationID.x;

    if (all_thread_sync(uint(deref(node).id)))
    {
        for (int o = 0; o < OUT; o++)
        {
            barrier();
            for (int k = int(lane); k < IN; k += 32)
                _SH_ROW[k] = deref(resolve(deref(node).weight))[o * IN + k];
            barrier();

            float acc = deref(resolve(deref(node).bias))[o];
            for (int k = 0; k < IN; k++)
                acc += _input[k] * _SH_ROW[k];
            _output[o] = acc;
        }
    }
    else
    {
        for (int o = 0; o < OUT; o++)
        {
            float acc = deref(resolve(deref(node).bias))[o];
            for (int k = 0; k < IN; k++)
                acc += _input[k] * deref(resolve(deref(node).weight))[o * IN + k];
            _output[o] = acc;
        }
    }
}
#undef _SH_ROW
