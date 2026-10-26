// Unlike tensor.h's struct (deliberately inlined wherever referenced,
// never behind a pointer - see Tensor's own docstring), TensorBarrier is
// pointer-referenced like any other Component - Component's own
// _generate_code already auto-emits the struct/pointer wrapper for
// shape/deferred_buffer (from cls.__fields__), so this file only needs
// the one extra deref() (unwrapping node's own pointer first) that a
// pointer-referenced field always needs. _input is a *block* index (see
// TensorBase), not an element index.
void forward(RDV_NODE node, int _input[IN], out float _output[32])
{
    int r0 = _input[0] * 4;
    int c0 = _input[1] * 8;
    int cols = deref(node).shape[1];
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 8; c++)
        {
            int flat_idx = (r0 + r) * cols + (c0 + c);
            _output[r * 8 + c] = deref(resolve(deref(node).deferred_buffer))[flat_idx];
        }
    }
}
