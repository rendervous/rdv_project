struct RDV_NODE {
    int shape[IN];
    D deferred_buffer;
};

T get_value(in RDV_NODE tensor, int indices[IN]) {
    return deref(resolve(tensor.deferred_buffer))[indices_to_global(tensor.shape, indices)];
}

void set_value(in RDV_NODE tensor, int indices[IN], T value) {
    deref(resolve(tensor.deferred_buffer))[indices_to_global(tensor.shape, indices)] = value;
}

// Reads this leaf's own 4x8 block directly from its buffer - _input is a
// *block* index (see TensorBase), not an element index. Only valid (and
// only compiled) for a 2D (rows, channels) Tensor - the general-purpose,
// arbitrary-rank role this same leaf also serves (e.g. TensorSampler's
// own t field, or test_tensor_reinterpret.py, both only ever using
// get_value()/set_value() above, never forward()) never calls this, and
// its own fixed-size node.shape[1] wouldn't even compile for IN != 2.
#if IN == 2
void forward(RDV_NODE node, int _input[IN], out float _output[32])
{
    int r0 = _input[0] * 4;
    int c0 = _input[1] * 8;
    int cols = node.shape[1];
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 8; c++)
        {
            int idx[2];
            idx[0] = r0 + r;
            idx[1] = c0 + c;
            _output[r * 8 + c] = float(get_value(node, idx));
        }
    }
}
#endif
