// f(x) = max(x, 0) - wraps another TensorBase g, same shape, one 4x8
// block at a time (see TensorBase) - purely pointwise, no cross-value
// dependency within the block.
void forward(RDV_NODE node, int _input[IN], out float _output[32])
{
    float in_block[32];
    forward(deref(node).g, _input, in_block);
    for (int i = 0; i < 32; i++)
        _output[i] = max(0.0, in_block[i]);
}
