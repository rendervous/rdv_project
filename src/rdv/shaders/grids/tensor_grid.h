/*
Code for the TensorGrid class.
*/

void forward(RDV_NODE node, int _input[IN], out float _ouput[OUT])
{
    int global_idx = indices_to_global(deref(node).shape, _input);
    global_idx *= OUT * 4;
    uint64_t ptr = resolve(deref(node).ptr) + global_idx;
    _ouput = deref(cast(B, ptr));
}