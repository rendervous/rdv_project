// One thread per 4x8 block (see TensorBase's own docstring) - idx.x is a
// *block* index, unflattened against the block-shape (element shape / 4,
// 8), and the resulting 32 values are scattered out to their real (row,
// channel) positions in the element-shaped output buffer.
void rdv_main(RDV_NODE node, uvec3 idx) {
    int block_shape[IN];
    block_shape[0] = deref(node).shape[0] / 4;
    block_shape[1] = deref(node).shape[1] / 8;

    int indices[IN];
    global_to_indices(block_shape, int(idx.x), indices);

    float values[32];
    forward(deref(node).g, indices, values);

    int cols = deref(node).shape[1];
    int r0 = indices[0] * 4;
    int c0 = indices[1] * 8;
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 8; c++)
        {
            uint64_t output_ptr = deref(node).output_ptr + uint64_t((r0 + r) * cols + (c0 + c)) * 4;
            deref(cast(F1, output_ptr))[0] = values[r * 8 + c];
        }
    }
}
