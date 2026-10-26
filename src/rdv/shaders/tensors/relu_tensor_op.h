// f(x) = max(x, 0) - one 4x8 tile per thread, writing straight into this
// op's own result buffer (see TensorOperation).
void forward(RDV_NODE node)
{
    int tile_id = int(gl_GlobalInvocationID.x);
    int cols = deref(node).shape[1];
    int col_blocks = cols / 8;
    int row_block = tile_id / col_blocks;
    int ch_block = tile_id % col_blocks;
    int r0 = row_block * 4;
    int c0 = ch_block * 8;

    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 8; c++)
        {
            int flat_idx = (r0 + r) * cols + (c0 + c);
            float v = deref(resolve(deref(node).x_buf))[flat_idx];
            deref(resolve(deref(node).result))[flat_idx] = max(0.0, v);
        }
    }
}
