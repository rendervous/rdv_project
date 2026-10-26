// Channel concat of two [H*W, C1]/[H*W, C2] inputs (same H*W, both C1
// and C2 multiples of 8, so no 4x8 tile ever straddles the boundary) -
// one 4x8 tile per thread.
void forward(RDV_NODE node)
{
    int tile_id = int(gl_GlobalInvocationID.x);
    int out_cols = deref(node).shape[1];
    int col_blocks = out_cols / 8;
    int row_block = tile_id / col_blocks;
    int ch_block = tile_id % col_blocks;
    int r0 = row_block * 4;
    int c0 = ch_block * 8;
    int C1 = deref(node).C1;
    int C2 = out_cols - C1;

    for (int r = 0; r < 4; r++)
    {
        int row = r0 + r;
        for (int c = 0; c < 8; c++)
        {
            int cc = c0 + c;
            float v = cc < C1
                ? deref(resolve(deref(node).x1_buf))[row * C1 + cc]
                : deref(resolve(deref(node).x2_buf))[row * C2 + (cc - C1)];
            deref(resolve(deref(node).result))[row * out_cols + cc] = v;
        }
    }
}
