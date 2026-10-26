// 2x2 average-pool, stride 2, over a flattened [H*W, C] input (H, W both
// even) - one 4x8 tile per thread (4 flattened output rows x 8
// channels), channel count unchanged.
void forward(RDV_NODE node)
{
    int tile_id = int(gl_GlobalInvocationID.x);
    int C = deref(node).shape[1];
    int col_blocks = C / 8;
    int row_block = tile_id / col_blocks;
    int ch_block = tile_id % col_blocks;
    int r0 = row_block * 4;
    int c0 = ch_block * 8;
    int W_out = deref(node).W;
    int W_in = W_out * 2;

    for (int r = 0; r < 4; r++)
    {
        int orow = r0 + r;
        int oh = orow / W_out;
        int ow = orow % W_out;
        int ih0 = oh * 2;
        int iw0 = ow * 2;
        int base00 = (ih0 * W_in + iw0) * C;
        int base01 = (ih0 * W_in + iw0 + 1) * C;
        int base10 = ((ih0 + 1) * W_in + iw0) * C;
        int base11 = ((ih0 + 1) * W_in + iw0 + 1) * C;
        for (int c = 0; c < 8; c++)
        {
            int cc = c0 + c;
            float v = deref(resolve(deref(node).x_buf))[base00 + cc]
                    + deref(resolve(deref(node).x_buf))[base01 + cc]
                    + deref(resolve(deref(node).x_buf))[base10 + cc]
                    + deref(resolve(deref(node).x_buf))[base11 + cc];
            deref(resolve(deref(node).result))[orow * C + cc] = v * 0.25;
        }
    }
}
