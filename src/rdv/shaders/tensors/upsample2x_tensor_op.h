// Nearest-neighbor 2x upsample over a flattened [H*W, C] input - one 4x8
// tile per thread (4 flattened output rows x 8 channels), channel count
// unchanged. W here is the *output* width (2x the input's).
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
    int W_in = W_out / 2;

    for (int r = 0; r < 4; r++)
    {
        int orow = r0 + r;
        int oh = orow / W_out;
        int ow = orow % W_out;
        int ih = oh / 2;
        int iw = ow / 2;
        int in_row = ih * W_in + iw;
        for (int c = 0; c < 8; c++)
        {
            int cc = c0 + c;
            deref(resolve(deref(node).result))[orow * C + cc] = deref(resolve(deref(node).x_buf))[in_row * C + cc];
        }
    }
}
