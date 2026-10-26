// 'same'-padded (zero border), stride-1 2D convolution over a flattened
// [H*W, in_channels] input (row-major, channels-last, H*W flattened the
// same way LinearTensorOp's own BATCH dimension is) - one 4x8 tile per
// thread (4 flattened (h, w) rows x 8 out_channels), producing
// [H*W, out_channels]. H/W are generics (bound once, at construction,
// like every other shape parameter here - see Conv2DTensorOp's own
// docstring) rather than runtime struct fields, so the compiler sees
// col_blocks/h_of/w_of's own row/col decoding as divides/modulos by a
// compile-time constant (cheaply strength-reduced to a multiply) instead
// of a genuine runtime division, and every (kh, kw) tap's own
// out-of-bounds check is then done once per row too, letting each tap's
// own 8x8 weight tile (out x in channels, channels-last - see
// Conv2DTensorOp's own docstring for why, unlike Conv2dGrid's PyTorch
// layout) be read once per (kh, kw, K-chunk) and cached in registers,
// reused across all 4 (valid) rows - the same "cache the tile" principle
// LinearTensorOp's own K-chunk loop uses. x_buf/weight/bias are all
// plain, already-materialized flat buffers (no recursive forward() calls
// into another node at all).
#define col_blocks (COUT / 8)

void forward(RDV_NODE node)
{
    int tile_id = int(gl_GlobalInvocationID.x);
    int cin_tiles = CIN / 8;
    int row_block = tile_id / col_blocks;
    int ch_block = tile_id % col_blocks;
    int r0 = row_block * 4;
    int c0 = ch_block * 8;
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;

    int h_of[4];
    int w_of[4];
    for (int r = 0; r < 4; r++)
    {
        int flat_idx = r0 + r;
        h_of[r] = flat_idx / W;
        w_of[r] = flat_idx % W;
    }

    float acc[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            acc[r][c] = deref(resolve(deref(node).bias))[c0 + c];

    for (int kh = 0; kh < KH; kh++)
    {
        for (int kw = 0; kw < KW; kw++)
        {
            int in_row_of[4];
            bool valid_of[4];
            for (int r = 0; r < 4; r++)
            {
                int ih = h_of[r] + kh - pad_h;
                int iw = w_of[r] + kw - pad_w;
                valid_of[r] = (ih >= 0 && ih < H && iw >= 0 && iw < W);
                in_row_of[r] = valid_of[r] ? (ih * W + iw) : 0;
            }

            for (int kt = 0; kt < cin_tiles; kt++)
            {
                int k0 = kt * 8;

                float w_tile[8][8];
                for (int c = 0; c < 8; c++)
                    for (int k = 0; k < 8; k++)
                        w_tile[c][k] = deref(resolve(deref(node).weight))[((kh * KW + kw) * CIN + (k0 + k)) * COUT + (c0 + c)];

                for (int r = 0; r < 4; r++)
                {
                    if (!valid_of[r])
                        continue;
                    for (int k = 0; k < 8; k++)
                    {
                        float in_val = deref(resolve(deref(node).x_buf))[in_row_of[r] * CIN + (k0 + k)];
                        for (int c = 0; c < 8; c++)
                            acc[r][c] += in_val * w_tile[c][k];
                    }
                }
            }
        }
    }

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            deref(resolve(deref(node).result))[(r0 + r) * COUT + (c0 + c)] = acc[r][c];
}
