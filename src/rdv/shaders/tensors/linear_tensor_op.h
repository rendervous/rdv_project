// y = W @ x + b - one 4x8 (rows x out_features) tile per thread, tiled
// as a 4x8x8 GEMM over 8-wide chunks of in_features: each chunk's own
// 8x8 weight tile is read once and cached in registers, reused across
// the tile's own 4 rows, then written straight into this op's own
// result buffer (see TensorOperation) - input/weight/bias are all plain,
// already-materialized flat buffers (no recursive forward() calls into
// another node at all).
void forward(RDV_NODE node)
{
    int tile_id = int(gl_GlobalInvocationID.x);
    int col_blocks = OUTF / 8;
    int row_block = tile_id / col_blocks;
    int ch_block = tile_id % col_blocks;
    int r0 = row_block * 4;
    int c0 = ch_block * 8;
    int k_tiles = INF / 8;

    float acc[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            acc[r][c] = deref(resolve(deref(node).bias))[c0 + c];

    for (int kt = 0; kt < k_tiles; kt++)
    {
        int k0 = kt * 8;

        float w_tile[8][8];
        for (int c = 0; c < 8; c++)
            for (int k = 0; k < 8; k++)
                w_tile[c][k] = deref(resolve(deref(node).weight))[(c0 + c) * INF + (k0 + k)];

        float in_tile[4][8];
        for (int r = 0; r < 4; r++)
            for (int k = 0; k < 8; k++)
                in_tile[r][k] = deref(resolve(deref(node).x_buf))[(r0 + r) * INF + (k0 + k)];

        for (int r = 0; r < 4; r++)
        {
            for (int k = 0; k < 8; k++)
            {
                float in_val = in_tile[r][k];
                for (int c = 0; c < 8; c++)
                    acc[r][c] += in_val * w_tile[c][k];
            }
        }
    }

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            deref(resolve(deref(node).result))[(r0 + r) * OUTF + (c0 + c)] = acc[r][c];
}
