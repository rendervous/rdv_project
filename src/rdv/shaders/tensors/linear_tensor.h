// y = W @ x + b - wraps a 2D TensorBase g (shape [BATCH, in_features]),
// producing shape [BATCH, out_features] - one 4x8 (BLOCK_ROWS x
// BLOCK_CHANNELS) block per thread (see TensorBase), not one scalar,
// computed as a tiled 4x8x8 GEMM: for each 8-wide K-chunk of in_features,
// read this thread's own 4x8 input tile from g (one call - g's own
// forward() already returns a whole 4x8 block, one 8-wide K-chunk of
// in_features at a time, matching this layer's own channel-block width)
// and this thread's own 8x8 weight tile (out_features x in_features,
// PyTorch nn.Linear layout - this is the "cache the matrix tile" part:
// each of the 8 output channels' 8 weight values for this K-chunk is
// read once and reused across all 4 rows of the accumulation below,
// instead of re-reading it once per row), and accumulate their product
// into this thread's own 4x8 running total.
//
// A group-shared version of this (staging the input tile - identical
// across the 32-thread workgroup whenever they all share the same
// row_block, since only their own channel_block differs - into shared
// memory once via CooperativeLibrary.all_thread_sync, instead of every
// thread separately re-walking g's own forward() for it) was tried and
// measured *slower* in every configuration tested (small out_features,
// where a workgroup rarely covers one row_block at all since
// out_features/8 < 32; and out_features=256, exactly matching workgroup
// width, where it should help most) - the extra barrier()s cost more
// than the redundant recomputation they save, at least for the MLP
// depths/widths measured. Kept to plain independent-per-thread reads.
void forward(RDV_NODE node, int _input[IN], out float _output[32])
{
    int row_block = _input[0];
    int ch_block = _input[1];
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

        int in_idx[2];
        in_idx[0] = row_block;
        in_idx[1] = kt;
        float in_tile[32];
        forward(deref(node).g, in_idx, in_tile);

        for (int r = 0; r < 4; r++)
        {
            for (int k = 0; k < 8; k++)
            {
                float in_val = in_tile[r * 8 + k];
                for (int c = 0; c < 8; c++)
                    acc[r][c] += in_val * w_tile[c][k];
            }
        }
    }

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            _output[r * 8 + c] = acc[r][c];
}
