// 'same'-padded (zero border), stride-1 2D convolution: output[h,w,:] over
// a materialized grid g (IN=2, OUT=CIN) - each of the KH*KW taps is read
// directly from g's own forward(), so g must already be buffer-backed
// (e.g. via `>> barrier`) for that to be a cheap read instead of a
// redundant re-evaluation of however g's own value was computed.
void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    int h = _input[0];
    int w = _input[1];
    int H = deref(deref(node).g).shape[0];
    int W = deref(deref(node).g).shape[1];
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;

    uint64_t w_base = resolve(deref(node).weight);

    for (int o = 0; o < OUT; o++)
        _output[o] = deref(cast(F1, resolve(deref(node).bias) + uint64_t(o) * 4))[0];

    for (int kh = 0; kh < KH; kh++)
    {
        int ih = h + kh - pad_h;
        if (ih < 0 || ih >= H)
            continue;
        for (int kw = 0; kw < KW; kw++)
        {
            int iw = w + kw - pad_w;
            if (iw < 0 || iw >= W)
                continue;

            float in_vec[CIN];
            forward(deref(node).g, int[2](ih, iw), in_vec);

            for (int o = 0; o < OUT; o++)
            {
                for (int c = 0; c < CIN; c++)
                {
                    uint64_t idx = uint64_t(((o * CIN + c) * KH + kh) * KW + kw);
                    _output[o] += deref(cast(F1, w_base + idx * 4))[0] * in_vec[c];
                }
            }
        }
    }
}
