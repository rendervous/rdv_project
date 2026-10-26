// 'same'-padded (zero border), stride-1 2D convolution, one invocation
// per (h, w, o) output channel (OUT=1 - a single scalar), not one
// invocation per pixel computing the whole out_channels-length vector
// (see Conv2dGrid's own conv2d.h for that version) - over an already
// buffer-backed grid g (IN=2, OUT=CIN); each of the KH*KW taps is read
// directly from g's own forward(), so g must already be materialized
// (e.g. via `>> barrier`) for that to be a cheap read instead of a
// redundant re-evaluation of however g's own value was computed.
//
// weight is [KH, KW, CIN, COUT] (channels-last, out-channel fastest-
// varying) rather than Conv2dGrid's PyTorch [O, C, KH, KW] layout - this
// grid's own shape puts o in the fastest-varying dispatch axis (H, W, O,
// 1 - see global_to_indices' own fastest-axis-first convention), so
// neighboring lanes in a dispatch typically share (h, w) and only differ
// in o; a channels-last weight layout keeps each lane's own o column
// contiguous, ready for a later cooperative-matrix upgrade (the same
// K*K*CIN input patch read once per (h, w), multiplied against
// different columns of the [K*K*CIN, COUT] weight matrix by different
// lanes - not yet implemented here, this version still just loops
// per-lane/independently, same as Conv2dGrid).
void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    int h = _input[0];
    int w = _input[1];
    int o = _input[2];
    // _input[3] is always 0 - the trailing unit dispatch axis, one
    // invocation per o.

    int H = deref(deref(node).g).shape[0];
    int W = deref(deref(node).g).shape[1];
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;

    float acc = deref(resolve(deref(node).bias))[o];

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

            for (int c = 0; c < CIN; c++)
            {
                int idx = ((kh * KW + kw) * CIN + c) * COUT + o;
                acc += deref(resolve(deref(node).weight))[idx] * in_vec[c];
            }
        }
    }
    _output[0] = acc;
}
