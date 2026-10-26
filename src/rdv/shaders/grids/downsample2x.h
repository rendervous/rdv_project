// 2x2 average-pool, stride 2 (H and W both assumed even - see
// Downsample2x's own Python __init__) - shape is g's own shape halved,
// OUT unchanged (per-channel).
void forward(RDV_NODE node, int _input[2], out float _output[OUT])
{
    int h = _input[0] * 2;
    int w = _input[1] * 2;

    float v00[OUT], v01[OUT], v10[OUT], v11[OUT];
    forward(deref(node).g, int[2](h, w), v00);
    forward(deref(node).g, int[2](h, w + 1), v01);
    forward(deref(node).g, int[2](h + 1, w), v10);
    forward(deref(node).g, int[2](h + 1, w + 1), v11);

    for (int c = 0; c < OUT; c++)
        _output[c] = 0.25 * (v00[c] + v01[c] + v10[c] + v11[c]);
}
