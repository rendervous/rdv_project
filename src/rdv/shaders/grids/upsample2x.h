// Nearest-neighbor 2x upsample - shape is g's own shape doubled, OUT
// unchanged (per-channel).
void forward(RDV_NODE node, int _input[2], out float _output[OUT])
{
    int h = _input[0] / 2;
    int w = _input[1] / 2;
    forward(deref(node).g, int[2](h, w), _output);
}
