void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    forward(deref(node).g1, _input, _output);
    float g2_out[OUT];
    forward(deref(node).g2, _input, g2_out);
    for (int i=0; i<OUT; i++)
        _output[i] = BINARY_OP(_output[i], g2_out[i]);
}
