void forward (RDV_NODE node, int _input[IN], out float _output[OUT])
{
    float _g1_output[G1_OUT];
    forward (deref(node).g1, _input, _g1_output);
    for (int i=0; i<G1_OUT; i++)
        _output[i] = _g1_output[i];
    float _g2_output[G2_OUT];
    forward (deref(node).g2, _input, _g2_output);
    for (int i=0; i<G2_OUT; i++)
        _output[G1_OUT + i] = _g2_output[i];
}