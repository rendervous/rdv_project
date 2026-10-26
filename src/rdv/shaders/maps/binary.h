void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    forward(deref(node).m1, _input, _output);
    float m2_out[OUT];
    forward(deref(node).m2, _input, m2_out);
    for (int i=0; i<OUT; i++)
        _output[i] = BINARY_OP(_output[i], m2_out[i]);
}