void forward (RDV_NODE node, float _input[IN], out float _output[OUT])
{
    float _m1_output[M1_OUT];
    forward (deref(node).m1, _input, _m1_output);
    for (int i=0; i<M1_OUT; i++)
        _output[i] = _m1_output[i];
    float _m2_output[M2_OUT];
    forward (deref(node).m2, _input, _m2_output);
    for (int i=0; i<M2_OUT; i++)
        _output[M1_OUT + i] = _m2_output[i];
}