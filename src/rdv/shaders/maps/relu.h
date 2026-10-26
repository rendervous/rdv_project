void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    for (int i = 0; i < OUT; i++)
        _output[i] = max(_input[i], 0.0);
}
