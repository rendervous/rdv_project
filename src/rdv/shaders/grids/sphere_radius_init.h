void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    float value[1];
    forward(deref(node).g, _input, value);
    _output[0] = value[0]; // majorant
    _output[1] = deref(node).radius;
}
