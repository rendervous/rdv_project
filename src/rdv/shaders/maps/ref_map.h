#if IN == 0
void forward(RDV_NODE node, out float _output[OUT])
{
    forward(deref(node).map, _output);
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    forward(deref(node).map, _input, _output);
}
#endif