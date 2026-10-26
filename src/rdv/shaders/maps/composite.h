#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
    float values[V];
    forward(deref(node).g, values);
    forward(deref(node).f, values, _output);
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    float values[V];
    forward(deref(node).g, _input, values);
    forward(deref(node).f, values, _output);
}
#endif
