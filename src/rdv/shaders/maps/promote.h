#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
    float value[1];
    forward(deref(node).f, value);
    for (int k = 0; k < OUT; k++) _output[k] = value[0];
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    float value[1];
    forward(deref(node).f, _input, value);
    for (int k = 0; k < OUT; k++) _output[k] = value[0];
}
#endif
