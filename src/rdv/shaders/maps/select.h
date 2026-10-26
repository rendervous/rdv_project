#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
    float values[V];
    forward(deref(node).m, values);
    for (int k = 0; k < OUT; k++) _output[k] = values[deref(node).sel[k]];
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    float values[V];
    forward(deref(node).m, _input, values);
    for (int k = 0; k < OUT; k++) _output[k] = values[deref(node).sel[k]];
}
#endif
