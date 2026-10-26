void forward(RDV_NODE node, int _input[IN], out float _output[OUT]) {
    float values[V];
    forward(deref(node).g, _input, values);
    for (int k = 0; k < OUT; k++) _output[k] = values[deref(node).sel[k]];
}
