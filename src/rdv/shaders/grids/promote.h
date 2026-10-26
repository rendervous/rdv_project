void forward(RDV_NODE node, int _input[IN], out float _output[OUT]) {
    float value[1];
    forward(deref(node).g, _input, value);
    for (int k = 0; k < OUT; k++) _output[k] = value[0];
}
