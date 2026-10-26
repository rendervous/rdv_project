void forward(RDV_NODE node, int _input[IN], out float _output[OUT]) {
    float values[V];
    forward(deref(node).grid, _input, values);
    forward(deref(node).map, values, _output);
}
