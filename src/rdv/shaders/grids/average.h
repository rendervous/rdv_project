void forward(RDV_NODE node, int _input[IN], out float _output[OUT]) {
    int n = deref(node).samples;
    for (int k = 0; k < OUT; k++) _output[k] = 0.0;
    for (int s = 0; s < n; s++) {
        float sample_out[OUT];
        forward(deref(node).g, _input, sample_out);
        for (int k = 0; k < OUT; k++) _output[k] += sample_out[k];
    }
    for (int k = 0; k < OUT; k++) _output[k] /= float(n);
}
