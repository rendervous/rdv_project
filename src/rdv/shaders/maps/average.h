#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
    int n = deref(node).samples;
    float accum[OUT]; // local accumulator, NOT the out param - _output is a
    // reference to the caller's own storage, so accumulating into it
    // directly means every `+=` across all n iterations is a read-modify-
    // write through that reference instead of a plain register update;
    // accumulate locally and write to _output exactly once at the end.
    for (int k = 0; k < OUT; k++) accum[k] = 0.0;
    float sample_out[OUT]; // hoisted out of the loop below - n is a runtime value (deref(node).samples), not a compile-time constant, so the loop can never be unrolled anyway; re-declaring this array each iteration only adds needless per-iteration stack/register churn
    [[dont_unroll]]
    for (int s = 0; s < n; s++) {
        forward(deref(node).m, sample_out);
        for (int k = 0; k < OUT; k++) accum[k] += sample_out[k];
    }
    for (int k = 0; k < OUT; k++) _output[k] = accum[k] / float(n);
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    for (int k = 0; k < OUT; k++) _output[k] = 0.0;
    float sample_out[OUT]; // hoisted out of the loop below - see the IN==0 branch's own note
    for (int s = 0; s < deref(node).samples; s++) {
        forward(deref(node).m, _input, sample_out);
        for (int k = 0; k < OUT; k++) _output[k] += sample_out[k] / float(deref(node).samples);
    }
}
#endif
