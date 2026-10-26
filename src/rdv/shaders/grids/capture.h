void rdv_main(RDV_NODE node, uvec3 idx) {
    int indices[IN];
    global_to_indices(deref(deref(node).g).shape, int(idx.x), indices);
    float acc[OUT];
    for (int k = 0; k < OUT; k++) acc[k] = 0;
    int n = deref(node).samples;
    for (int s = 0; s < n; s++) {
        float values[V];
        forward(deref(node).g, indices, values);
        float result[OUT];
        forward(deref(node).m, values, result);
        for (int k = 0; k < OUT; k++) acc[k] += result[k];
    }
    float inv = 1.0 / float(n);
    for (int k = 0; k < OUT; k++) acc[k] *= inv;
    uint64_t output_ptr = deref(node).output_ptr + idx.x * OUT * 4;
    deref(cast(OB, output_ptr)) = acc;
}
