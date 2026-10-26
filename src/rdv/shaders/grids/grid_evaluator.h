void rdv_main(RDV_NODE node, uvec3 idx) {
    int indices[IN];
    global_to_indices(deref(deref(node).g).shape, int(idx.x), indices);
    uint64_t output_ptr = deref(node).output_ptr + idx.x * OUT * 4;
    forward(deref(node).g, indices, deref(cast(OB, output_ptr)));
}
