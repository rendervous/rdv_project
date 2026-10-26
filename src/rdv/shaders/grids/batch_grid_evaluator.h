void rdv_main(RDV_NODE node, uvec3 idx) {
    uint64_t input_ptr = deref(node).input_ptr + idx.x * IN * 4;
    uint64_t output_ptr = deref(node).output_ptr + idx.x * OUT * 4;
    forward(deref(node).g, deref(cast(IB, input_ptr)), deref(cast(OB, output_ptr)));
}
