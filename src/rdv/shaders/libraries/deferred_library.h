layout(set=0, binding=BINDING_SLOT(0), scalar) buffer rdv_deferred_pointers_ssbo {
    uint64_t[] pointers;
} rdv_deferred_pointers;

#ifdef RDV_WEBGPU
// WebGPU export (see rdv/core/_webgpu.py): 64-bit integers are narrowed to
// 32 bits, so a name index can't live in the high bits. Every deferred code
// is re-encoded at export time as (entry << 1) | 1 instead, entry indexing
// this table, which the exported evaluate() fills with each entry's already
// resolved address (bound buffer + offset, or 0 if unbound).
uint64_t rdv_resolve_ptr(uint64_t ptr) {
    if ((ptr & 1) == 0) return ptr;
    return rdv_deferred_pointers.pointers[int(ptr >> 1)];
}
#else
uint64_t rdv_resolve_ptr(uint64_t ptr) {
    if ((ptr & 1) == 0) return ptr;
    int index = int(ptr >> 48);
    uint64_t deferred_ptr = rdv_deferred_pointers.pointers[index];
    if (deferred_ptr == 0) return 0; // null bindings doesnt offset
    // + binds tighter than >> in GLSL (as in C) - without the inner parens
    // this adds the *whole* masked value to deferred_ptr and only then
    // shifts that sum right by 1, instead of adding just the (already
    // marker-bit-shifted) offset - silently corrupting the resolved
    // pointer for any offset != 0.
    return deferred_ptr + ((ptr & 0x0000FFFFFFFFFFFFul) >> 1);
}
#endif

void RDV_NODE(uvec3 idx) { }