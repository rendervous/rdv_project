void environment(RDV_NODE node, vec3 w, out float result[SPECTRAL_DIM])
{
    float _input[3] = float[3](w.x, w.y, w.z);
    forward(deref(node).environment, _input, result);
}
