void scattering_albedo(RDV_NODE node, vec3 x, out float sa[SPECTRAL_DIM])
{
    float _input[3] = float[3](x.x, x.y, x.z);
    forward(deref(node).scattering_albedo, _input, sa);
}
