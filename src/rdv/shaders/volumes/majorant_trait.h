float majorant(RDV_NODE node, vec3 x, vec3 w, out float distance)  {
    float _output[2];
    forward(deref(node).majorant, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _output);
    distance = _output[1];
    return _output[0];
}