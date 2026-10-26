float extinction(RDV_NODE node, vec3 x) {
    float value[1];
    forward(deref(node).extinction, float[3](x.x, x.y, x.z), value);
    return value[0];
}
