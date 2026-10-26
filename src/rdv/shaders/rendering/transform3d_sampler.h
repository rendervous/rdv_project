void forward(RDV_NODE node, float _input[3], out float _output[OUT]) {
    vec3 p = vec3(_input[0], _input[1], _input[2]);
    vec3 tp = deref(node).transform * vec4(p, 1.0);
    float transformed[3] = float[3](tp.x, tp.y, tp.z);
    forward(deref(node).inner, transformed, _output);
}
