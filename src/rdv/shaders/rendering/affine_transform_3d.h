void forward(RDV_NODE node, float _input[3], out float _output[3]) {
    vec4 p = vec4(_input[0], _input[1], _input[2], 1.0);
    vec3 o = deref(node).transform * p;
    _output = float[3] (o.x, o.y, o.z);
}