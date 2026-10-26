void forward(RDV_NODE node, float _input[2], out float _output[3]) {
    vec3 dir = oct2dir(vec2(_input[0], _input[1]));
    _output = float[3] (dir.x, dir.y, dir.z);
}
