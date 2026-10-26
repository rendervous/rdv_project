void forward(RDV_NODE node, float _input[3], out float _output[2]) {
    vec2 xr = dir2xr(vec3(_input[0], _input[1], _input[2]));
    _output = float[2] (xr.x, xr.y);
}
