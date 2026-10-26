void forward(RDV_NODE node, float _input[3], out float _output[2]) {
    vec2 oct = dir2oct(vec3(_input[0], _input[1], _input[2]));
    _output = float[2] (oct.x, oct.y);
}
