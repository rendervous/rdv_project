// SingleGPUGrid: the exact texel at an integer grid index (Grid convention:
// _input[0] is the slowest axis), fetched from the single image bound for this
// node's own ID (SingleImageLibrary[ID]) - no filtering, so values come back
// bit-exact (what a majorant grid needs).
void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
#if IN == 1
    vec4 value = RDV_CONCAT(rdv_fetch_image_, ID)(_input[0]);
#elif IN == 2
    vec4 value = RDV_CONCAT(rdv_fetch_image_, ID)(ivec2(_input[1], _input[0]));
#else
    vec4 value = RDV_CONCAT(rdv_fetch_image_, ID)(ivec3(_input[2], _input[1], _input[0]));
#endif
    for (int k = 0; k < OUT; k++)
        _output[k] = value[k];
}
