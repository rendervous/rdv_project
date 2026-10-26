// SingleGPUSampler: GPUSampler's coordinate convention (_input in [-1, 1],
// fastest-component-first, `aligned` per axis - see gpu_sampler.h), sampling
// the single image bound for this node's own ID (SingleImageLibrary[ID])
// instead of an index into ImagingLibrary's arrays.
void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    float ac = float(deref(node).aligned);
#if IN == 1
    float size = float(deref(node).shape[0]);
    float c = ((_input[0] * 0.5 + 0.5) * (size - ac) + 0.5 * ac) / size;
#elif IN == 2
    vec2 size = vec2(deref(node).shape[1], deref(node).shape[0]);
    vec2 t = vec2(_input[0], _input[1]) * 0.5 + 0.5;
    vec2 c = (t * (size - ac) + 0.5 * ac) / size;
#else
    vec3 size = vec3(deref(node).shape[2], deref(node).shape[1], deref(node).shape[0]);
    vec3 t = vec3(_input[0], _input[1], _input[2]) * 0.5 + 0.5;
    vec3 c = (t * (size - ac) + 0.5 * ac) / size;
#endif
    vec4 sampled_value = RDV_CONCAT(rdv_sample_image_, ID)(c);
    for (int k = 0; k < OUT; k++)
        _output[k] = sampled_value[k];
}
