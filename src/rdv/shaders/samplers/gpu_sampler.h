void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    // convert _input -1..1 (fastest-component-first: x, y, z, ... - same
    // convention as GridSampler/TensorSampler) into 0..1 texture
    // coordinates, honoring `aligned` per axis. Image axis k already
    // samples shape[IN-1-k] (see __init__'s width/height/depth reversal),
    // which is exactly the desired pairing for _input[k] - so no further
    // reversal is needed here, unlike GridSampler/TensorSampler which
    // address shape[i] directly and so must reverse _input themselves.
    float ac = float(deref(node).aligned);
#if IN == 1
    float size = float(deref(node).shape[0]);
    float t = _input[0] * 0.5 + 0.5;
    float c = (t * (size - ac) + 0.5 * ac) / size;
#elif IN == 2
    float size0 = float(deref(node).shape[0]);
    float size1 = float(deref(node).shape[1]);
    float t0 = _input[0] * 0.5 + 0.5;
    float t1 = _input[1] * 0.5 + 0.5;
    // image axis k already samples shape[IN-1-k] (see __init__'s own
    // width/height/depth reversal) - which is exactly the desired pairing
    // for _input[k] (fastest-component-first vs shape's
    // fastest-dimension-last), so axis k reads _input[k] directly here.
    vec2 c = vec2(
        (t0 * (size1 - ac) + 0.5 * ac) / size1,
        (t1 * (size0 - ac) + 0.5 * ac) / size0);
#else
    float size0 = float(deref(node).shape[0]);
    float size1 = float(deref(node).shape[1]);
    float size2 = float(deref(node).shape[2]);
    float t0 = _input[0] * 0.5 + 0.5;
    float t1 = _input[1] * 0.5 + 0.5;
    float t2 = _input[2] * 0.5 + 0.5;
    vec3 c = vec3(
        (t0 * (size2 - ac) + 0.5 * ac) / size2,
        (t1 * (size1 - ac) + 0.5 * ac) / size1,
        (t2 * (size0 - ac) + 0.5 * ac) / size0);
#endif
    vec4 sampled_value = sample_image(
        deref(node).image_index,
        deref(node).sampler_index,
        c);
    for (int k = 0; k < OUT; k ++)
        _output[k] = sampled_value[k];
}
