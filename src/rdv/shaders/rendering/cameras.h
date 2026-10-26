void forward(RDV_NODE node, int _input[IN], out float _output[OUT]) {
    uint64_t poses_ptr = resolve(deref(node).poses) + _input[0] * 9 * 4;
    P poses_array = cast(P, poses_ptr);
    vec3 o = deref(poses_array)[0];
    vec3 t = deref(poses_array)[1];
    vec3 n = deref(poses_array)[2];
    vec2 subsample = deref(node).pixel_jitter == 0 ? vec2(0.5, 0.5) : vec2(random(), random());
    vec2 p = (vec2(_input[2], _input[1]) + subsample) / vec2(deref(node).shape[2], deref(node).shape[1]) * 2.0 - 1.0;
    float sx = p.x * deref(node).aspect_ratio;
    float sy = p.y;
    float sz = 1.0 / tan(deref(node).fov * 0.5);

    vec3 zaxis = normalize(t - o);
    vec3 xaxis = normalize(cross(n, zaxis));
    vec3 yaxis = cross(zaxis, xaxis);

    vec3 w = xaxis * sx + yaxis * sy + zaxis * sz;
    w = normalize(w);

    _output = float[6]( o.x, o.y, o.z, w.x, w.y, w.z );
}