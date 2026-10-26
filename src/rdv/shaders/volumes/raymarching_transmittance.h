void forward (RDV_NODE node, float _input[6], out float _output[1]) {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    mat4x3 M = deref(node).transform;
    mat3 L = inverse(mat3(M[0].xyz, M[1].xyz, M[2].xyz));
    vec3 O = M[3].xyz;
    x = L * (x - O); // convert world position to object space position (use same x)
    w = L * w; // convert world direction to object space direction (unnormalized, fine for traversal)

    float tMin, tMax;
    ray_box_intersection(x, w, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray never touches the volume - fully transparent
    {
        _output[0] = 1.0;
        return;
    }
    tMin = max(0, tMin);
    x += w * tMin; // entry point, object space
    float d = tMax - tMin;

    _output[0] = transmittance_rm(node, x, w, d, deref(node).step_size, deref(node).density_scale);
}