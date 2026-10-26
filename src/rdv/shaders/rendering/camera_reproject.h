void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    for (int i = 0; i < OUT; i++)
        _output[i] = 0.0;

    // --- reconstruct this pixel's own (new-camera) world-space ray, same
    // math as Cameras.h's own forward, but against fixed origin/target/up/
    // fov/aspect_ratio fields instead of a poses tensor row.
    vec3 o = deref(node).origin;
    vec3 t = deref(node).target;
    vec3 n = deref(node).up;
    vec2 p = (vec2(_input[2], _input[1]) + 0.5) / vec2(deref(node).shape[2], deref(node).shape[1]) * 2.0 - 1.0;
    float sx = p.x * deref(node).aspect_ratio;
    float sy = p.y;
    float sz = 1.0 / tan(deref(node).fov * 0.5);

    vec3 zaxis = normalize(t - o);
    vec3 xaxis = normalize(cross(n, zaxis));
    vec3 yaxis = cross(zaxis, xaxis);
    vec3 w = normalize(xaxis * sx + yaxis * sy + zaxis * sz);

    // --- reconstruct the world-space hit point from the new camera's own
    // depth map (parametric distance along w, e.g. a raymarched "depth 0").
    float depth_value[1];
    forward(deref(node).depth, _input, depth_value);
    float depth = depth_value[0];
    if (depth <= 0.00001)
        return; // no hit at this pixel - nothing to reproject, leave OUT at 0 (invalid)

    vec3 X = o + depth * w;

    // --- project X into the PREVIOUS camera's own screen space (inverse of
    // the same ray-generation formula above, solved for p given X instead
    // of the other way around: dx/dz == sx/sz, dy/dz == sy/sz since a point
    // along the ray keeps the same ratios as its own direction vector).
    vec3 po = deref(node).previous_origin;
    vec3 pt = deref(node).previous_target;
    vec3 pn = deref(node).previous_up;
    vec3 pzaxis = normalize(pt - po);
    vec3 pxaxis = normalize(cross(pn, pzaxis));
    vec3 pyaxis = cross(pzaxis, pxaxis);

    vec3 d = X - po;
    float dz = dot(d, pzaxis);
    if (dz <= 0.00001)
        return; // X is behind (or right on top of) the previous camera

    float psz = 1.0 / tan(deref(node).previous_fov * 0.5);
    float ppx = (psz * dot(d, pxaxis) / dz) / deref(node).previous_aspect_ratio;
    float ppy = psz * dot(d, pyaxis) / dz;
    if (ppx < -1.0 || ppx > 1.0 || ppy < -1.0 || ppy > 1.0)
        return; // outside the previous camera's own frustum - e.g. newly disoccluded

    // --- bilinear/nearest-sample the previous grid (and its own recorded
    // depth) at that reprojected coordinate (see GridSampler: _input is
    // fastest-component-first, i.e. (x, y, cam) here, not (cam, y, x) - a
    // single camera moving over time, not a batch, so the cam axis is
    // always sampled dead-center).
    float coords[3];
    coords[0] = ppx;
    coords[1] = ppy;
    coords[2] = 0.0;
    float sampled[OUT - 2];
    forward(deref(node).prev_sampler, coords, sampled);
    for (int i = 0; i < OUT - 2; i++)
        _output[i] = sampled[i];
    _output[OUT - 2] = 1.0; // valid

    // previous_depth is whatever the previous camera actually recorded at
    // this reprojected pixel - comparing it against X's own RADIAL distance
    // from the previous camera (not `dz`, which is the *planar* z-axis-only
    // component used for the perspective divide above - VolumeProfile's own
    // depth convention, like this one, is the radial/Euclidean distance
    // along each pixel's own ray, so this is the apples-to-apples
    // comparison) is a purely geometric occlusion check, independent of the
    // frustum/behind-camera test above.
    float radial_depth = length(d);
    float previous_depth_value[1];
    forward(deref(node).prev_depth_sampler, coords, previous_depth_value);
    _output[OUT - 1] = abs(radial_depth - previous_depth_value[0]); // depth discrepancy
}
