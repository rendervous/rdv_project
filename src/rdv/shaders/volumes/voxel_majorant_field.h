/*
node:
- majorant_grid (each voxel represents a majorant).
*/

void forward (RDV_NODE node, float _input[6], out float _output[2])
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    ivec3 dim = ivec3(
        deref(deref(node).grid).shape[2],
        deref(deref(node).grid).shape[1],
        deref(deref(node).grid).shape[0]
    );
    ivec3 cell = clamp(ivec3((x*0.5 + 0.5) * dim), ivec3(0), dim - 1);
    float value[1];
    forward(deref(node).grid, int[3](cell.z, cell.y, cell.x), value);
    _output[0] = value[0]; // majorant value

    vec3 voxel_size = vec3(2.0) / dim;
    vec3 bmin = cell * voxel_size - vec3(1.0);
    vec3 bmax = bmin + voxel_size;
    // Exit distance along w, per axis - t0/t1 computed independently (not
    // one from a shift of the other) so a zero (or near-zero) component of
    // w gives a well-defined +-infinity on that axis instead of an
    // inf + (-inf) == NaN that would poison the min() below and stall the
    // caller's free-flight loop (same robust pattern as
    // ray_box_intersection in math_library.h).
    vec3 t0 = (bmin - x) / w;
    vec3 t1 = (bmax - x) / w;
    vec3 texit = max(t0, t1);
    _output[1] = min(min(texit.x, texit.y), texit.z);
}