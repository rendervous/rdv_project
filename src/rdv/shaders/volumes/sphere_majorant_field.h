/*
node:
- grid (each voxel holds (majorant, radius) - see SphereRadiusField).

SphereRadiusField's own guarantee is really that `majorant` bounds a BOX of
half-width `radius` (voxel-index units) centered on the voxel - a sphere of
that radius is just a (looser, easier to reason about) subset of that box.
Querying the box directly here, scaled per-axis by this grid's own voxel
size (not collapsed to an isotropic sphere via e.g. the smallest per-axis
voxel size), matters a lot once voxels aren't cubic (a common case - see
MAJORANT_GRID_SHAPE in sandbox_pbvrt_majorant_grid_compare.py): an
isotropic-sphere radius conservative enough for the *smallest* axis is far
too small along the others, collapsing the step to near zero there and
defeating the whole point (worse: forcing thousands of near-zero-progress
steps, easily enough to trip the driver's TDR watchdog). Same robust
min/max-based exit-distance math as voxel_majorant_field.h (t0/t1 computed
independently, not one from a shift of the other, so a zero component of w
gives a well-defined +-infinity instead of a NaN-poisoned result).
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
    float value[2];
    forward(deref(node).grid, int[3](cell.z, cell.y, cell.x), value);
    float majorant = value[0];
    float radius_voxels = value[1]; // half-width, in voxel-index units, of the box this majorant bounds

    vec3 voxel_size = vec3(2.0) / dim;
    vec3 center = (vec3(cell) + 0.5) * voxel_size - vec3(1.0);
    vec3 half_extent = radius_voxels * voxel_size; // per-axis object-space half-width

    vec3 bmin = center - half_extent;
    vec3 bmax = center + half_extent;
    vec3 t0 = (bmin - x) / w;
    vec3 t1 = (bmax - x) / w;
    vec3 texit = max(t0, t1);
    float dist = min(min(texit.x, texit.y), texit.z);

    _output[0] = majorant;
    _output[1] = max(dist, 1e-6);
}
