/*
Physically based (unbiased, spectral) volume rendering via delta tracking
(null-collision, spatial-majorant free-flight sampling) - see PBVR (python).
Ported from rendervous.project/rdv/include/volume_rendering/pbvr.h, without
next-event-estimation (no environment importance sampler here - just a
plain environment(direction) lookup for escaped rays): noisier per sample,
but still unbiased, meant to be averaged over many dispatches.

`mg` (Generics.MG) is a MajorantField (python) providing the free-flight
majorant - see shaders/volumes/majorant_field_precode.h for the
local_majorant contract it implements. VoxelMajorantField, wrapping a
single-cell (shape [1,1,1]) grid holding one global value, recovers the
plain global-majorant algorithm exactly; the same field type wrapping a
real, higher-resolution MajorantGrid (python) instead lets the tracker
skip empty space, tightening the bound near/inside the medium - any other
MajorantField implementation (e.g. a sphere-based bound) drops in here
unchanged.
*/

void forward (RDV_NODE node, float _input[6], out float _output[3])
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    mat4x3 M = deref(node).transform; // object to world
    mat3 L = inverse(mat3(M));
    vec3 O = M[3];
    x = L * (x - O); // world position -> object space
    vec3 wo = L * w; // world direction -> object space (not normalized, fine for traversal)

    _output[0] = 0.0;
    _output[1] = 0.0;
    _output[2] = 0.0;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray never touches the volume
    {
        float env[3];
        forward(deref(node).env, float[](w.x, w.y, w.z), env);
        _output[0] = env[0];
        _output[1] = env[1];
        _output[2] = env[2];
        return;
    }
    tMin = max(0, tMin);
    x += wo * tMin; // entry point, object space
    float d = tMax - tMin;

    float g = deref(node).g;
    vec3 albedo = deref(node).albedo;
    int max_bounces = deref(node).max_bounces;

    vec3 W = vec3(1.0); // path throughput
    int bounces = 0;
    bool escaped = false;
    // The macrocell-crossing safety cap below (independent of max_bounces,
    // which only limits real scattering events) guards against a stalled
    // DDA step near a macrocell boundary (a step landing within float
    // precision of its own starting point) hanging the whole dispatch -
    // see local_majorant's own epsilon.
    for (int safety = 0; d > 0.0 && safety < 8192; safety++)
    {
        float max_step;
        // Floored: an empty-space cell has majorant == 0, which already
        // yields dt = +inf (still correctly bigger than max_step) for any
        // typical random() draw - the floor only prevents the 0/0 == NaN
        // case on the vanishingly rare draw of random() == 0 exactly.
        float majorant = max(local_majorant(deref(node).mg, x, wo, max_step), 1e-12);
        float dt = -log(1.0 - random()) / majorant; // free-flight distance within this cell
        float step = min(dt, min(max_step, d));
        x += step * wo;
        d -= step;
        if (step < dt)
        {
            if (d <= 0.0)
            {
                escaped = true; // reached the volume boundary without a collision
                break;
            }
            continue; // crossed into the next majorant cell, re-query its bound
        }

        float sigma_t[1];
        forward(deref(node).s, float[](x.x, x.y, x.z), sigma_t);
        if (random() >= sigma_t[0] / majorant)
            continue; // null collision, keep marching

        // -- real collision --
        W *= albedo;
        if (max(max(W.x, W.y), W.z) < 0.0001)
            return; // path throughput too low, terminate (no environment term)

        if (bounces >= max_bounces)
            return; // bounce budget exhausted, terminate (no environment term)

        w = hg_phase_sample(w, g); // scattering direction, world space
        wo = L * w;
        ray_box_intersection(x, wo, tMin, d); // remaining distance to the boundary from x
        bounces++;
    }

    if (escaped)
    {
        float env[3];
        forward(deref(node).env, float[](w.x, w.y, w.z), env);
        _output[0] = W.x * env[0];
        _output[1] = W.y * env[1];
        _output[2] = W.z * env[2];
    }
}
