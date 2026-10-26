/*
Physically based volume rendering with transmittance/alpha output and next-
event-estimation against a single fixed-direction light (e.g. a sun) - see
PBVRT (python). Ported from rendervous.project/rdv/include/volume_rendering/
pbvrt.h: the reference's environment *importance sampler* NEE (direction +
value/pdf + pdf) is replaced by a plain directional light (light_direction/
light_intensity fields) - a delta light needs no pdf/importance sampling, so
its NEE contribution is just throughput * light_intensity * phase * shadow
transmittance (transmittance_rt - see ratiotracking_transmittance_trait.h).
_output has SPECTRAL_DIM+1 entries: [0..SPECTRAL_DIM) is radiance,
[SPECTRAL_DIM] is 1 if the ray never scattered (still fully "background",
for external alpha compositing) or 0 once it did.

extinction/majorant/scattering_albedo/environment are the trait helpers
(see extinction_trait.h/majorant_trait.h/scattering_albedo_trait.h/
environment_trait.h) - majorant(node, x, w, out distance) also returns how
far (object-space, along w) that bound stays valid, so a spatially-varying
majorant (e.g. VoxelMajorantField wrapping a real MajorantGrid) lets this
loop skip empty space by re-querying at each cell boundary; a single-cell
VoxelMajorantField (one global bound) recovers the classic algorithm.
*/

void forward(RDV_NODE node, float _input[6], out float _output[OUT])
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    mat4x3 M = deref(node).transform; // object to world
    mat3 L = inverse(mat3(M));
    vec3 O = M[3];
    x = L * (x - O); // world position -> object space
    vec3 wo = L * w; // world direction -> object space (unnormalized, fine for traversal)

    for (int i = 0; i < OUT; i++)
        _output[i] = 0.0;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin >= tMax) // ray never touches the volume
        return;
    tMin = max(0, tMin) + 1e-5;
    x += wo * tMin; // entry point, object space
    float d = tMax - tMin;

    float g = deref(node).g;
    float density_scale = deref(node).density_scale;
    int max_bounces = deref(node).max_bounces;
    vec3 light_dir = normalize(deref(node).light_direction);

    float W[SPECTRAL_DIM]; // path throughput
    for (int i = 0; i < SPECTRAL_DIM; i++)
        W[i] = 1.0;

    int bounces = 0;
    // Total-iteration safety cap - NOT redundant with max_bounces (which
    // only limits real *scattering* events): a null collision (the common
    // case - see the `continue` below) never touches bounces at all, and a
    // coarse per-cell majorant grid (VoxelMajorantField/SphereMajorantField)
    // can have individual cells whose reported bound is far looser, relative
    // to the *local* density inside them, than the domain-wide majorant is
    // relative to the domain-wide average (one dense outlier voxel inflates
    // its whole cell's own bound) - a ray crossing such a cell can need an
    // enormous number of null-collision retries before making any real
    // progress, since the free-flight step is also capped by that same
    // cell's own (comparatively small) exit distance. A camera close to a
    // dense medium is exactly this worst case pushed further: extinction
    // (scaled by density_scale) close to the local majorant means
    // random() >= extinction/maj rarely trips the `continue` early, so the
    // *rate* of retries goes up too, not just how many are eventually
    // needed. Measured directly (see _bench_pbvrt_loop.py while
    // investigating this): `while (true)` here reliably drives the driver's
    // TDR watchdog (ErrorDeviceLost) for a per-cell majorant under that
    // combination, even with max_bounces cut down to 4 (ruling out bounce
    // count as the driver) - and a TDR reset while a host thread is mid-wait
    // on this dispatch's own fence is exactly what an unhandled access
    // violation (0xC0000005) crashing the whole process looks like, not
    // just a slow/degraded frame. Never matters for a single global-cell
    // majorant (whose own "cell" already spans the whole box, so the free-
    // flight step is never artificially capped small to begin with) - a
    // single-cell TensorGrid majorant is unaffected either way;
    // VoxelMajorantField/SphereMajorantField need this cap regardless of
    // how tight their own bound is, since delta tracking's worst case is
    // inherent to a *local* (not per-sample-exact) bound, not a sign the
    // acceleration structure itself is wrong. Raising the cap trades a
    // little more (already tiny, since it's only ever hit by the extreme
    // tail of rays) truncation bias for a little more worst-case accuracy.
    for (int safety = 0; safety < 16384; safety++)
    {
        float maj_distance;
        float maj = majorant(node, x, wo, maj_distance) + 1e-5;
        if (maj_distance <= 0.00001)
            maj_distance = 0.00001;
        float dt = min(min(d, maj_distance), -log(1.0 - random()) / (density_scale * maj)); // free-flight distance within this cell
        x += dt * wo;
        d -= dt;
        if (dt >= maj_distance) // free-flight in majorant distance
            continue;
        if (d <= 1e-7)
            break;
        if (random() >= extinction(node, x) / maj)
            continue; // null collision, keep marching
        bounces ++;
        if (bounces > max_bounces)
            return;

        // -- real collision --
        float sa[SPECTRAL_DIM];
        scattering_albedo(node, x, sa);
        for (int i = 0; i < SPECTRAL_DIM; i++)
            W[i] *= sa[i];

        bool some_throughput = false;
        for (int i = 0; i < SPECTRAL_DIM; i++)
            if (W[i] > 0.0001)
                some_throughput = true;
        if (!some_throughput)
            return; // path throughput too low, terminate
        // -- next event estimation against the fixed-direction light --
        float nee_phase = hg_phase_eval(w, light_dir, g);
        vec3 nee_wo = L * light_dir;
        float nee_tMin, nee_d;
        ray_box_intersection(x, nee_wo, nee_tMin, nee_d);
        float tr = transmittance_rt(node, x, nee_wo, nee_d, density_scale) * nee_phase;
        for (int i = 0; i < SPECTRAL_DIM; i++)
            _output[i] += W[i] * tr;

        // scatter ray
        w = hg_phase_sample(w, g); // world-space scattering direction
        wo = L * w;
        ray_box_intersection(x, wo, tMin, d); // remaining distance to the boundary from x
    }
}
