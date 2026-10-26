/*
Importance-samples a direction from a linearized quadtree of solid-angle-
weighted environment-map pixel densities (built once, at construction -
see XREnvironmentSampler/_build_xr_density_quadtree) instead of drawing a
uniform sphere direction - see this file's own Python-side class
docstring for why.

densities layout: node n's own 4 children's *cumulative* probabilities
(densities[n*4+3] ~= 1) live at densities[n*4 .. n*4+3]; child c's own
next node is n*4+1+c (1-indexed complete-quaternary-tree-as-array
addressing - see _build_xr_density_quadtree's own docstring for why the
+1 is required, not optional). Traversal narrows a [0,1]x[0,1] (u, v)
rectangle down to a single leaf pixel, LEVELS levels deep, accumulating
that leaf's own absolute selection probability (prob, the *product* of
every level's own conditional probability along the path) along the way.

A single random() draw drives the whole path (not one per level) - after
each level's own 4-ary search over its cumulative thresholds locates the
selected child's own [lo, hi) bucket, sel is rescaled back to a fresh
uniform [0,1) via (sel-lo)/(hi-lo) before being reused one level deeper;
without that rescale, the leftover sel range shrinks with every level
(it's never mapped back up to [0,1)), starving deeper levels' own
higher-index children of ever being selected regardless of their actual
density.
*/
// OUT - 4, under its own name (not SPECTRAL_DIM) - SPECTRAL_DIM itself is
// a framework-managed macro, #define'd/#undef'd around whichever other
// node's own generated block currently needs it (e.g. PBVR_ENV's own) -
// this file's own generated block is a separate, unrelated top-level
// block in the same compiled megakernel, so reusing that exact name here
// would either collide with an already-active definition or be undefined
// outside of that other block's own narrow scope.
#define XR_ENV_SPECTRAL_DIM (OUT - 4)

void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    int current_node = 0;
    float u0 = 0.0, u1 = 1.0, v0 = 0.0, v1 = 1.0;
    // densities stores *conditional* probabilities (a node's own 4
    // children, given that node - summing to ~1), not absolute ones - so
    // this leaf's own absolute probability is the *product* of every
    // level's own selected conditional probability along the path, not
    // just the last one.
    float prob = 1.0;
    float sel = random(); // single draw, reused (rescaled) every level

    for (int i = 0; i < LEVELS; i++)
    {
        int offset = current_node * 4;
        int selected_child = 3;
        float lo = 0.0, hi = deref(deref(node).densities)[offset + 3];
        for (int c = 0; c < 3; c++)
        {
            float cdf_c = deref(deref(node).densities)[offset + c];
            if (sel < cdf_c)
            {
                selected_child = c;
                hi = cdf_c;
                break;
            }
            lo = cdf_c;
        }
        prob *= max(0.0, hi - lo);
        sel = clamp((sel - lo) / max(0.000000001, hi - lo), 0.0, 1.0);
        float umed = (u0 + u1) * 0.5;
        float vmed = (v0 + v1) * 0.5;
        if (selected_child % 2 == 0) u1 = umed; else u0 = umed;
        if (selected_child / 2 == 0) v1 = vmed; else v0 = vmed;
        current_node = current_node * 4 + 1 + selected_child;
    }

    // this leaf pixel's own (alpha, beta) bounds - see xr2dir/dir2xr's
    // own convention (alpha in [-pi,pi], beta in [-pi/2,pi/2]).
    float alpha0 = (u0 * 2.0 - 1.0) * pi, alpha1 = (u1 * 2.0 - 1.0) * pi;
    float beta0 = (v0 * 2.0 - 1.0) * piOverTwo, beta1 = (v1 * 2.0 - 1.0) * piOverTwo;
    float sin_beta0 = sin(beta0), sin_beta1 = sin(beta1);
    float pixel_solid_angle = (alpha1 - alpha0) * (sin_beta1 - sin_beta0);
    float pdf = prob / max(0.000000001, pixel_solid_angle);
    float weight = 1.0 / max(0.000000001, pdf);

    // uniform-by-solid-angle sample within the leaf pixel (the "squeeze" -
    // alpha uniform, beta via inverse-CDF of the (already pole-squeezed)
    // sin(beta) distribution, so the sample is uniform in actual solid
    // angle across the pixel, not just uniform in (alpha, beta) itself).
    float alpha = mix(alpha0, alpha1, random());
    float s = mix(sin_beta0, sin_beta1, random());
    float beta = asin(clamp(s, -1.0, 1.0));
    vec3 dir = usc2dir(vec2(alpha, beta));

    float env[XR_ENV_SPECTRAL_DIM];
    forward(deref(node).environment, float[3](dir.x, dir.y, dir.z), env);

    _output[0] = dir.x;
    _output[1] = dir.y;
    _output[2] = dir.z;
    for (int i = 0; i < XR_ENV_SPECTRAL_DIM; i++)
        _output[3 + i] = env[i] * weight;
    _output[3 + XR_ENV_SPECTRAL_DIM] = pdf;
}

#undef XR_ENV_SPECTRAL_DIM
