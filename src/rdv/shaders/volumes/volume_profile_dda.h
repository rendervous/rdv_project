void forward(RDV_NODE node, float _input[6], out float _output[OUT])
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    // Transform ray to local space
    mat4x3 M = deref(node).transform; // from object to world space
    mat3 L = inverse(mat3(M));
    vec3 O = M[3];
    x = L * (x - O); // convert world position to object space position (use same x)
    vec3 wo = L * w; // convert world direction to object space direction
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function

    for (int i=0; i < OUT; i++)
        _output[i] = 0.0; // transmittance 1 till infinity

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    tMin = max(0, tMin);
    if (tMax > 0 && tMin <= tMax) // ray points away from the volume
    {
        // Cell-by-cell DDA traversal of majorant_grid (a plain index ->
        // max_value grid, e.g. rdv.MajorantGrid's own output wrapped in a
        // TensorGrid - see VolumeProfileDDA.__init__), instead of
        // volume_profile.h's repeated majorant()/MajorantField re-queries
        // (which only ever trims the ray's own two ends - see that file's
        // own comment on why interleaving it into the fixed-step marching
        // loops too was tried and rolled back). Classic Amanatides-Woo DDA:
        // tMaxAxis/tDelta are set up once, from the ray's own entry cell,
        // then updated with a handful of adds per cell step instead of
        // re-deriving the current cell/exit distance from scratch every
        // time (as majorant()/VoxelMajorantField's own forward() does) -
        // cheap enough to check *every* cell along the ray without the
        // per-iteration divergence cost that made interleaving the general
        // majorant() query into every fixed-step iteration a net loss.
        ivec3 dim = ivec3(
            deref(deref(node).majorant_grid).shape[2],
            deref(deref(node).majorant_grid).shape[1],
            deref(deref(node).majorant_grid).shape[0]
        );
        vec3 cell_size = vec3(2.0) / vec3(dim);
        vec3 entry_pos = x + wo * tMin;
        ivec3 entry_cell = clamp(ivec3(floor((entry_pos * 0.5 + 0.5) * vec3(dim))), ivec3(0), dim - ivec3(1));
        ivec3 dda_step = ivec3(sign(wo));
        vec3 inv_wo = 1.0 / wo; // a 0 component makes that axis' own tMaxAxis/tDelta +-inf below (never the smallest), same robust idiom math_library.h's ray_box_intersection relies on
        vec3 next_boundary = (vec3(entry_cell) + max(vec3(dda_step), vec3(0.0))) * cell_size - vec3(1.0);
        vec3 entry_tMaxAxis = (next_boundary - x) * inv_wo;
        vec3 tDelta = cell_size * abs(inv_wo);

        // compute tau
        float tau = .0;
        {
            ivec3 cell = entry_cell;
            vec3 tMaxAxis = entry_tMaxAxis;
            float t = tMin;
            for (int guard = 0; guard < 4096 && t < tMax; guard++)
            {
                float cell_exit = min(min(tMaxAxis.x, tMaxAxis.y), min(tMaxAxis.z, tMax));
                float cell_majorant[1];
                forward(deref(node).majorant_grid, int[3](cell.z, cell.y, cell.x), cell_majorant);
                if (cell_majorant[0] > 0.0)
                {
                    while (t < cell_exit)
                    {
                        float extinction_val = extinction(node, x + wo * t) * deref(node).density_scale; // sample extinction at the middle of the step for better accuracy
                        tau += extinction_val * deref(node).step_size;
                        t += deref(node).step_size;
                        if (tau > 20.0)
                            break; // transmittance is too low, can stop early
                    }
                    if (tau > 20.0)
                        break;
                }
                t = cell_exit;
                if (tMaxAxis.x <= tMaxAxis.y && tMaxAxis.x <= tMaxAxis.z)
                    { cell.x += dda_step.x; tMaxAxis.x += tDelta.x; }
                else if (tMaxAxis.y <= tMaxAxis.z)
                    { cell.y += dda_step.y; tMaxAxis.y += tDelta.y; }
                else
                    { cell.z += dda_step.z; tMaxAxis.z += tDelta.z; }
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dim)))
                    break; // walked off the majorant grid
            }
        }

        float current_tau = 0.0;
        int current_depth_index  = 0;
        float scale = deref(node).transmittance_threshold[current_depth_index];
        float tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau > 20 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau)); // expected tau when crossing the transmittance threshold, derived from the exponential distribution of free-flight distance
        if (tau > 0.0001)
        {
            ivec3 cell = entry_cell;
            vec3 tMaxAxis = entry_tMaxAxis;
            float t = tMin; // restart raymarching
            bool done = false;
            for (int guard = 0; guard < 4096 && t < tMax && !done; guard++)
            {
                float cell_exit = min(min(tMaxAxis.x, tMaxAxis.y), min(tMaxAxis.z, tMax));
                float cell_majorant[1];
                forward(deref(node).majorant_grid, int[3](cell.z, cell.y, cell.x), cell_majorant);
                if (cell_majorant[0] > 0.0)
                    while (t < cell_exit)
                    {
                        float extinction_val = extinction(node, x + wo * t) * deref(node).density_scale; // sample extinction at the middle of the step for better accuracy
                        current_tau += extinction_val * deref(node).step_size;
                        while (current_tau * scale >= tau_crossing)
                        {
                            _output[current_depth_index] = t; // 1 - 1 / (1 + current_tau*0.5); // 1 - exp(-current_tau);
                            current_depth_index++;
                            if (current_depth_index >= DEPTH_SAMPLES)
                                break;
                            scale = deref(node).transmittance_threshold[current_depth_index];
                            tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau > 20 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau));
                        }
                        if (current_depth_index >= DEPTH_SAMPLES)
                        {
                            done = true;
                            break;
                        }
                        t += deref(node).step_size;
                        if (current_tau > 20.0)
                        {
                            done = true;
                            break;
                        }
                    }
                if (done)
                    break;
                t = cell_exit;
                if (tMaxAxis.x <= tMaxAxis.y && tMaxAxis.x <= tMaxAxis.z)
                    { cell.x += dda_step.x; tMaxAxis.x += tDelta.x; }
                else if (tMaxAxis.y <= tMaxAxis.z)
                    { cell.y += dda_step.y; tMaxAxis.y += tDelta.y; }
                else
                    { cell.z += dda_step.z; tMaxAxis.z += tDelta.z; }
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dim)))
                    break; // walked off the majorant grid
            }
        }

        float S = 1 - exp(-tau);

        _output[OUT - 1] = S; // interaction probability

        // compute nee and light dir for the first sample
        if (current_depth_index > 0) // if we have at least one valid sample, we can compute nee and scattering for the first sample
        {
            float check_t = _output[deref(node).sample_index]; // index to the depth sample that we want to check for nee, scattering, etc.

            // reencode depths
            for (int i=DEPTH_SAMPLES-1; i >= 1; i--)
                _output[i] = 1 / (1 + 10 * pow(_output[i] - _output[i-1], 0.5));//1/(1 + 4 * pow(_output[i] - _output[i-1], 0.5));
            _output[0] = 1 / (0.1 + _output[0]);

            vec3 sample_pos = x + wo * check_t; // position of the last sample in object space
            // sample light source for the first sample
            float g = deref(node).g;
            // feature: transmittance on a reduced medium
            _output[DEPTH_SAMPLES] = 0.2 * log(1 + tau); // 1 - exp(-tau);
            float sa[SPECTRAL_DIM];
            scattering_albedo(node, sample_pos, sa);
            vec3 light_dir = deref(node).light_direction;
            //float rho_nee = hg_phase_eval(w, light_dir, g);
            vec3 light_dir_object_space = L * light_dir; // convert light direction to object space
            ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
            float nee_T = tMax < 0 || tMax < tMin ? 1.0 : transmittance_rm(node, sample_pos, light_dir_object_space, tMax, deref(node).step_size, (0.2 + 0.8 * (1 - g)) * deref(node).density_scale);

            _output[DEPTH_SAMPLES + 1] = S * nee_T; // env_sampler[3 + i] * nee_T; // env * phase * albedo * transmittance
            // Resampled importance sampling (RIS) over the environment
            // directions: draw env_samples i.i.d. candidates (same
            // phase-importance sampling as before - phase(w,x)/phase_pdf(x)
            // == 1 by construction, same as the plain-average estimator
            // this replaces), weight each one cheaply by its own near-
            // segment (nee_near_distance) transmittance via ratio tracking
            // - a proxy correlated with the true, expensive-to-get full
            // transmittance, since occlusion right next to a real collision
            // is the dominant source of variance here - then pick ONE
            // candidate via weighted reservoir sampling (single pass, no
            // need to store all M directions) instead of fully evaluating
            // (environment lookup + full near+far shadow ray) all of them.
            // RIS is unbiased for *any* positive resampling weight (not
            // just one proportional to the true integrand) as long as the
            // final contribution is rescaled by (mean weight over all
            // candidates) / (selected candidate's own weight) - see
            // selected_weight/total_weight below.
            int env_samples = deref(node).environment_samples;
            for (int s=0; s < env_samples; s++)
            {
                light_dir = hg_phase_sample(w, g); // importance sample wrt the phase function
                float env[SPECTRAL_DIM];
                environment(node, light_dir, env);
                light_dir_object_space = L * light_dir; // convert light direction to object space
                ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
                nee_T = transmittance_rt(node, sample_pos, light_dir_object_space, tMax, (0.2 + 0.8 * (1 - g)) * deref(node).density_scale);

                for (int i=0; i<SPECTRAL_DIM; i++)
                    // nee contribution
                    _output[DEPTH_SAMPLES + 2 + i] += S * nee_T * env[i] / env_samples; // env * phase * albedo * transmittance
            }
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 2 + SPECTRAL_DIM + i] = 1 / (1 - log(1-sa[i])); // albedo
            _output[DEPTH_SAMPLES + 2 + 2 * SPECTRAL_DIM] = g; // anisotropy
            // not used as input of the model
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 3 + 2 * SPECTRAL_DIM + i] = deref(node).light_intensity[i];
        }
    }
}
