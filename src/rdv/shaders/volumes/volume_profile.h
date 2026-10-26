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
//    float step_size = deref(node).step_size;
//    float density_scale = deref(node).density_scale;
    tMin = max(0, tMin);
    if (tMax > 0 && tMin <= tMax) // ray points away from the volume
    {
        // Skip empty space before the first possible intersection, and
        // past the last one - majorant(node, x, w, out max_step) reports
        // both an upper bound on extinction over some neighborhood of x
        // along w AND how far (parametrically, along wo) that bound holds
        // (see MajorantTrait/VoxelMajorantField/SphereMajorantField);
        // wherever the bound is exactly 0 there is provably no density to
        // raymarch through, so jump straight to the far edge of that
        // region instead of stepping through it one step_size at a time
        // below. Bounded by a safety cap (a macrocell field can in
        // principle report an arbitrarily small max_step right at a cell
        // boundary) - same defensive idiom PBVRT's own free-flight loop
        // uses (see pbvrt.h) so a degenerate majorant field can never hang
        // the whole dispatch. Only trims the ray's own two ends, not gaps
        // in the middle (interleaving this same per-iteration check into
        // the fixed-step marching loops below was tried and rolled back -
        // divergence made it a net loss whenever density/empty regions
        // alternate along the ray; see VolumeProfileDDA/
        // volume_profile_dda.h for a real cell-by-cell DDA alternative
        // that skips interior gaps too, compared side by side in
        // sandbox_volume_profile_interactive.py).
        for (int skip = 0; skip < 4096 && tMin < tMax; skip++)
        {
            float max_step;
            float maj = majorant(node, x + wo * tMin, wo, max_step);
            if (maj > 0.0)
                break;
            tMin += max(max_step, 1e-5);
        }

        for (int skip = 0; skip < 4096 && tMin < tMax; skip++)
        {
            float max_step;
            float maj = majorant(node, x + wo * tMax, -wo, max_step);
            if (maj > 0.0)
                break;
            tMax -= max(max_step, 1e-5);
        }

        // compute tau
        float tau = .0;
        float t = tMin ;//+ random() * deref(node).step_size;
        while (t < tMax)
        {
            float extinction_val = extinction(node, x + wo * t) * deref(node).density_scale; // sample extinction at the middle of the step for better accuracy
            tau += extinction_val * deref(node).step_size;
            t += deref(node).step_size;
            if (tau > 20.0)
                break; // transmittance is too low, can stop early
        }

        t = tMin ;//+ random() * deref(node).step_size;  // restart raymarching
        float current_tau = 0.0;
        int current_depth_index  = 0;
        float scale = deref(node).transmittance_threshold[current_depth_index];
        float tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau > 20 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau)); // expected tau when crossing the transmittance threshold, derived from the exponential distribution of free-flight distance
        if (tau > 0.0001)
            while (t < tMax)
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
                        break;
                t += deref(node).step_size;
                if (current_tau > 20.0)
                    break; // transmittance is too low, can stop early
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

/* DO NOT TOUCH

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
        // compute tau
        float tau = .0;
        float t = tMin + random() * deref(node).step_size;
        while (t < tMax)
        {
            float extinction_val = extinction(node, x + wo * t) * deref(node).density_scale; // sample extinction at the middle of the step for better accuracy
            tau += extinction_val * deref(node).step_size;
            t += deref(node).step_size;
            if (tau > 20.0)
                break; // transmittance is too low, can stop early
        }

        t = tMin + random() * deref(node).step_size;  // restart raymarching
        float current_tau = 0.0;
        int current_depth_index  = 0;
        float scale = deref(node).transmittance_threshold[current_depth_index];
        float tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau > 20 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau)); // expected tau when crossing the transmittance threshold, derived from the exponential distribution of free-flight distance
        if (tau > 0.0001)
            while (t < tMax)
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
                        break;
                t += deref(node).step_size;
                if (current_tau > 20.0)
                    break; // transmittance is too low, can stop early
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
            float nee_density_scale = (0.2 + 0.8 * (1 - g)) * deref(node).density_scale;
            float nee_T;
            if (tMax < 0 || tMax < tMin)
            {
                nee_T = 1.0;
            }
            else
            {
                // Split the shadow ray: a short, deterministic raymarched
                // segment (finer step, step_size*0.5) right next to
                // sample_pos, then ratio-traced for the rest. sample_pos is
                // itself already a real collision, i.e. a locally dense
                // spot - a coarse spatial majorant is at its loosest right
                // around such spots/near the effective volume's own
                // boundary (see MajorantGrid/SphereMajorantField), and ratio
                // tracking's own rejection cost scales with how loose the
                // majorant is *locally*, not with distance travelled - so
                // ratio tracking alone would burn many null-collision
                // retries in just this first short stretch for no benefit.
                // The remainder of the ray is usually far emptier - exactly
                // where ratio tracking's own free-flight skipping (majorant
                // distance can jump whole macrocells at once) actually pays
                // off over raymarching's fixed small steps.
                float near_d = min(deref(node).nee_near_distance, tMax);
                float T_near = transmittance_rm(node, sample_pos, light_dir_object_space, near_d, deref(node).step_size * 0.5, nee_density_scale);
                if (near_d < tMax)
                {
                    // Russian roulette on T_near instead of always
                    // continuing: survive with probability T_near and, on
                    // survival, take the far segment's own transmittance
                    // un-scaled (not T_near*T_far) - T_near itself *is* the
                    // survival probability here, not a 1/p reweighting
                    // factor, so E[estimator] = T_near*T_far + (1-T_near)*0
                    // = T_near*T_far, exactly the deterministic product,
                    // while skipping the far segment's own ratio-tracing
                    // cost entirely once the near segment already mostly
                    // occludes the light (T_near small) instead of tracing
                    // it every time only to multiply by a near-zero factor.
                    nee_T = random() < T_near
                        ? transmittance_rt(node, sample_pos + light_dir_object_space * near_d, light_dir_object_space, tMax - near_d, nee_density_scale)
                        : 0.0;
                }
                else
                {
                    nee_T = T_near;
                }
            }
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
            vec3 selected_dir = w;
            vec3 selected_dir_object_space = wo;
            float selected_weight = 0.0;
            float selected_tMax = 0.0;
            float total_weight = 0.0;
            for (int s=0; s <  env_samples; s++)
            {
                vec3 cand_dir = hg_phase_sample(w, g); // importance sample wrt the phase function
                vec3 cand_dir_object_space = L * cand_dir;
                float cand_tMin, cand_tMax;
                ray_box_intersection(sample_pos, cand_dir_object_space, cand_tMin, cand_tMax);
                float cand_weight = 0.0;
                if (cand_tMax >= 0 && cand_tMax >= cand_tMin)
                {
                    float cand_near_d = min(deref(node).nee_near_distance, cand_tMax);
                    cand_weight = transmittance_rt(node, sample_pos, cand_dir_object_space, cand_near_d, nee_density_scale);
                }
                total_weight += cand_weight;
                if (cand_weight > 0.0 && random() < cand_weight / total_weight) // weighted reservoir update
                {
                    selected_dir = cand_dir;
                    selected_dir_object_space = cand_dir_object_space;
                    selected_weight = cand_weight;
                    selected_tMax = cand_tMax;
                }
            }
            if (total_weight > 0.0)
            {
                float env[SPECTRAL_DIM];
                environment(node, selected_dir, env);
                // Continue past the already-computed near segment
                // (selected_weight) with the same Russian-roulette
                // near/far split used for the direct-light shadow ray.
                float near_d = min(deref(node).nee_near_distance, selected_tMax);
                float nee_T;
                if (near_d < selected_tMax)
                {
                    nee_T = random() < selected_weight
                        ? transmittance_rt(node, sample_pos + selected_dir_object_space * near_d, selected_dir_object_space, selected_tMax - near_d, nee_density_scale)
                        : 0.0;
                }
                else
                {
                    nee_T = selected_weight;
                }
                float ris_scale = total_weight / (float(env_samples) * selected_weight);
                for (int i=0; i<SPECTRAL_DIM; i++)
                    // nee contribution
                    _output[DEPTH_SAMPLES + 2 + i] = S * nee_T * env[i] * ris_scale; // env * phase * albedo * transmittance
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

*/