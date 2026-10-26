void scanline_interpolator(RDV_NODE node, inout float dst[OUT],
    F src_left, F src_right, float x_weight, float rest_weight) {
    for (int i=0; i<OUT; i++)
        dst[i] += rest_weight * mix(deref(src_left)[i], deref(src_right)[i], x_weight);
}

void blend_last_pair(RDV_NODE node, inout float dst[OUT], GPUPtr ptr, int stride,
    bool left_inside, bool right_inside, float x_weight, float rest_weight)
{
    // Blends the two (memory-adjacent, `stride` bytes apart) OUT-sized cells
    // at ptr (left) and ptr+stride (right) - via scanline_interpolator when
    // both are valid, or a direct single-side contribution when only one is
    // (clamp == 0 border fade).
    if (!left_inside && !right_inside)
        return;
    F src_left = cast(F, ptr);
    F src_right = cast(F, ptr + stride);
    if (left_inside && right_inside)
        scanline_interpolator(node, dst, src_left, src_right, x_weight, rest_weight);
    else if (left_inside)
        for (int k = 0; k < OUT; k++) dst[k] += rest_weight * (1.0 - x_weight) * deref(src_left)[k];
    else
        for (int k = 0; k < OUT; k++) dst[k] += rest_weight * x_weight * deref(src_right)[k];
}

void forward(RDV_NODE node, float _input[IN], out float _output[OUT])
{
    for (int k = 0; k < OUT; k++) _output[k] = 0;

    // t's own data is one contiguous, row-major buffer (shape[0..IN-1]
    // spatial cells of OUT floats each) - resolve its base pointer once,
    // every corner is then just base_ptr + a byte offset built from
    // per-dimension strides.
    GPUPtr base_ptr = address(resolve(deref(node).t.deferred_buffer));
    // Both aligned modes collapse into one branch-free formula: aligned
    // ties -1/+1 to cell centers (ac=1), non-aligned to the outer edges
    // (ac=0) - see the derivation in GridSampler's own docstring.
    float ac = deref(node).aligned != 0 ? 1.0 : 0.0;
    // clamp != 0 (border mode) means every corner is always "valid" (it
    // gets clamped to the nearest edge cell instead of being dropped) -
    // folded directly into lo_in/hi_in below so every downstream branch
    // that only checks "in range" gets clamp's override for free.
    bool border = deref(node).clamp != 0;

#if IN == 1
    // 1D: no "outer" dimension at all - just the last-dimension pair.
    float gsize = float(deref(node).t.shape[0]);
    float index_f = (_input[0] * (gsize - ac) + gsize) * 0.5 - 0.5;
    int lo = int(floor(index_f));
    int hi = lo + 1;
    float alpha = index_f - float(lo);

    if (deref(node).linear == 0) {
        int idx = lo + (alpha >= 0.5 ? 1 : 0);
        bool inside = idx >= 0 && idx < int(gsize);
        idx = clamp(idx, 0, int(gsize) - 1);
        if (deref(node).clamp != 0 || inside) {
            F ptr = cast(F, base_ptr + uint64_t(idx) * OUT * 4);
            for (int k = 0; k < OUT; k++) _output[k] = deref(ptr)[k];
        }
        return;
    }

    bool lo_in = border || (lo >= 0 && lo < int(gsize));
    bool hi_in = border || (hi >= 0 && hi < int(gsize));
    int clo = clamp(lo, 0, int(gsize) - 1);
    int chi = clamp(hi, 0, int(gsize) - 1);

    int stride0 = OUT * 4;
    GPUPtr ptr = base_ptr + uint64_t(clo) * stride0;
    int d0 = stride0 * (chi - clo);
    blend_last_pair(node, _output, ptr, d0, lo_in, hi_in, alpha, 1.0);

#elif IN == 2
    // 2D: dim 0 is the (single) outer dimension, dim 1 the last/scanline
    // one - position/voxel/alpha for both dims computed together as
    // vec2/ivec2 (mul-add friendly), clamp applied exactly once per
    // dimension/side, and the two corners walked via a single pointer,
    // moved by a precomputed stride instead of two independent address
    // computations.
    uvec2 dim = uvec2(deref(node).t.shape[0], deref(node).t.shape[1]);
    vec2 gsize = vec2(dim);
    // _input is fastest-component-first (x, y); t's shape is
    // fastest-dimension-last (row-major) - reverse to pair dim i with
    // _input[IN-1-i].
    vec2 index_f = (vec2(_input[1], _input[0]) * (gsize - ac) + gsize) * 0.5 - vec2(0.5);
    ivec2 lo = ivec2(floor(index_f));
    ivec2 hi = lo + ivec2(1);
    vec2 alpha = index_f - vec2(lo);

    if (deref(node).linear == 0) {
        ivec2 idx = lo + ivec2(greaterThanEqual(alpha, vec2(0.5)));
        bool inside = all(greaterThanEqual(idx, ivec2(0))) && all(lessThan(idx, ivec2(dim)));
        idx = clamp(idx, ivec2(0), ivec2(dim) - 1);
        if (deref(node).clamp != 0 || inside) {
            int spatial = idx.x * int(dim.y) + idx.y;
            F ptr = cast(F, base_ptr + uint64_t(spatial) * OUT * 4);
            for (int k = 0; k < OUT; k++) _output[k] = deref(ptr)[k];
        }
        return;
    }

    bvec2 lo_in = bvec2(border || (lo.x >= 0 && lo.x < int(dim.x)), border || (lo.y >= 0 && lo.y < int(dim.y)));
    bvec2 hi_in = bvec2(border || (hi.x >= 0 && hi.x < int(dim.x)), border || (hi.y >= 0 && hi.y < int(dim.y)));
    ivec2 clo = clamp(lo, ivec2(0), ivec2(dim) - 1);
    ivec2 chi = clamp(hi, ivec2(0), ivec2(dim) - 1);

    int stride1 = OUT * 4;                    // dim 1 (last/fastest)
    int stride0 = int(dim.y) * stride1;       // dim 0
    GPUPtr ptr = base_ptr + uint64_t(clo.x) * stride0 + uint64_t(clo.y) * stride1;
    int d1 = stride1 * (chi.y - clo.y);
    int d0 = stride0 * (chi.x - clo.x);

    blend_last_pair(node, _output, ptr, d1, lo_in.x && lo_in.y, lo_in.x && hi_in.y, alpha.y, 1.0 - alpha.x);
    ptr += d0;
    blend_last_pair(node, _output, ptr, d1, hi_in.x && lo_in.y, hi_in.x && hi_in.y, alpha.y, alpha.x);

#elif IN == 3
    // 3D: dims 0,1 are the outer dimensions, dim 2 the last/scanline one -
    // same idea as IN==2 (vec3/ivec3, clamp once, single walked pointer),
    // visiting the 4 outer corners in Gray-code order (lo,lo)->(lo,hi)->
    // (hi,hi)->(hi,lo) so each step moves the pointer by one precomputed
    // delta instead of recomputing a full address.
    uvec3 dim = uvec3(deref(node).t.shape[0], deref(node).t.shape[1], deref(node).t.shape[2]);
    vec3 gsize = vec3(dim);
    // _input is fastest-component-first (x, y, z); t's shape is
    // fastest-dimension-last (row-major) - reverse to pair dim i with
    // _input[IN-1-i].
    vec3 index_f = (vec3(_input[2], _input[1], _input[0]) * (gsize - ac) + gsize) * 0.5 - vec3(0.5);
    ivec3 lo = ivec3(floor(index_f));
    ivec3 hi = lo + ivec3(1);
    vec3 alpha = index_f - vec3(lo);

    if (deref(node).linear == 0) {
        ivec3 idx = lo + ivec3(greaterThanEqual(alpha, vec3(0.5)));
        bool inside = all(greaterThanEqual(idx, ivec3(0))) && all(lessThan(idx, ivec3(dim)));
        idx = clamp(idx, ivec3(0), ivec3(dim) - 1);
        if (deref(node).clamp != 0 || inside) {
            int spatial = (idx.x * int(dim.y) + idx.y) * int(dim.z) + idx.z;
            F ptr = cast(F, base_ptr + uint64_t(spatial) * OUT * 4);
            for (int k = 0; k < OUT; k++) _output[k] = deref(ptr)[k];
        }
        return;
    }

    bvec3 lo_in = bvec3(border || (lo.x >= 0 && lo.x < int(dim.x)), border || (lo.y >= 0 && lo.y < int(dim.y)), border || (lo.z >= 0 && lo.z < int(dim.z)));
    bvec3 hi_in = bvec3(border || (hi.x >= 0 && hi.x < int(dim.x)), border || (hi.y >= 0 && hi.y < int(dim.y)), border || (hi.z >= 0 && hi.z < int(dim.z)));
    ivec3 clo = clamp(lo, ivec3(0), ivec3(dim) - 1);
    ivec3 chi = clamp(hi, ivec3(0), ivec3(dim) - 1);

    int stride2 = OUT * 4;                    // dim 2 (last/fastest)
    int stride1 = int(dim.z) * stride2;       // dim 1
    int stride0 = int(dim.y) * stride1;       // dim 0
    GPUPtr ptr = base_ptr + uint64_t(clo.x) * stride0 + uint64_t(clo.y) * stride1 + uint64_t(clo.z) * stride2;
    int d2 = stride2 * (chi.z - clo.z);
    int d1 = stride1 * (chi.y - clo.y);
    int d0 = stride0 * (chi.x - clo.x);

    blend_last_pair(node, _output, ptr, d2, lo_in.x && lo_in.y && lo_in.z, lo_in.x && lo_in.y && hi_in.z,
        alpha.z, (1.0 - alpha.x) * (1.0 - alpha.y));
    ptr += d1;
    blend_last_pair(node, _output, ptr, d2, lo_in.x && hi_in.y && lo_in.z, lo_in.x && hi_in.y && hi_in.z,
        alpha.z, (1.0 - alpha.x) * alpha.y);
    ptr += d0;
    blend_last_pair(node, _output, ptr, d2, hi_in.x && hi_in.y && lo_in.z, hi_in.x && hi_in.y && hi_in.z,
        alpha.z, alpha.x * alpha.y);
    ptr -= d1;
    blend_last_pair(node, _output, ptr, d2, hi_in.x && lo_in.y && lo_in.z, hi_in.x && lo_in.y && hi_in.z,
        alpha.z, alpha.x * (1.0 - alpha.y));

#else
    // generic fallback for IN != 1, 2, 3 - shape/clo/chi/validity resolved
    // once per dimension up front (not re-clamped every outer corner).
    int shape[IN];
    int lo[IN];
    int clo[IN];
    int chi[IN];
    float alpha[IN];
    bool lo_in[IN];
    bool hi_in[IN];
    for (int i = 0; i < IN; i++) {
        shape[i] = deref(node).t.shape[i];
        float gsize = float(shape[i]);
        // _input is fastest-component-first, shape is fastest-dimension-last.
        float index_f = (_input[IN - 1 - i] * (gsize - ac) + gsize) * 0.5 - 0.5;
        int l = int(floor(index_f));
        int h = l + 1;
        lo[i] = l;
        alpha[i] = index_f - float(l);
        lo_in[i] = border || (l >= 0 && l < shape[i]);
        hi_in[i] = border || (h >= 0 && h < shape[i]);
        clo[i] = clamp(l, 0, shape[i] - 1);
        chi[i] = clamp(h, 0, shape[i] - 1);
    }

    if (deref(node).linear == 0) {
        // nearest: round every dimension to its own closest index, then
        // read a single OUT-sized chunk directly - no blending.
        int idx[IN];
        bool inside = true;
        for (int i = 0; i < IN; i++) {
            bool pick_hi = alpha[i] >= 0.5;
            idx[i] = pick_hi ? chi[i] : clo[i];
            if (!(pick_hi ? hi_in[i] : lo_in[i]))
                inside = false;
        }
        if (deref(node).clamp != 0 || inside) {
            int spatial_index = 0;
            for (int i = 0; i < IN; i++)
                spatial_index = spatial_index * shape[i] + idx[i];
            F ptr = cast(F, base_ptr + uint64_t(spatial_index) * OUT * 4);
            for (int k = 0; k < OUT; k++) _output[k] = deref(ptr)[k];
        }
        return;
    }

    // for all 2^(IN-1) bits configuration of every dimension but the last
    //     query the two (memory-adjacent) neighbors along the last
    //     dimension in one visit via scanline_interpolator, weighted by
    //     the accumulated alpha of the other dimensions (rest_weight)
    int last = IN - 1;
    int outer_count = 1 << last;
    for (int bits = 0; bits < outer_count; bits++) {
        int idx[IN];
        float rest_weight = 1.0;
        bool outer_inside = true;
        for (int i = 0; i < last; i++) {
            int bit = (bits >> i) & 1;
            rest_weight *= bit == 1 ? alpha[i] : (1.0 - alpha[i]);
            idx[i] = bit == 1 ? chi[i] : clo[i];
            if (!(bit == 1 ? hi_in[i] : lo_in[i]))
                outer_inside = false;
        }
        if (deref(node).clamp == 0 && !outer_inside)
            continue;

        // left/right differ only in the last (fastest-varying) dimension,
        // so their linear indices are always exactly `chi[last]-clo[last]`
        // apart - one full address computation, then plain pointer
        // arithmetic (stride) for the other corner instead of a second one.
        idx[last] = clo[last];
        int left_spatial = 0;
        for (int i = 0; i < IN; i++)
            left_spatial = left_spatial * shape[i] + idx[i];

        GPUPtr ptr = base_ptr + uint64_t(left_spatial) * OUT * 4;
        int stride = (chi[last] - clo[last]) * OUT * 4;
        blend_last_pair(node, _output, ptr, stride, lo_in[last], hi_in[last], alpha[last], rest_weight);
    }
#endif
}
