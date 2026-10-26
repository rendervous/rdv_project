float voxel_to_alpha(RDV_NODE node, int voxel[IN], float alpha[IN], int corner_bits)
{
    // Given a indices of a voxel, and a frac/normalized alpha values inside that voxel
    // returns the multiplication of all alphas (or 1-alphas) depending on corner bits
    float w = 1.0;
    for (int i = 0; i < IN; i++)
        w *= (((corner_bits >> i) & 1) == 1) ? alpha[i] : (1.0 - alpha[i]);
    return w;
}

void voxel_index_to_corner(RDV_NODE node, int voxel[IN], int corner_bits, out int indices[IN])
{
    // gets the corner indices depending on the corner bits
    for (int i = 0; i < IN; i++) {
        int c = voxel[i] + ((corner_bits >> i) & 1);
        if (deref(node).clamp != 0)
            c = clamp(c, 0, deref(deref(node).g).shape[i] - 1);
        indices[i] = c;
    }
}

bool corner_is_inside(RDV_NODE node, int indices[IN])
{
    // whether every dimension of indices is a valid (in-range) grid index -
    // only meaningful when clamp == 0 (see forward): a clamp != 0 corner is
    // always already forced inside by voxel_index_to_corner.
    for (int i = 0; i < IN; i++)
        if (indices[i] < 0 || indices[i] >= deref(deref(node).g).shape[i])
            return false;
    return true;
}

void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    // map _input coordinate[-1...1] to voxel indices using shape - 1
    // get the frac between 0..1 of _input scaled to shape dimensions
    int voxel[IN];
    float alpha[IN];
    for (int i = 0; i < IN; i++) {
        // _input is fastest-component-first (x, y, z, ...); shape/g's own
        // indices are fastest-dimension-last (row-major) - dimension i of
        // the grid takes its coordinate from _input[IN-1-i].
        float t = _input[IN - 1 - i] * 0.5 + 0.5;
        float pos = deref(node).aligned != 0
            ? t * float(deref(deref(node).g).shape[i] - 1)
            : t * float(deref(deref(node).g).shape[i]) - 0.5;
        float fv = floor(pos);
        voxel[i] = int(fv);
        alpha[i] = pos - fv;
    }

    for (int k = 0; k < OUT; k++) _output[k] = 0;

    if (deref(node).linear == 0) {
        // nearest: round each dimension's continuous position to its own
        // closest voxel index (alpha >= 0.5 means the +1 side is closer),
        // then query that single cell with no blending.
        int indices[IN];
        for (int i = 0; i < IN; i++) {
            int c = voxel[i] + (alpha[i] >= 0.5 ? 1 : 0);
            if (deref(node).clamp != 0)
                c = clamp(c, 0, deref(deref(node).g).shape[i] - 1);
            indices[i] = c;
        }
        if (deref(node).clamp != 0 || corner_is_inside(node, indices))
            forward(deref(node).g, indices, _output);
        return;
    }

    // for all 2^IN bits configuration query the grid g
    //     multiply for the specific accumulated alpha
    int corner_count = 1 << IN;
    for (int corner_bits = 0; corner_bits < corner_count; corner_bits++) {
        int indices[IN];
        voxel_index_to_corner(node, voxel, corner_bits, indices);
        if (deref(node).clamp == 0 && !corner_is_inside(node, indices))
            continue;
        float w = voxel_to_alpha(node, voxel, alpha, corner_bits);
        float values[OUT];
        forward(deref(node).g, indices, values);
        for (int k = 0; k < OUT; k++) _output[k] += w * values[k];
    }
}