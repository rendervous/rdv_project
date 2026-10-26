float regular_coord(RDV_NODE node, int index, int n) {
    if (deref(node).mode == 0)
        // corner: n-1 voxels span the volume, points ARE the n corners.
        return n > 1 ? (float(index) / float(n - 1)) * 2.0 - 1.0 : 0.0;
    float offset = deref(node).mode == 2 ? random() : 0.5;
    return ((float(index) + offset) / float(n)) * 2.0 - 1.0;
}

void forward(RDV_NODE node, int _input[IN], out float _output[OUT]) {
    for (int k = 0; k < OUT; k++) {
        int dim = IN - 1 - k;
        _output[k] = regular_coord(node, _input[dim], deref(node).shape[dim]);
    }
}
