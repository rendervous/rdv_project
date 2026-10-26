void RDV_NODE(uvec3 idx) { }

int indices_to_global(in int shape[N], int indices[N]) {
    int global_index = 0;
    for (int i = 0; i < N; i++)
        global_index = global_index * shape[i] + indices[i];
    return global_index;
}

void global_to_indices(in int shape[N], int global_index, out int indices[N])
{
    for (int i = N - 1; i >= 0; i--) {
        indices[i] = global_index % shape[i];
        global_index /= shape[i];
    }
}
