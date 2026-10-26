void forward(RDV_NODE node, int _input[IN], out float _output[OUT])
{
    float chosen_value[2];
    forward(deref(node).chosen, _input, chosen_value);
    float candidate_value[1];
    forward(deref(node).candidate, _input, candidate_value);

    float new_radius = deref(node).new_radius;
    float voxel_size = deref(node).voxel_size;
    float density_scale = deref(node).density_scale;

    float diff = candidate_value[0] - chosen_value[0];
    float threshold = 1.0 / (voxel_size * new_radius * density_scale);
    if (diff < threshold)
    {
        _output[0] = candidate_value[0];
        _output[1] = new_radius;
    }
    else
    {
        _output[0] = chosen_value[0];
        _output[1] = chosen_value[1];
    }
}
