float transmittance_rm(RDV_NODE node, vec3 x, vec3 w, float d, float step_size, float density_scale)
{
    float tau = 0.0;
    for (int safety = 0; d > 0.0 && safety < 4096; safety++)
    {
        float current_extinction = extinction(node, x) * density_scale;
        tau += current_extinction * step_size; // trapezoidal rule for better accuracy
        if (tau > 16)
            return 0.0;
        d -= step_size;
        x += step_size * w;
    }
    return exp(-tau);
}
