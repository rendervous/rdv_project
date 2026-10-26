/*
Evaluates the sampler and gets the direction (return) and env contribution already weighted.
*/
vec3 environment_sampler(RDV_NODE node, vec3 x, vec3 win, out float env[SPECTRAL_DIM])
{
    float result[3 + SPECTRAL_DIM + 1]; // dir(3) weighed_env(SPECTRAL_DIM) inv_pdf(1) <- only used by samplers in MIS
    forward(deref(node).environment_sampler, float[6](x.x, x.y, x.z, win.x, win.y, win.z), result);
    for(int i=0; i<SPECTRAL_DIM; i++)
        env[i] = result[3 + i];
    return vec3(result[0], result[1], result[2]);
}
