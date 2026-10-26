float transmittance_rt(RDV_NODE node, vec3 x, vec3 w, float d, float density_scale)
{
    float T = 1.0;
    // Safety cap (see pbvr.h's own copy of this same guard): a DDA step
    // landing within float precision of a macrocell boundary can report a
    // ~0 max_distance, which would otherwise make no progress and hang the
    // whole dispatch. max_distance is also floored for the same reason -
    // both are defensive, not expected to matter in the common case.
    for (int safety = 0; d > 0.0 && safety < 1024; safety++)
//    while(d > 0.0)
    {
        float maj_distance;
        float maj = max(majorant(node, x, w, maj_distance), 1e-12);
        maj_distance = max(maj_distance, 1e-6);
        float dt = min(maj_distance, -log(1.0 - random()) / (maj * density_scale)); // free-flight distance
        d -= dt;
        x += dt * w;
        if (dt >= maj_distance)
            continue;
        if (d <= 0.0)
            break; // reached the target distance without a collision
        float sigma_t = extinction(node, x);
        T *= (1.0 - sigma_t / maj); // necessary multiply two times by density_scale.
        if (T < 0.05)
        {
            if (random() >= T)
                return 0.0;
            T = 1.0; // Russian roulette
        }
    }
    return T;
}
