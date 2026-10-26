void generate_random_sphere(RDV_NODE node, out float _output[OUT])
{
    float sqr_len = 1.0;
    bool accepted = false;
    // Safety-capped rejection sampling (was `while (true)`) - accept rate
    // for a unit ball inside its bounding box is pi/4 (~78.5%) at OUT=2 and
    // (4/3*pi)/8 (~52.4%) at OUT=3, so 64 tries failing is astronomically
    // unlikely; caps worst-case iteration count instead of letting a GPU
    // thread spin indefinitely (see pbvrt.h's own note on why an unbounded
    // loop here is a real TDR/crash risk, not just a theoretical one).
    for (int safety = 0; safety < 64; safety++) {
        sqr_len = .0;
        // generate random in box
        for (int k = 0; k < OUT; k++) {
            _output[k] = 2 * random() - 1;
            sqr_len += _output[k] * _output[k];
        }
        // check sqr length
        if (sqr_len <= 1.0 && sqr_len > 0.0) {
            accepted = true;
            break;
        }
    }
    if (!accepted) {
        // deterministic fallback sample - never reached in practice
        _output[0] = 1.0;
        for (int k = 1; k < OUT; k++)
            _output[k] = 0.0;
        sqr_len = 1.0;
    }
    sqr_len = sqrt(sqr_len);
    for (int i=0; i<OUT; i++)
        _output[i] /= sqr_len;
}
#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
    generate_random_sphere(node, _output);
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    generate_random_sphere(node, _output);
}
#endif
