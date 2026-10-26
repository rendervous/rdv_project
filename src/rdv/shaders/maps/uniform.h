#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
    for (int k = 0; k < OUT; k++) _output[k] = random();
}
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
    for (int k = 0; k < OUT; k++) _output[k] = random();
}
#endif
