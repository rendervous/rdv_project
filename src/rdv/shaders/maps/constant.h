#if IN == 0
void forward(RDV_NODE node, out float _output[OUT]) {
#else
void forward(RDV_NODE node, float _input[IN], out float _output[OUT]) {
#endif
    _output = deref(resolve(deref(node).buf_));
}
