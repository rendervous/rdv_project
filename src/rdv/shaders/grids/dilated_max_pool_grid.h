/*
Code for the DilatedMaxPoolGrid class - see its own (python) docstring.
Each output voxel is the max of g over the 3^IN neighborhood at offsets
{-dilation, 0, +dilation} per axis (boundary-clamped).
*/

#if IN == 1
void forward(RDV_NODE node, int _input[1], out float _output[OUT])
{
    int size0 = deref(deref(node).g).shape[0];
    int dil = deref(node).dilation;
    float m = 0.0;
    bool first = true;
    for (int d0 = -dil; d0 <= dil; d0 += dil)
    {
        int i0 = clamp(_input[0] + d0, 0, size0 - 1);
        float value[1];
        forward(deref(node).g, int[1](i0), value);
        m = first ? value[0] : max(m, value[0]);
        first = false;
    }
    _output[0] = m;
}
#elif IN == 2
void forward(RDV_NODE node, int _input[2], out float _output[OUT])
{
    int size0 = deref(deref(node).g).shape[0];
    int size1 = deref(deref(node).g).shape[1];
    int dil = deref(node).dilation;
    float m = 0.0;
    bool first = true;
    for (int d0 = -dil; d0 <= dil; d0 += dil)
    {
        int i0 = clamp(_input[0] + d0, 0, size0 - 1);
        for (int d1 = -dil; d1 <= dil; d1 += dil)
        {
            int i1 = clamp(_input[1] + d1, 0, size1 - 1);
            float value[1];
            forward(deref(node).g, int[2](i0, i1), value);
            m = first ? value[0] : max(m, value[0]);
            first = false;
        }
    }
    _output[0] = m;
}
#else
void forward(RDV_NODE node, int _input[3], out float _output[OUT])
{
    int size0 = deref(deref(node).g).shape[0];
    int size1 = deref(deref(node).g).shape[1];
    int size2 = deref(deref(node).g).shape[2];
    int dil = deref(node).dilation;
    float m = 0.0;
    bool first = true;
    for (int d0 = -dil; d0 <= dil; d0 += dil)
    {
        int i0 = clamp(_input[0] + d0, 0, size0 - 1);
        for (int d1 = -dil; d1 <= dil; d1 += dil)
        {
            int i1 = clamp(_input[1] + d1, 0, size1 - 1);
            for (int d2 = -dil; d2 <= dil; d2 += dil)
            {
                int i2 = clamp(_input[2] + d2, 0, size2 - 1);
                float value[1];
                forward(deref(node).g, int[3](i0, i1, i2), value);
                m = first ? value[0] : max(m, value[0]);
                first = false;
            }
        }
    }
    _output[0] = m;
}
#endif
