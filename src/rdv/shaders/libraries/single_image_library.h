// One image + one sampler, bound at their own two descriptor slots, for the
// generic ID this library was specialized with (see SingleImageLibrary in
// maths/__init__.py) - unlike ImagingLibrary's bindless arrays, every
// SingleImageLibrary[ID] is a separate, plain (non-array) binding pair, which
// is what WebGPU requires. Everything is named after ID so several IDs can
// coexist in one megakernel; SingleGPUSampler/SingleGPUGrid with the same ID
// call rdv_sample_image_<ID>/rdv_fetch_image_<ID>.
#if IN == 1
layout(set=0, binding=BINDING_SLOT(0)) uniform texture1D RDV_CONCAT(rdv_single_image_, ID);
#elif IN == 2
layout(set=0, binding=BINDING_SLOT(0)) uniform texture2D RDV_CONCAT(rdv_single_image_, ID);
#else
layout(set=0, binding=BINDING_SLOT(0)) uniform texture3D RDV_CONCAT(rdv_single_image_, ID);
#endif
layout(set=0, binding=BINDING_SLOT(1)) uniform sampler RDV_CONCAT(rdv_single_sampler_, ID);

// explicit LOD 0: implicit-derivative sampling doesn't exist in compute
// shaders (and WGSL has no textureSample outside fragment shaders)
#if IN == 1
vec4 RDV_CONCAT(rdv_sample_image_, ID)(float c) {
    return textureLod(sampler1D(RDV_CONCAT(rdv_single_image_, ID), RDV_CONCAT(rdv_single_sampler_, ID)), c, 0.0);
}
vec4 RDV_CONCAT(rdv_fetch_image_, ID)(int c) {
    return texelFetch(sampler1D(RDV_CONCAT(rdv_single_image_, ID), RDV_CONCAT(rdv_single_sampler_, ID)), c, 0);
}
#elif IN == 2
vec4 RDV_CONCAT(rdv_sample_image_, ID)(vec2 c) {
    return textureLod(sampler2D(RDV_CONCAT(rdv_single_image_, ID), RDV_CONCAT(rdv_single_sampler_, ID)), c, 0.0);
}
vec4 RDV_CONCAT(rdv_fetch_image_, ID)(ivec2 c) {
    return texelFetch(sampler2D(RDV_CONCAT(rdv_single_image_, ID), RDV_CONCAT(rdv_single_sampler_, ID)), c, 0);
}
#else
vec4 RDV_CONCAT(rdv_sample_image_, ID)(vec3 c) {
    return textureLod(sampler3D(RDV_CONCAT(rdv_single_image_, ID), RDV_CONCAT(rdv_single_sampler_, ID)), c, 0.0);
}
vec4 RDV_CONCAT(rdv_fetch_image_, ID)(ivec3 c) {
    return texelFetch(sampler3D(RDV_CONCAT(rdv_single_image_, ID), RDV_CONCAT(rdv_single_sampler_, ID)), c, 0);
}
#endif

void RDV_NODE(uvec3 idx) { }
