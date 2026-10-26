// Array sizes must match ImagingLibrary.MAX_IMAGES/MAX_SAMPLERS in maps.py
// - the underlying vk binding layer has no support for a true
// runtime-sized/"unbound" descriptor array, so these are plain fixed-size
// arrays; still-unused slots are simply never indexed by a valid image/
// sampler index (see ImagingLibrary.bind_image_1d/2d/3d/bind_sampler).
layout(set=0, binding=BINDING_SLOT(0)) uniform texture1D rdv_images_1d[1024*16];
layout(set=0, binding=BINDING_SLOT(1)) uniform texture2D rdv_images_2d[1024*16];
layout(set=0, binding=BINDING_SLOT(2)) uniform texture3D rdv_images_3d[1024*16];
layout(set=0, binding=BINDING_SLOT(3)) uniform sampler rdv_samplers[1024*16];


vec4 sample_image(int image_index, int sampler_index, float c) {
    return texture(
        sampler1D(
            rdv_images_1d[image_index],
            rdv_samplers[sampler_index]), c);
}

vec4 sample_image(int image_index, int sampler_index, vec2 c) {
    return texture(
        sampler2D(
            rdv_images_2d[image_index],
            rdv_samplers[sampler_index]), c);
}

vec4 sample_image(int image_index, int sampler_index, vec3 c) {
    return texture(
        sampler3D(
            rdv_images_3d[image_index],
            rdv_samplers[sampler_index]), c);
}

void RDV_NODE(uvec3 idx) { }
