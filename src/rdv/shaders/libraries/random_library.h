layout(set=0, binding=BINDING_SLOT(0), scalar) uniform RandomSeedsBlock {
    uvec4 seeds;
} random_seeds;

// adapted from NVidia
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
uint TausStep(uint z, int S1, int S2, int S3, uint M) { uint b = (((z << S1) ^ z) >> S2); return ((z & M) << S3) ^ b; }
uint LCGStep(uint z, uint A, uint C) { return A * z + C; }

uvec4 rdv_rng_state;

void random_step(inout uvec4 rng_state)
{
	rng_state.x = TausStep(rng_state.x, 13, 19, 12, 4294967294U);
	rng_state.y = TausStep(rng_state.y, 2, 25, 4, 4294967288U);
	rng_state.z = TausStep(rng_state.z, 3, 11, 17, 4294967280U);
	rng_state.w = LCGStep(rng_state.w, 1664525, 1013904223U);
}

void random_step(){
    random_step(rdv_rng_state);
}

uvec4 random_seed()
{
    return rdv_rng_state;
}

float random(inout uvec4 rng_state)
{
    random_step(rng_state);
	uint v = rng_state.x ^ rng_state.y ^ rng_state.z ^ rng_state.w;
	// more robust computation
	return uintBitsToFloat(v & 0x007FFFFF | 0x3F800000) - 1;
	// this has an error and produces randoms = 1
    // f = 2.3283064364387e-10 * uint(rdv_rng_state.x ^ rdv_rng_state.y ^ rdv_rng_state.z ^ rdv_rng_state.w); // THERE WAS AN ERROR HERE!
}

float random()
{
    return random(rdv_rng_state);
}

uvec4 random_spawn(uvec4 rng_state, int index)
{
    rng_state = rng_state ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (rng_state.x + 13 * rng_state.y));
    // uvec4 state = rng_state ^ uvec4(0x23F1 * index + index*(~index),0x3137 % index, index ^ index, index + 129);
    random_step(rng_state);
    return rng_state;
}

uvec4 random_seed(uvec4 rng_state)
{
    uvec4 old_state = rdv_rng_state;
    rdv_rng_state = rng_state;
    return old_state;
}

uvec4 random_branch()
{
    uvec4 old_rng = rdv_rng_state;
    rdv_rng_state = floatBitsToUint(vec4(random(), random(), random(), random())) + 129;
    random_step();
    return old_rng;
}

vec3 random_direction(vec3 D) {
	float r1 = random();
	float r2 = random() * 2 - 1;
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	create_ortho_basis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction()
{
    return random_direction(vec3(0.0, 1.0, 0.0));
}

vec3 random_direction(vec3 D, float fov) {
	float r1 = random();
	float r2 = 1 - random() * (1 - cos(fov));
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	create_ortho_basis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction(float alpha0, float alpha1, float beta0, float beta1) {
	float r1 = random() * (alpha1 - alpha0) + alpha0;
	float r2 = sin(beta0) + random() * (sin(beta1) - sin(beta0));
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0 = vec3(1, 0, 0);
	vec3 t1 = vec3(0, 0, 1);
	vec3 D = vec3(0, 1, 0);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction_HS(vec3 D) {
	float r1 = random();
	float r2 = random();
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	create_ortho_basis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction_HS_cosine_weighted(vec3 N, out float NdotD)
{
	vec3 t0, t1;
	create_ortho_basis(N, t0, t1);

	// Safety-capped rejection sampling (was `while (true)`) - expected
	// accept rate here is pi/4 (~78.5%), so 64 tries failing is
	// astronomically unlikely; caps worst-case iteration count instead of
	// letting a GPU thread spin indefinitely (see pbvrt.h's own note on
	// why an unbounded loop here is a real TDR/crash risk, not just a
	// theoretical one).
	for (int safety = 0; safety < 64; safety++) {
		float x = random() * 2 - 1;
		float y = random() * 2 - 1;
		float d2 = x * x + y * y;
		if (d2 > 0.001 && d2 < 1)
		{
			float z = sqrt(1 - d2);
			NdotD = z;
			return t0 * x + t1 * y + N * z;
		}
	}
	// deterministic fallback sample (x=0.5,y=0 always satisfies 0.001<d2<1) - never reached in practice
	NdotD = sqrt(0.75);
	return t0 * 0.5 + N * sqrt(0.75);
}

vec3 random_direction_HS_cosine_weighted()
{
	// Safety-capped rejection sampling (was `while (true)`) - see the other
	// overload's own note.
	for (int safety = 0; safety < 64; safety++) {
		float x = random() * 2 - 1;
		float y = random() * 2 - 1;
		float d2 = x * x + y * y;
		if (d2 > 0.001 && d2 < 1)
		{
			float z = sqrt(1 - d2);
			return vec3(x, y, z);
		}
	}
	return vec3(0.5, 0.0, sqrt(0.75)); // deterministic fallback sample - never reached in practice
}



vec2 rdv_BM() {
	float u1 = 1.0 - random(); //uniform(0,1] random doubles
	float u2 = 1.0 - random();
	float r = sqrt(-2.0 * log(max(0.0000000001, u1)));
	float t = 2.0 * pi * u2;
	return r * vec2(cos(t), sin(t));
}

float random_normal() {
	return rdv_BM().x;
}

vec2 random_normal_2()
{
    return rdv_BM();
}

vec3 random_normal_3()
{
    return vec3(rdv_BM(), rdv_BM().x);
}

vec4 random_normal_4()
{
    return vec4(rdv_BM(), rdv_BM());
}

float random_normal(float mu, float sd)
{
    return sd * random_normal() + mu;
}

vec2 random_normal(vec2 mu, vec2 sd)
{
    return sd * random_normal_2() + mu;
}

vec3 random_normal(vec3 mu, vec3 sd)
{
    return sd * random_normal_3() + mu;
}

vec4 random_normal(vec4 mu, vec4 sd)
{
    return sd * random_normal_4() + mu;
}



vec3 hg_phase_sample(vec3 w_in, float g) {
    float phi = random() * 2 * pi;
    float xi = random();
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    float one_over_2g = 0.5 / g;

    float t = one_minus_g2 / (1.0 - g + 2.0 * g * xi);
    float invertcdf = one_over_2g * (one_plus_g2 - t * t);
    float cosTheta = abs(g) < 0.001 ? 2 * xi - 1 : invertcdf;
    float sinTheta = sqrt(max(0, 1.0 - cosTheta * cosTheta));
    vec3 t0, t1;
    create_ortho_basis(w_in, t0, t1);
    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * w_in;
}

float hg_phase_eval(float cos_theta, float g)
{
    if (abs(g) < 0.001)
        return 0.25 / pi;
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    return 0.25 / pi * (one_minus_g2) / pow(one_plus_g2 - 2 * g * cos_theta, 1.5);
}

float hg_phase_eval(vec3 w_in, vec3 w_out, float g)
{
    return hg_phase_eval(dot(w_in, w_out), g);
}

void RDV_NODE(uvec3 idx) {
    // rdv_dim_x/y (the dispatcher uniform's own dims) aren't declared yet at
    // this point in the assembled shader (this library's own init runs
    // before the rdv_dispatcher block) - a fixed-radix mix of idx alone is
    // enough for a unique per-thread seed.
    int index = int(idx.x + idx.y * 65521u + idx.z * 2097143u);
    rdv_rng_state = random_seeds.seeds;
    // random_spawn() takes rng_state by value and returns the spawned
    // state - its result was previously discarded here, leaving every
    // thread's rdv_rng_state identical (just random_seeds.seeds, with no
    // per-thread differentiation at all).
    rdv_rng_state = random_spawn(rdv_rng_state, index);
}