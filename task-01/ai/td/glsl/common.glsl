// Shared palette and helpers for the view_* shaders (pasted at the top of each shader by the builder).
const vec3 BG = vec3(0.055, 0.063, 0.078);
const vec3 PANEL = vec3(0.095, 0.105, 0.125);
const vec3 TEXT_DIM = vec3(0.45, 0.48, 0.53);
const vec3 COLD = vec3(0.22, 0.48, 1.00);  // negative
const vec3 WARM = vec3(1.00, 0.58, 0.18);  // positive
const vec3 HAND_COLORS[2] = vec3[2](vec3(0.15, 0.95, 0.90), vec3(1.00, 0.30, 0.80));

vec2 PixelTopLeft() {  // pixel coords with origin top-left (TD's UV origin is bottom-left)
	vec2 res = uTDOutputInfo.res.zw;
	return vec2(vUV.s * res.x, (1.0 - vUV.t) * res.y);
}

float Box(vec2 p, vec2 lo, vec2 hi, float aa) {  // 1 inside the axis-aligned box, soft edge of `aa` px
	vec2 d = max(lo - p, p - hi);
	return 1.0 - smoothstep(0.0, aa, max(d.x, d.y));
}

float SegmentDistance(vec2 p, vec2 a, vec2 b) {
	vec2 pa = p - a, ba = b - a;
	float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
	return length(pa - ba * h);
}

vec3 MoodColor(float valence) {  // -1 cold blue, 0 neutral grey, +1 warm amber
	vec3 neutral = vec3(0.55, 0.55, 0.60);
	return valence < 0.0 ? mix(neutral, COLD, -valence) : mix(neutral, WARM, valence);
}
