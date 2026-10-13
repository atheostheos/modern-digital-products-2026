// Six generative worlds + emoji sprite/burst (#include <glsl_worlds>), used by view_llm.frag.
// The including shader must #define EMOJI_INPUT (index of the emoji texture input) first.
// World order matches the server: storm, sunrise, fireworks, ocean, neon, starfield.

const float TAU = 6.2831853;
const int BURST_COPIES = 14;

float Hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float GridLine(float v) {  // anti-aliased line at every integer of v
	float d = min(fract(v), 1.0 - fract(v));
	return 1.0 - smoothstep(0.0, fwidth(v) * 1.5, d);
}

vec3 Storm(vec2 uv, float t) {
	vec3 c = mix(vec3(0.04, 0.05, 0.08), vec3(0.16, 0.20, 0.26), uv.y);
	vec2 r = vec2(uv.x * 90.0 + uv.y * 18.0, uv.y * 4.0 + t * 7.0);
	vec2 cell = floor(r), f = fract(r);
	float drop = step(0.88, Hash(cell)) * smoothstep(0.0, 0.1, f.y) * (1.0 - smoothstep(0.1, 0.45, f.y))
		* (1.0 - smoothstep(0.03, 0.09, abs(f.x - 0.5)));
	float flash = step(0.97, Hash(vec2(floor(t * 5.0), 3.0))) * (1.0 - fract(t * 5.0));
	return c + drop * vec3(0.45, 0.5, 0.6) + flash * vec3(0.55, 0.6, 0.8);
}

vec3 Sunrise(vec2 uv, float t, float aspect) {
	vec3 sky = mix(vec3(1.0, 0.78, 0.45), vec3(0.95, 0.5, 0.35), smoothstep(0.1, 0.45, uv.y));
	sky = mix(sky, vec3(0.22, 0.25, 0.55), smoothstep(0.45, 1.0, uv.y));
	vec2 d = (uv - vec2(0.5, 0.2 + 0.06 * sin(t * 0.3))) * vec2(aspect, 1.0);
	float sun = 1.0 - smoothstep(0.095, 0.1, length(d));
	vec3 c = sky + exp(-length(d) * 5.0) * vec3(1.0, 0.6, 0.3) * 0.5 + sun * vec3(1.0, 0.95, 0.8);
	return mix(vec3(0.12, 0.08, 0.1), c, smoothstep(0.1, 0.115, uv.y));
}

vec3 Fireworks(vec2 uv, float t, float aspect) {
	vec3 c = mix(vec3(0.02, 0.02, 0.06), vec3(0.07, 0.04, 0.13), uv.y);
	for (int i = 0; i < 4; i++) {
		float fi = float(i), period = 2.2 + fi * 0.37;
		float local = mod(t + fi * 0.7, period), seed = floor((t + fi * 0.7) / period) + fi * 13.0;
		vec2 center = vec2(0.2 + 0.6 * Hash(vec2(seed, 1.0)), 0.45 + 0.35 * Hash(vec2(seed, 2.0)));
		vec3 col = 0.55 + 0.45 * cos(TAU * (Hash(vec2(seed, 3.0)) + vec3(0.0, 0.33, 0.67)));
		vec2 d = (uv - center) * vec2(aspect, 1.0);
		float ring = exp(-abs(length(d) - local * 0.17) * 55.0);
		float sparks = pow(abs(cos(atan(d.y, d.x) * 12.0)), 24.0);
		c += col * ring * (0.25 + sparks) * (1.0 - smoothstep(0.55, 1.0, local / period)) * 1.4;
	}
	return c;
}

vec3 Ocean(vec2 uv, float t) {
	vec3 c = mix(vec3(0.78, 0.9, 0.98), vec3(0.3, 0.55, 0.9), uv.y);
	for (int i = 0; i < 4; i++) {
		float fi = float(i);
		float y = 0.5 - fi * 0.11 + 0.02 * sin(uv.x * (8.0 + fi * 3.0) + t * (1.0 + fi * 0.4) + fi);
		float below = step(uv.y, y);
		c = mix(c, mix(vec3(0.05, 0.38, 0.58), vec3(0.02, 0.14, 0.34), fi / 3.0), below);
		c += below * exp(-abs(uv.y - y) * 120.0) * 0.6;
	}
	return c;
}

vec3 Neon(vec2 uv, float t, float aspect) {
	const float HORIZON = 0.45;
	vec3 c = mix(vec3(0.02, 0.0, 0.05), vec3(0.12, 0.0, 0.2), smoothstep(HORIZON, 1.0, uv.y));
	vec2 sd = vec2((uv.x - 0.5) * aspect, uv.y - 0.6);
	float sun = (1.0 - smoothstep(0.17, 0.175, length(sd))) * step(HORIZON, uv.y);
	float stripes = max(step(0.45, fract(uv.y * 28.0)), step(0.66, uv.y));
	c += sun * stripes * mix(vec3(1.0, 0.15, 0.55), vec3(1.0, 0.8, 0.2), clamp((uv.y - 0.45) / 0.35, 0.0, 1.0));
	if (uv.y < HORIZON) {
		float depth = HORIZON - uv.y + 0.002;
		float vx = (uv.x - 0.5) / depth * 1.2;  // ~3 lines at the bottom, many towards the horizon
		float vz = 0.6 / depth + t * 1.5;
		float line = max(GridLine(vx), GridLine(vz));
		c += line * mix(vec3(0.0, 0.9, 1.0), vec3(1.0, 0.2, 0.8), uv.y / HORIZON) * smoothstep(0.0, 0.05, depth);
	}
	return c + exp(-abs(uv.y - HORIZON) * 40.0) * vec3(1.0, 0.25, 0.7) * 0.5;
}

vec3 Starfield(vec2 uv, float t, float aspect) {
	vec3 c = vec3(0.0, 0.0, 0.025);
	for (int l = 0; l < 3; l++) {
		float fl = float(l), scale = 40.0 + 30.0 * fl;
		vec2 q = uv * vec2(aspect, 1.0) * scale + vec2(t * (0.3 + 0.4 * fl), 0.0);
		vec2 f = fract(q) - 0.5;
		float h = Hash(floor(q) + fl * 7.0);
		float core = 1.0 - smoothstep(0.0, 0.12 + 0.06 * fl, length(f));
		float glow = exp(-length(f) * 9.0) * 0.5;
		float star = step(0.88, h) * (core + glow);
		c += star * (0.65 + 0.35 * sin(t * 3.0 + h * 40.0)) * mix(vec3(0.75, 0.82, 1.0), vec3(1.0, 0.9, 0.75), h);
	}
	return c;
}

vec3 Sprite(vec3 c, vec2 uv, vec2 center, float size, float aspect, float alpha) {
	vec2 d = (uv - center) * vec2(aspect, 1.0) / max(size, 1e-4) + 0.5;
	if (any(lessThan(d, vec2(0.0))) || any(greaterThan(d, vec2(1.0)))) return c;
	vec4 s = texture(sTD2DInputs[EMOJI_INPUT], d) * alpha;  // premultiplied
	return c * (1.0 - s.a) + s.rgb;
}

vec3 Burst(vec3 c, vec2 uv, float aspect, float b) {  // b = seconds since the burst started
	if (b < 0.0 || b > 4.0) return c;
	float flyAlpha = 1.0 - smoothstep(0.6, 1.6, b);
	for (int i = 0; i < BURST_COPIES; i++) {
		float angle = float(i) * 2.39996, speed = 0.25 + 0.2 * Hash(vec2(float(i), 9.0));
		vec2 pos = vec2(0.5, 0.55) + vec2(cos(angle) / aspect, sin(angle)) * speed * b * 0.9;
		c = Sprite(c, uv, pos, 0.08, aspect, flyAlpha);
	}
	float pop = smoothstep(0.0, 0.25, b) * (1.0 + 0.3 * exp(-b * 6.0));
	return Sprite(c, uv, vec2(0.5, 0.55), 0.32 * pop, aspect, 1.0 - smoothstep(2.5, 4.0, b));
}

vec3 WorldsBlend(vec2 uv, float t, float aspect, float w[6]) {
	float total = max(w[0] + w[1] + w[2] + w[3] + w[4] + w[5], 1e-3);
	vec3 c = vec3(0.0);
	if (w[0] > 0.002) c += w[0] * Storm(uv, t);
	if (w[1] > 0.002) c += w[1] * Sunrise(uv, t, aspect);
	if (w[2] > 0.002) c += w[2] * Fireworks(uv, t, aspect);
	if (w[3] > 0.002) c += w[3] * Ocean(uv, t);
	if (w[4] > 0.002) c += w[4] * Neon(uv, t, aspect);
	if (w[5] > 0.002) c += w[5] * Starfield(uv, t, aspect);
	return c / total;
}
