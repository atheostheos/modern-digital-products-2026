#define EMOJI_INPUT 0
#include <glsl_common>
#include <glsl_worlds>
// Scene director view: six generative worlds blended by the LLM's (lagged) choice + an emoji burst.
// Input 0: emoji picture (premultiplied RGBA). Weights: uW1 = storm, sunrise, fireworks; uW2 = ocean, neon,
// starfield. uBurst = seconds since the last direction (negative/large = no burst).
uniform vec3 uW1;
uniform vec3 uW2;
uniform float uTime;
uniform float uValence;
uniform float uBurst;
out vec4 fragColor;

void main() {
	vec2 uv = vUV.st;
	float aspect = uTDOutputInfo.res.z / uTDOutputInfo.res.w;
	float w[6] = float[6](uW1.x, uW1.y, uW1.z, uW2.x, uW2.y, uW2.z);
	vec3 c = WorldsBlend(uv, uTime, aspect, w);
	c = mix(c, c * MoodColor(uValence) * 1.6, 0.12);
	c = Burst(c, uv, aspect, uBurst);
	fragColor = TDOutputSwizzle(vec4(c, 1.0));
}
