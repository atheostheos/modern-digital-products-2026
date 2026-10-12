#include <glsl_common>
// Mood-tinted backdrop for the narrator's caption (text drawn by Text TOPs over this).
uniform float uValence;
out vec4 fragColor;

void main() {
	vec2 res = uTDOutputInfo.res.zw;
	vec2 centered = (vUV.st - 0.5) * vec2(res.x / res.y, 1.0);
	float vignette = smoothstep(1.1, 0.2, length(centered));
	vec3 mood = MoodColor(uValence);
	vec3 color = mix(BG, mood * 0.55, 0.35 * vignette);
	vec2 p = PixelTopLeft();
	color = mix(color, mood, Box(p, vec2(24.0, res.y - 40.0), vec2(res.x - 24.0, res.y - 36.0), 1.0));
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
