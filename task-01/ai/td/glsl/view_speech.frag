#include <glsl_common>
// Caption panel with a "speaking" indicator; transcript text is drawn by Text TOPs over this.
uniform float uSpeaking;  // 0/1 from the speech node
uniform float uPulse;     // 0..1 from an LFO CHOP
out vec4 fragColor;

void main() {
	vec2 p = PixelTopLeft();
	vec2 res = uTDOutputInfo.res.zw;
	vec3 color = BG;
	color = mix(color, PANEL, Box(p, vec2(16.0, 64.0), vec2(res.x - 16.0, res.y - 72.0), 1.0));
	float dot = 1.0 - smoothstep(9.0, 11.0, length(p - vec2(res.x - 36.0, 32.0)));
	vec3 live = mix(vec3(0.9, 0.15, 0.2), vec3(1.0, 0.45, 0.45), uPulse);
	color = mix(color, uSpeaking > 0.5 ? live : TEXT_DIM * 0.6, dot);
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
