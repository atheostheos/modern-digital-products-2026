#include <glsl_common>
// Top-5 image classes as horizontal bars (labels are drawn by a Text TOP over this).
uniform vec4 uScores;  // scores of ranks 1-4 (lagged)
uniform float uScore5;
out vec4 fragColor;

const float ROW_TOP = 96.0;   // px from the top to the first bar
const float ROW_PITCH = 72.0;
const float BAR_H = 14.0;
const float X0 = 32.0;
const float X1 = 608.0;

void main() {
	vec2 p = PixelTopLeft();
	vec3 color = BG;
	float scores[5] = float[5](uScores.x, uScores.y, uScores.z, uScores.w, uScore5);
	for (int i = 0; i < 5; i++) {
		float y = ROW_TOP + float(i) * ROW_PITCH + 34.0;  // bar sits under its label line
		float track = Box(p, vec2(X0, y), vec2(X1, y + BAR_H), 1.0);
		float fill = Box(p, vec2(X0, y), vec2(X0 + clamp(scores[i], 0.0, 1.0) * (X1 - X0), y + BAR_H), 1.0);
		vec3 accent = mix(WARM, COLD, float(i) / 4.0);
		color = mix(color, PANEL, track);
		color = mix(color, accent, fill);
	}
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
