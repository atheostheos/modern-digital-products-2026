#include <glsl_common>
// Server latency history per node on a log scale (1-1000 ms); the table is a Text TOP over this.
// Input 0: trail of <node>_p50_ms (CHOP to TOP, N x 5 rows: gesture classify speech sentiment llm).
out vec4 fragColor;

const vec2 PLOT_LO = vec2(32.0, 268.0);  // top-left origin
const vec2 PLOT_HI = vec2(608.0, 448.0);
const vec3 NODE_COLORS[5] = vec3[5](vec3(0.15, 0.95, 0.90), vec3(1.0, 0.58, 0.18), vec3(1.0, 0.30, 0.80),
	vec3(0.55, 0.85, 0.30), vec3(0.65, 0.55, 1.0));

float LogY(float ms) {  // 1 ms -> bottom, 1000 ms -> top
	float t = clamp(log(max(ms, 1.0)) / log(1000.0), 0.0, 1.0);
	return mix(PLOT_HI.y, PLOT_LO.y, t);
}

void main() {
	vec2 p = PixelTopLeft();
	vec3 color = BG;
	float inPlot = Box(p, PLOT_LO, PLOT_HI, 1.0);
	color = mix(color, PANEL, inPlot);
	for (int g = 1; g < 3; g++) {  // grid at 10 ms and 100 ms
		float gy = LogY(pow(10.0, float(g)));
		color = mix(color, TEXT_DIM, inPlot * 0.35 * (1.0 - smoothstep(0.5, 1.2, abs(p.y - gy))));
	}
	float u = clamp((p.x - PLOT_LO.x) / (PLOT_HI.x - PLOT_LO.x), 0.0, 1.0);
	for (int n = 0; n < 5; n++) {
		float ms = texture(sTD2DInputs[0], vec2(u, (float(n) + 0.5) / 5.0)).r;
		float line = 1.0 - smoothstep(1.0, 2.5, abs(p.y - LogY(ms)));
		color = mix(color, NODE_COLORS[n], inPlot * line * step(0.5, ms));
	}
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
