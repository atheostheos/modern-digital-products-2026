#include <glsl_common>
// Mood wash, valence history and gauge.
// Input 0: valence trail (CHOP to TOP, N x 1, r = valence in -1..1, oldest at the left).
uniform float uValence;
out vec4 fragColor;

const vec2 PLOT_LO = vec2(32.0, 96.0);   // top-left origin
const vec2 PLOT_HI = vec2(608.0, 352.0);
const float GAUGE_Y = 400.0;

void main() {
	vec2 p = PixelTopLeft();
	vec2 res = uTDOutputInfo.res.zw;
	vec3 mood = MoodColor(uValence);
	vec3 color = mix(BG, mood * 0.35, 0.25 + 0.55 * (1.0 - vUV.t));  // wash, stronger at the bottom

	vec2 plotSize = PLOT_HI - PLOT_LO;
	float u = clamp((p.x - PLOT_LO.x) / plotSize.x, 0.0, 1.0);
	float midY = PLOT_LO.y + plotSize.y * 0.5;
	float value = texture(sTD2DInputs[0], vec2(u, 0.5)).r;
	float lineY = midY - value * plotSize.y * 0.45;
	float inPlot = Box(p, PLOT_LO, PLOT_HI, 1.0);
	color = mix(color, TEXT_DIM, inPlot * (1.0 - smoothstep(0.5, 1.5, abs(p.y - midY))) * 0.5);
	color = mix(color, MoodColor(value), inPlot * (1.0 - smoothstep(1.5, 3.0, abs(p.y - lineY))));

	float gx0 = PLOT_LO.x, gx1 = PLOT_HI.x, gc = (gx0 + gx1) * 0.5;
	float valueX = gc + uValence * (gx1 - gc);
	color = mix(color, PANEL, Box(p, vec2(gx0, GAUGE_Y), vec2(gx1, GAUGE_Y + 16.0), 1.0));
	color = mix(color, mood, Box(p, vec2(min(gc, valueX), GAUGE_Y), vec2(max(gc, valueX), GAUGE_Y + 16.0), 1.0));
	color = mix(color, vec3(1.0), Box(p, vec2(gc - 1.0, GAUGE_Y - 6.0), vec2(gc + 1.0, GAUGE_Y + 22.0), 0.5));
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
