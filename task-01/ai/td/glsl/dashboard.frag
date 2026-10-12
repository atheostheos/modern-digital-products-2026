#include <glsl_common>
// 3x2 grid of the six 640x480 viewers, scaled into 1280x640 (TD Non-Commercial caps textures at 1280).
// Inputs: 0 gesture, 1 classify, 2 speech (top row), 3 sentiment, 4 llm, 5 metrics (bottom row).
out vec4 fragColor;

const vec2 GRID = vec2(3.0, 2.0);
const float GAP_PX = 2.0;

vec4 Cell(int index, vec2 uv) {
	switch (index) {  // constant sampler indices (no dynamic indexing needed)
		case 0: return texture(sTD2DInputs[0], uv);
		case 1: return texture(sTD2DInputs[1], uv);
		case 2: return texture(sTD2DInputs[2], uv);
		case 3: return texture(sTD2DInputs[3], uv);
		case 4: return texture(sTD2DInputs[4], uv);
		default: return texture(sTD2DInputs[5], uv);
	}
}

void main() {
	vec2 scaled = vUV.st * GRID;
	ivec2 cell = ivec2(floor(scaled));
	vec2 uv = fract(scaled);
	int index = (int(GRID.y) - 1 - cell.y) * int(GRID.x) + cell.x;  // row 0 at the top
	vec4 color = Cell(index, uv);
	vec2 cellPx = uv * uTDOutputInfo.res.zw / GRID;
	vec2 cellSize = uTDOutputInfo.res.zw / GRID;
	float edge = min(min(cellPx.x, cellSize.x - cellPx.x), min(cellPx.y, cellSize.y - cellPx.y));
	color.rgb = mix(BG * 0.5, color.rgb, step(GAP_PX * 0.5, edge));
	fragColor = TDOutputSwizzle(vec4(color.rgb, 1.0));
}
