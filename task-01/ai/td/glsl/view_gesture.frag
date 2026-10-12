#include <glsl_common>
// Hand skeleton over the input video.
// Input 0: video (640x480). Input 1: landmarks texture from CHOP to TOP, 21 x 6 texels, r = value,
// rows = h0tx h0ty h0tz h1tx h1ty h1tz (normalised image coords, y already flipped to bottom-left).
uniform float uHands;
out vec4 fragColor;

const int BONES[42] = int[42](0,1, 1,2, 2,3, 3,4, 0,5, 5,6, 6,7, 7,8, 5,9, 9,10, 10,11, 11,12,
	9,13, 13,14, 14,15, 15,16, 13,17, 17,18, 18,19, 19,20, 0,17);
const float BONE_PX = 2.5;
const float JOINT_PX = 4.5;

vec2 Landmark(int hand, int index, vec2 res) {
	float x = texelFetch(sTD2DInputs[1], ivec2(index, hand * 3 + 0), 0).r;
	float y = texelFetch(sTD2DInputs[1], ivec2(index, hand * 3 + 1), 0).r;
	return vec2(x, y) * res;
}

void main() {
	vec2 res = uTDOutputInfo.res.zw;
	vec2 px = vUV.st * res;
	vec3 color = texture(sTD2DInputs[0], vUV.st).rgb * 0.7;
	for (int hand = 0; hand < min(int(uHands + 0.5), 2); hand++) {
		float bone = 1e9;
		for (int b = 0; b < 21; b++) {
			bone = min(bone, SegmentDistance(px, Landmark(hand, BONES[b * 2], res), Landmark(hand, BONES[b * 2 + 1], res)));
		}
		float joint = 1e9;
		for (int i = 0; i < 21; i++) {
			joint = min(joint, length(px - Landmark(hand, i, res)));
		}
		vec3 tint = HAND_COLORS[hand];
		color = mix(color, tint, 1.0 - smoothstep(BONE_PX - 1.0, BONE_PX + 1.0, bone));
		color = mix(color, vec3(1.0), 1.0 - smoothstep(JOINT_PX - 1.0, JOINT_PX + 1.0, joint));
	}
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
