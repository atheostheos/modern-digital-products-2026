#define LANDMARK_INPUT 1
#include <glsl_common>
#include <glsl_hands>
// Hand skeleton over the input video. Input 0: video (640x480). Input 1: landmarks texture (see hands.glsl).
uniform float uHands;
out vec4 fragColor;

const float BONE_PX = 2.5;
const float JOINT_PX = 4.5;

void main() {
	vec2 res = uTDOutputInfo.res.zw;
	vec2 px = vUV.st * res;
	vec3 color = texture(sTD2DInputs[0], vUV.st).rgb * 0.7;
	for (int hand = 0; hand < min(int(uHands + 0.5), 2); hand++) {
		float bone = BoneDistance(hand, px, vec2(0.0), res);
		float joint = JointDistance(hand, px, vec2(0.0), res);
		color = mix(color, HAND_COLORS[hand], 1.0 - smoothstep(BONE_PX - 1.0, BONE_PX + 1.0, bone));
		color = mix(color, vec3(1.0), 1.0 - smoothstep(JOINT_PX - 1.0, JOINT_PX + 1.0, joint));
	}
	fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
