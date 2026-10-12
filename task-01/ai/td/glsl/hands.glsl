// Hand skeleton helpers (#include <glsl_hands>), used by view_gesture.frag.
// The including shader must #define LANDMARK_INPUT: the CHOP-to-TOP landmarks texture (21 x 6 texels, r = value,
// rows h0tx h0ty h0tz h1tx h1ty h1tz; normalised image coords with a bottom-left origin).
const int BONES[42] = int[42](0,1, 1,2, 2,3, 3,4, 0,5, 5,6, 6,7, 7,8, 5,9, 9,10, 10,11, 11,12,
	9,13, 13,14, 14,15, 15,16, 13,17, 17,18, 18,19, 19,20, 0,17);

vec2 LandmarkUV(int hand, int index) {
	return vec2(texelFetch(sTD2DInputs[LANDMARK_INPUT], ivec2(index, hand * 3 + 0), 0).r,
		texelFetch(sTD2DInputs[LANDMARK_INPUT], ivec2(index, hand * 3 + 1), 0).r);
}

// Distances in pixels from px to the hand's bones / joints, with the camera image mapped to [origin, origin+size].
float BoneDistance(int hand, vec2 px, vec2 origin, vec2 size) {
	float d = 1e9;
	for (int b = 0; b < 21; b++) {
		d = min(d, SegmentDistance(px, origin + LandmarkUV(hand, BONES[b * 2]) * size,
			origin + LandmarkUV(hand, BONES[b * 2 + 1]) * size));
	}
	return d;
}

float JointDistance(int hand, vec2 px, vec2 origin, vec2 size) {
	float d = 1e9;
	for (int i = 0; i < 21; i++) {
		d = min(d, length(px - (origin + LandmarkUV(hand, i) * size)));
	}
	return d;
}
