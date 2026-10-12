"""ml_gesture: streams the input TOP to the server's MediaPipe gesture node.

Outputs: script_landmarks (21 samples, h0tx h0ty h0tz h1tx ...; y flipped to TD's bottom-left origin),
script_gestures (hands, h0_score, h1_score and one channel per gesture = best score), table_hands.
"""

from MlClientExt import MlClientExt, isNumber

GESTURES = ['None', 'Closed_Fist', 'Open_Palm', 'Pointing_Up', 'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou']
MAX_HANDS = 2
NUM_LANDMARKS = 21


def isValidHand(hand):
	"""A hand the CHOPs can draw: 21 landmarks of 3 numbers each, a gesture name and a numeric score."""
	if not isinstance(hand, dict) or not isNumber(hand.get('gesture_score')):
		return False
	points = hand.get('landmarks')
	return (isinstance(points, list) and len(points) == NUM_LANDMARKS
		and all(isinstance(p, list) and len(p) == 3 and all(isNumber(v) for v in p) for p in points))


class GestureExt(MlClientExt):
	NODE = 'gesture'

	def __init__(self, ownerComp):
		self.Hands = []
		super().__init__(ownerComp)

	def Tick(self, frame):
		if self.ready():
			self.SendFrame(self.ownerComp.op('fit_send'))

	def OnResult(self, data, reply):
		hands = data.get('hands')
		hands = [hand for hand in hands if isValidHand(hand)] if isinstance(hands, list) else []
		self.Hands = hands[:MAX_HANDS]
		table = self.ownerComp.op('table_hands')
		table.clear()
		table.appendRow(['hand', 'handedness', 'handedness_score', 'gesture', 'gesture_score'])
		for i, hand in enumerate(self.Hands):
			table.appendRow([i] + [hand.get(key, '') for key in ('handedness', 'handedness_score', 'gesture', 'gesture_score')])
		self.ownerComp.op('script_landmarks').cook(force=True)
		self.ownerComp.op('script_gestures').cook(force=True)

	def fill_landmarks(self, scriptOp):
		scriptOp.numSamples = NUM_LANDMARKS
		for h in range(MAX_HANDS):
			points = self.Hands[h]['landmarks'] if h < len(self.Hands) else [[0.0, 0.0, 0.0]] * NUM_LANDMARKS
			tx, ty, tz = (scriptOp.appendChan('h{}t{}'.format(h, axis)) for axis in 'xyz')
			tx.vals = [p[0] for p in points]
			ty.vals = [1.0 - p[1] if h < len(self.Hands) else 0.0 for p in points]
			tz.vals = [p[2] for p in points]

	def fill_gestures(self, scriptOp):
		scriptOp.numSamples = 1
		scriptOp.appendChan('hands')[0] = len(self.Hands)
		for h in range(MAX_HANDS):
			score = self.Hands[h]['gesture_score'] if h < len(self.Hands) else 0.0
			scriptOp.appendChan('h{}_score'.format(h))[0] = score
		for name in GESTURES:
			best = max((hand['gesture_score'] for hand in self.Hands if hand.get('gesture') == name), default=0.0)
			scriptOp.appendChan(name)[0] = best
