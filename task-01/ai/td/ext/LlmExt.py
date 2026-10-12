"""ml_llm: scene director. After each new finished sentence (from the Speech node on the Context page, or a
wired in_scene text) it asks the server's gemma3:1b to pick one of six visual worlds and one emoji.

Outputs: text_scene (chosen world), text_symbol (the emoji character), text_context (what was sent), script_emoji (emoji picture,
RGBA, from the server's PNG), script_llm (scene index, one-hot scene_<name> channels, directed_frame = the
frame the last direction arrived, tokens_per_s, latency_ms, directions).
"""

import base64
import json
import time

import numpy as np

from MlClientExt import MlClientExt

SCENES = ['storm', 'sunrise', 'fireworks', 'ocean', 'neon', 'starfield']


class LlmExt(MlClientExt):
	NODE = 'llm'

	def __init__(self, ownerComp):
		self.lastSpeech = None
		self.lastRequest = 0.0
		self.emoji = np.zeros((1, 1, 4), dtype=np.float32)
		self.Stats = {'scene': 5.0, 'directed_frame': -1e9, 'tokens_per_s': 0.0, 'latency_ms': 0.0, 'directions': 0}
		super().__init__(ownerComp)

	# ------------------------------------------------------------ input
	def wiredScene(self):
		"""Text arriving on in_scene (a COMP-level wire; In DATs do not list it in .inputs, so test the text)."""
		dat = self.ownerComp.op('in_scene')
		return dat.text.strip() if dat is not None else ''

	def contextScene(self):
		par = self.ownerComp.par
		scene = {'speech': '', 'gesture': 'None', 'valence': 0.0}
		speech = par.Speech.eval()
		if speech is not None:
			scene['speech'] = speech.op('text_final').text.strip()
		gesture = par.Gesture.eval()
		if gesture is not None:
			hands = gesture.op('table_hands')
			scene['gesture'] = hands[1, 'gesture'].val if hands is not None and hands.numRows > 1 else 'None'
		sentiment = par.Sentiment.eval()
		if sentiment is not None:
			chop = sentiment.op('script_sentiment')
			scene['valence'] = round(chop['valence'][0], 3) if chop is not None and chop.numChans else 0.0
		return scene

	def Tick(self, frame):
		"""React to each new sentence (at most once per Interval seconds)."""
		par = self.ownerComp.par
		if not par.Auto.eval() or time.perf_counter() - self.lastRequest < float(par.Interval.eval()):
			return
		wired = self.wiredScene()
		speech = wired or self.contextScene()['speech']
		if speech and speech != self.lastSpeech:
			self.SendNow()

	def SendNow(self):
		wired = self.wiredScene()
		scene = self.contextScene()
		payload = wired or json.dumps(scene)
		if self.SendText(payload):
			self.lastRequest = time.perf_counter()
			self.lastSpeech = wired or scene['speech']
			self.ownerComp.op('text_context').text = payload

	# ------------------------------------------------------------ output
	def OnResult(self, data, reply):
		scene = data.get('scene', 'starfield')
		self.ownerComp.op('text_scene').text = scene
		self.ownerComp.op('text_symbol').text = data.get('emoji', '')
		self.Stats = {
			'scene': float(SCENES.index(scene)) if scene in SCENES else 5.0,
			'directed_frame': float(absTime.frame),
			'tokens_per_s': data.get('tokens_per_s') or 0.0,
			'latency_ms': self.Status['rtt_ms'],
			'directions': self.Stats['directions'] + 1,
		}
		self.emoji = self.decodeEmoji(data.get('emoji_png'))
		self.ownerComp.op('script_llm').cook(force=True)
		self.ownerComp.op('script_emoji').cook(force=True)

	def decodeEmoji(self, b64png):
		if not b64png:
			return np.zeros((1, 1, 4), dtype=np.float32)
		import cv2  # bundled with TouchDesigner
		bgra = cv2.imdecode(np.frombuffer(base64.b64decode(b64png), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
			return np.zeros((1, 1, 4), dtype=np.float32)
		rgba = bgra[:, :, [2, 1, 0, 3]].astype(np.float32) / 255.0
		rgba[:, :, :3] *= rgba[:, :, 3:4]  # premultiply alpha for clean blending
		return np.ascontiguousarray(np.flipud(rgba))  # TD's origin is bottom-left

	def fill_llm(self, scriptOp):
		scriptOp.numSamples = 1
		for key, value in self.Stats.items():
			scriptOp.appendChan(key)[0] = value
		current = int(self.Stats['scene'])
		for i, name in enumerate(SCENES):
			scriptOp.appendChan('scene_' + name)[0] = 1.0 if i == current else 0.0

	def filltop_emoji(self, scriptOp):
		scriptOp.copyNumpyArray(self.emoji)
