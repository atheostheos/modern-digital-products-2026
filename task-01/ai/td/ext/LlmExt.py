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

from MlClientExt import MlClientExt, isNumber

SCENES = ['storm', 'sunrise', 'fireworks', 'ocean', 'neon', 'starfield']
SENTIMENT_WAIT_S = 1.0  # max wait for the sentence's own sentiment (normally ~25 ms) before asking the LLM


class LlmExt(MlClientExt):
	NODE = 'llm'

	def __init__(self, ownerComp):
		self.lastSpeech = None
		self.lastRequest = 0.0
		self.waitingFor = None  # (sentence, since) while waiting for its sentiment
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
		final = self.childOp(par.Speech.eval(), 'text_final')
		if final is not None:
			scene['speech'] = final.text.strip()
		hands = self.childOp(par.Gesture.eval(), 'table_hands')
		cell = hands[1, 'gesture'] if hands is not None and hands.numRows > 1 else None
		if cell is not None:
			scene['gesture'] = cell.val
		chop = self.childOp(par.Sentiment.eval(), 'script_sentiment')
		channel = chop['valence'] if chop is not None else None
		if channel is not None:
			scene['valence'] = round(channel[0], 3)
		return scene

	@staticmethod
	def childOp(comp, name):
		return comp.op(name) if comp is not None else None

	def sentimentReady(self, speech):
		"""True once the Sentiment node has scored this sentence (its valence drives the fallback), or after
		SENTIMENT_WAIT_S, or when there is no Sentiment node to wait for."""
		sentiment = self.ownerComp.par.Sentiment.eval()
		ext = getattr(sentiment.ext, 'SentimentExt', None) if sentiment is not None else None
		if ext is None or getattr(ext, 'ScoredText', None) == speech:
			return True
		now = time.perf_counter()
		if self.waitingFor is None or self.waitingFor[0] != speech:
			self.waitingFor = (speech, now)
		return now - self.waitingFor[1] >= SENTIMENT_WAIT_S

	def Tick(self, frame):
		"""React to each new sentence (at most once per Interval seconds)."""
		par = self.ownerComp.par
		if not par.Auto.eval() or time.perf_counter() - self.lastRequest < float(par.Interval.eval()):
			return
		wired = self.wiredScene()
		speech = wired or self.contextScene()['speech']
		if speech and speech != self.lastSpeech and (wired or self.sentimentReady(speech)):
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
		scene = data.get('scene') if data.get('scene') in SCENES else 'starfield'
		tokens = data.get('tokens_per_s')
		self.emoji = self.decodeEmoji(data.get('emoji_png'))
		self.ownerComp.op('text_scene').text = scene
		self.ownerComp.op('text_symbol').text = str(data.get('emoji', ''))
		self.Stats = {
			'scene': float(SCENES.index(scene)),
			'directed_frame': float(absTime.frame),
			'tokens_per_s': tokens if isNumber(tokens) else 0.0,
			'latency_ms': self.Status['rtt_ms'],
			'directions': self.Stats['directions'] + 1,
		}
		self.ownerComp.op('script_llm').cook(force=True)
		self.ownerComp.op('script_emoji').cook(force=True)

	def decodeEmoji(self, b64png):
		if not isinstance(b64png, str) or not b64png:
			return np.zeros((1, 1, 4), dtype=np.float32)
		import cv2  # bundled with TouchDesigner
		try:
			raw = np.frombuffer(base64.b64decode(b64png, validate=True), dtype=np.uint8)
			bgra = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
		except (ValueError, cv2.error) as exc:  # binascii.Error is a ValueError
			self.recordError('bad emoji picture: {}'.format(exc))
			bgra = None
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
