"""ml_llm: sends a scene description to the server's gemma3:1b narrator every `Interval` seconds (when
Auto is on) or on the Send pulse. The scene is the input DAT's text if one is wired; otherwise it is built
from the nodes on the Context page (gesture, image label, last transcript, sentiment valence).
Outputs: text_caption, text_scene (what was sent), script_llm (tokens_per_s, latency_ms, captions).
"""

import json
import time

from MlClientExt import MlClientExt


class LlmExt(MlClientExt):
	NODE = 'llm'

	def __init__(self, ownerComp):
		self.lastRequest = 0.0
		self.Stats = {'tokens_per_s': 0.0, 'latency_ms': 0.0, 'captions': 0}
		super().__init__(ownerComp)

	def scene(self):
		dat = self.ownerComp.op('in_scene')
		wired = dat.text.strip() if dat is not None and dat.inputs else ''
		return wired or json.dumps(self.contextScene())

	def contextScene(self):
		par = self.ownerComp.par
		scene = {}
		gesture = par.Gesture.eval()
		if gesture is not None:
			hands = gesture.op('table_hands')
			scene['gesture'] = hands[1, 'gesture'].val if hands is not None and hands.numRows > 1 else 'None'
		classify = par.Classify.eval()
		if classify is not None:
			scene['image'] = classify.op('text_label').text
		speech = par.Speech.eval()
		if speech is not None:
			scene['speech'] = speech.op('text_final').text
		sentiment = par.Sentiment.eval()
		if sentiment is not None:
			chop = sentiment.op('script_sentiment')
			scene['valence'] = round(chop['valence'][0], 3) if chop is not None and chop.numChans else 0.0
		return scene

	def Tick(self, frame):
		par = self.ownerComp.par
		if not par.Auto.eval() or time.perf_counter() - self.lastRequest < float(par.Interval.eval()):
			return
		self.SendNow()

	def SendNow(self):
		scene = self.scene()
		if self.SendText(scene):
			self.lastRequest = time.perf_counter()
			self.ownerComp.op('text_scene').text = scene

	def OnResult(self, data, reply):
		self.ownerComp.op('text_caption').text = data.get('caption', '')
		self.Stats = {
			'tokens_per_s': data.get('tokens_per_s') or 0.0,
			'latency_ms': self.Status['rtt_ms'],
			'captions': self.Stats['captions'] + 1,
		}
		self.ownerComp.op('script_llm').cook(force=True)

	def fill_llm(self, scriptOp):
		scriptOp.numSamples = 1
		for key, value in self.Stats.items():
			scriptOp.appendChan(key)[0] = value
