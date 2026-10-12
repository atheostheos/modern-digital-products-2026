"""ml_speech: streams the input audio CHOP to the server's speech-to-text node.

Each frame the time-sliced audio CHOP holds the samples since the last frame; they are mixed to mono,
buffered and sent as float32 chunks of `Chunk` seconds with their sample rate (the server resamples).
Outputs: text_transcript (last `Lines` finals), text_final, text_partial, script_speech (speaking,
latency_ms = end-of-speech chunk sent -> final received, finals).
"""

import time

import numpy as np

from MlClientExt import AUDIO_HEADER, MlClientExt


class SpeechExt(MlClientExt):
	NODE = 'speech'

	def __init__(self, ownerComp):
		self.buffer = np.zeros(0, dtype=np.float32)
		self.lastCookFrame = -1
		self.Speaking = 0
		self.LatencyMs = 0.0
		self.Finals = 0
		super().__init__(ownerComp)

	def Tick(self, frame):
		chop = self.ownerComp.op('in_audio')
		if not self.ownerComp.par.Active.eval() or chop is None or chop.numChans == 0:
			return
		samples = chop.numpyArray()
		if samples is None or samples.size == 0 or chop.cookFrame == self.lastCookFrame:
			return
		self.lastCookFrame = chop.cookFrame
		self.buffer = np.concatenate([self.buffer, samples.mean(axis=0).astype(np.float32)])
		rate = int(chop.rate)
		chunk = max(1, int(rate * float(self.ownerComp.par.Chunk.eval())))
		while len(self.buffer) >= chunk:
			piece, self.buffer = self.buffer[:chunk], self.buffer[chunk:]
			header = AUDIO_HEADER.pack(self.nextSeq(), time.perf_counter(), rate)
			self.sendBinary(header + piece.astype('<f4').tobytes())

	def OnResult(self, data, reply):
		kind = data.get('type')
		if kind == 'start':
			self.Speaking = 1
		elif kind == 'partial':
			self.ownerComp.op('text_partial').text = data.get('text', '')
		elif kind == 'final':
			self.Speaking = 0
			self.ownerComp.op('text_partial').text = ''
			text = data.get('text', '')
			if text:
				self.Finals += 1
				self.LatencyMs = self.Status['rtt_ms']
				self.ownerComp.op('text_final').text = text
				lines = [line for line in self.ownerComp.op('text_transcript').text.splitlines() if line] + [text]
				self.ownerComp.op('text_transcript').text = '\n'.join(lines[-int(self.ownerComp.par.Lines.eval()):])
		self.ownerComp.op('script_speech').cook(force=True)

	def fill_speech(self, scriptOp):
		scriptOp.numSamples = 1
		scriptOp.appendChan('speaking')[0] = self.Speaking
		scriptOp.appendChan('latency_ms')[0] = self.LatencyMs
		scriptOp.appendChan('finals')[0] = self.Finals
