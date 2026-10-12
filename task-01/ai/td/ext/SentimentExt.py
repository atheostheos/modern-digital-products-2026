"""ml_sentiment: sends the input DAT's text to the server's DistilBERT SST-2 node whenever it changes.

Outputs: script_sentiment (positive, negative, valence in -1..1), text_label.
"""

from MlClientExt import MlClientExt


class SentimentExt(MlClientExt):
	NODE = 'sentiment'

	def __init__(self, ownerComp):
		self.lastText = None
		self.Scores = {'positive': 0.0, 'negative': 0.0, 'valence': 0.0}
		super().__init__(ownerComp)

	def inputText(self):
		dat = self.ownerComp.op('in_text')
		return dat.text.strip() if dat is not None else ''

	def Tick(self, frame):
		text = self.inputText()
		if text and text != self.lastText and self.ready():
			if self.SendText(text):
				self.lastText = text

	def SendNow(self):
		self.SendText(self.inputText())

	def OnResult(self, data, reply):
		self.Scores = {key: data.get(key, 0.0) for key in self.Scores}
		self.ownerComp.op('text_label').text = data.get('label', '')
		self.ownerComp.op('script_sentiment').cook(force=True)

	def fill_sentiment(self, scriptOp):
		scriptOp.numSamples = 1
		for key, value in self.Scores.items():
			scriptOp.appendChan(key)[0] = value
