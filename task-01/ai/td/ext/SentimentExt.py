"""ml_sentiment: sends the input DAT's text to the server's DistilBERT SST-2 node whenever it changes.

Outputs: script_sentiment (positive, negative, valence in -1..1), text_label.
"""

from MlClientExt import MlClientExt, isNumber


class SentimentExt(MlClientExt):
	NODE = 'sentiment'

	def __init__(self, ownerComp):
		self.lastText = None
		self.pendingText = ''
		self.ScoredText = ''  # the text the current scores belong to (ml_llm waits for it)
		self.Scores = {'positive': 0.0, 'negative': 0.0, 'valence': 0.0}
		super().__init__(ownerComp)

	def inputText(self):
		dat = self.ownerComp.op('in_text')
		return dat.text.strip() if dat is not None else ''

	def Tick(self, frame):
		text = self.inputText()
		if text and text != self.lastText and self.ready():
			if self.SendText(text):
				self.lastText = self.pendingText = text

	def SendNow(self):
		text = self.inputText()
		if self.SendText(text):
			self.pendingText = text

	def OnResult(self, data, reply):
		self.Scores = {key: data[key] if isNumber(data.get(key)) else 0.0 for key in self.Scores}
		self.ScoredText = self.pendingText
		self.ownerComp.op('text_label').text = str(data.get('label', ''))
		self.ownerComp.op('script_sentiment').cook(force=True)

	def fill_sentiment(self, scriptOp):
		scriptOp.numSamples = 1
		for key, value in self.Scores.items():
			scriptOp.appendChan(key)[0] = value
