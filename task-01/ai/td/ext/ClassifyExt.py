"""ml_classify: streams the input TOP to the server's MobileNetV3 image classifier.

Outputs: table_top (rank, label, score), script_classes (top1_score + score1..scoreN), text_label.
"""

from MlClientExt import MlClientExt


class ClassifyExt(MlClientExt):
	NODE = 'classify'

	def __init__(self, ownerComp):
		self.Top = []
		super().__init__(ownerComp)

	def Tick(self, frame):
		if self.ready():
			self.SendFrame(self.ownerComp.op('fit_send'))

	def OnResult(self, data, reply):
		self.Top = data.get('top', [])
		table = self.ownerComp.op('table_top')
		table.clear()
		table.appendRow(['rank', 'label', 'score'])
		for rank, item in enumerate(self.Top, start=1):
			table.appendRow([rank, item['label'], item['score']])
		self.ownerComp.op('text_label').text = data.get('label', '')
		self.ownerComp.op('script_classes').cook(force=True)

	def fill_classes(self, scriptOp):
		scriptOp.numSamples = 1
		scriptOp.appendChan('top1_score')[0] = self.Top[0]['score'] if self.Top else 0.0
		for rank, item in enumerate(self.Top, start=1):
			scriptOp.appendChan('score{}'.format(rank))[0] = item['score']
