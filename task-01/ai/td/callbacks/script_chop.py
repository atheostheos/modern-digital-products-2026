# Script CHOP callbacks shared by ml_* nodes: the parent extension fills the CHOP by its name.

def onCook(scriptOp):
	parent().FillChop(scriptOp)
