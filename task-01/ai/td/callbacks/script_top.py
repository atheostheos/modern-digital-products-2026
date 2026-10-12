# Script TOP callbacks shared by ml_* nodes: the parent extension fills the TOP by its name.

def onCook(scriptOp):
	parent().FillTop(scriptOp)
