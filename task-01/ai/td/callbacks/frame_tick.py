# Execute DAT shared by ml_* nodes: let the parent extension send on its own schedule.

def onFrameEnd(frame):
	parent().Tick(frame)
