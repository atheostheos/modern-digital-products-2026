# DAT Execute on the caption input of view_llm: restart the typewriter whenever the caption changes.

def onTableChange(dat):
	parent().store('typed_start', absTime.frame)
