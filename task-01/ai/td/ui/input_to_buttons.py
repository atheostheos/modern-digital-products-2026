# Parameter Execute DAT in /project1/ui: when ml_input changes elsewhere, keep the buttons in sync.
# (Setting a button to the value it already has triggers nothing, so the two DATs cannot loop.)

def onValueChange(par, prev):
	ui = parent()
	if par.name == 'Source':
		ui.op('button_live').par.value0 = par.eval() == 'live'
	elif par.name == 'Monitor':
		ui.op('button_monitor').par.value0 = bool(par.eval())
