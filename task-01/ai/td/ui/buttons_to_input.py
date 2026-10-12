# Panel Execute DAT in /project1/ui: clicking a button sets the matching ml_input parameter.

def _apply(button, on):
	source = parent.Project.op('ml_input')
	if button.name == 'button_live':
		source.par.Source = 'live' if on else 'recorded'
	elif button.name == 'button_monitor':
		source.par.Monitor = on


def onOffToOn(panelValue):
	_apply(panelValue.owner, True)


def onOnToOff(panelValue):
	_apply(panelValue.owner, False)
