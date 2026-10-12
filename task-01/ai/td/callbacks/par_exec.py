# Parameter Execute DAT shared by ml_* nodes: connection settings and pulses go to the parent extension.

CONNECTION_PARS = ('Host', 'Port', 'Active')


def onValueChange(par, prev):
	if par.name in CONNECTION_PARS:
		parent().Configure()


def onPulse(par):
	parent().OnPulse(par.name)
