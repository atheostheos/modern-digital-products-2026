"""Builds /project1/ui: a clickable control bar (Recorded/Live source, audio monitor) above the dashboard,
and points window_dashboard at it.

    exec(open(project.folder + '/td/builder/build_ui.py', encoding='utf-8').read())
    build_ui()
"""

PARENT = '/project1'
BAR_H = 48
VIEW_W, VIEW_H = 1280, 640
BUTTON_H, BUTTON_PAD = 32, 8
BG = (0.055, 0.063, 0.078)
LIVE_LABEL = "'●  LIVE  ·  camera + microphone' if me.par.value0 else '▶  RECORDED  ·  media files'"
MONITOR_LABEL = "'🔊  monitor audio: on' if me.par.value0 else '🔈  monitor audio: off'"


def _button(ui, name, x, w, label_expr, on_color):
	button = ui.create(buttonCOMP, name)
	button.name = name  # TD may append a digit on create; force the exact name the callbacks expect
	button.par.buttontype = 'toggledown'
	button.par.x, button.par.y = x, VIEW_H + (BAR_H - BUTTON_H) // 2
	button.par.w, button.par.h = w, BUTTON_H
	button.par.label.expr = label_expr
	button.par.fontsize = 15
	button.par.colorr.expr, button.par.colorg.expr, button.par.colorb.expr = (
		"{} if me.par.value0 else 0.16".format(on_color[0]),
		"{} if me.par.value0 else 0.17".format(on_color[1]),
		"{} if me.par.value0 else 0.20".format(on_color[2]))
	return button


def build_ui(parent_path=PARENT, pos=(1075, -200)):
	parent = op(parent_path)
	if parent.op('ui') is not None:
		parent.op('ui').destroy()
	ui = parent.create(containerCOMP, 'ui')
	ui.nodeX, ui.nodeY = pos
	ui.par.w, ui.par.h = VIEW_W, VIEW_H + BAR_H
	ui.par.bgcolorr, ui.par.bgcolorg, ui.par.bgcolorb = BG
	view = ui.create(containerCOMP, 'panel_view')
	view.par.x, view.par.y, view.par.w, view.par.h = 0, 0, VIEW_W, VIEW_H
	view.par.top = parent_path + '/dashboard/null_dashboard'
	live = _button(ui, 'button_live', BUTTON_PAD * 2, 320, LIVE_LABEL, (0.85, 0.15, 0.2))
	monitor = _button(ui, 'button_monitor', BUTTON_PAD * 3 + 320, 240, MONITOR_LABEL, (0.2, 0.55, 0.9))
	source = parent.op('ml_input')
	live.par.value0 = source.par.Source.eval() == 'live'
	monitor.par.value0 = bool(source.par.Monitor.eval())
	clicks = ui.create(panelexecuteDAT, 'buttons_to_input')
	clicks.par.file, clicks.par.syncfile, clicks.par.language = 'td/ui/buttons_to_input.py', True, 'python'
	clicks.par.panels.mode = ParMode.CONSTANT  # defaults to the expression parent()
	clicks.par.panels = 'button_live button_monitor'
	clicks.par.offtoon, clicks.par.ontooff = True, True
	follow = ui.create(parameterexecuteDAT, 'input_to_buttons')
	follow.par.file, follow.par.syncfile, follow.par.language = 'td/ui/input_to_buttons.py', True, 'python'
	follow.par.op.mode = ParMode.CONSTANT
	follow.par.op = parent_path + '/ml_input'
	follow.par.pars = 'Source Monitor'
	follow.par.custom, follow.par.builtin, follow.par.valuechange = True, False, True
	for i, o in enumerate((view, live, monitor, clicks, follow)):
		o.nodeX, o.nodeY = 0, i * -125
	window = parent.op('window_dashboard')
	if window is not None:
		window.par.winop = 'ui'
		window.par.winw, window.par.winh = VIEW_W, VIEW_H + BAR_H
	return ui
