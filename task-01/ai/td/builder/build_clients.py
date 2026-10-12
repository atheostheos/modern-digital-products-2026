"""Builds the ml_* client COMPs in TouchDesigner from short specs (run inside TD).

    exec(open(project.folder + '/td/builder/build_clients.py', encoding='utf-8').read())
    build('ml_classify')          # or build_all()

Each node: Server page (Host/Port/Active/Reconnect) + its own page, file-synced code DATs from td/ext and
td/callbacks, inputs, Script CHOP / table / text outputs ending in Out OPs, a WebSocket DAT, and the
extension bound and re-initialised. Rebuilding replaces the COMP (wire inputs again afterwards).
"""

PARENT = '/project1'
CODE_X, SRC_X, OUT_X = -275, 0, 175
ROW = -125

SERVER_PARS = [
	('Str', 'Host', 'Host', '127.0.0.1'),
	('Int', 'Port', 'Port', 8000, 1, 65535),
	('Toggle', 'Active', 'Active', True),
	('Pulse', 'Reconnect', 'Reconnect'),
]

SPECS = {
	'ml_gesture': dict(ext='GestureExt', page='Gesture', input='video', pos=(0, -25),
		pars=[('Float', 'Rate', 'Send Rate (fps)', 15, 0.5, 60), ('Int', 'Size', 'Send Size (px)', 256, 64, 1024)],
		outputs=[('chop', 'landmarks'), ('chop', 'gestures'), ('table', 'hands')]),
	'ml_classify': dict(ext='ClassifyExt', page='Classify', input='video', pos=(0, -200),
		pars=[('Float', 'Rate', 'Send Rate (fps)', 2, 0.1, 30), ('Int', 'Size', 'Send Size (px)', 224, 64, 1024)],
		outputs=[('table', 'top'), ('chop', 'classes'), ('text', 'label')]),
	'ml_speech': dict(ext='SpeechExt', page='Speech', input='audio', pos=(0, -375),
		pars=[('Float', 'Chunk', 'Chunk (s)', 0.25, 0.05, 1.0), ('Int', 'Lines', 'Transcript Lines', 10, 1, 100)],
		outputs=[('text', 'transcript'), ('text', 'final'), ('text', 'partial'), ('chop', 'speech')]),
	'ml_sentiment': dict(ext='SentimentExt', page='Sentiment', input='text', pos=(175, -375),
		pars=[('Float', 'Rate', 'Max Rate (msg/s)', 4, 0.1, 30), ('Pulse', 'Send', 'Send Now')],
		outputs=[('chop', 'sentiment'), ('text', 'label')]),
	'ml_llm': dict(ext='LlmExt', page='Narrator', input='scene', pos=(350, -375),
		pars=[('Toggle', 'Auto', 'Auto', True), ('Float', 'Interval', 'Interval (s)', 8, 1, 120),
			('Pulse', 'Send', 'Send Now'), ('COMP', 'Gesture', 'Gesture Node'), ('COMP', 'Classify', 'Classify Node'),
			('COMP', 'Speech', 'Speech Node'), ('COMP', 'Sentiment', 'Sentiment Node')],
		outputs=[('text', 'caption'), ('text', 'scene'), ('chop', 'llm')]),
	'ml_server': dict(ext='ServerExt', page=None, input=None, pos=(-175, -200), pars=[],
		outputs=[('table', 'nodes'), ('chop', 'metrics')]),
}

CODE_DATS = [  # (name, type, file, extra params)
	('MlClientExt', textDAT, 'td/ext/MlClientExt.py', {}),
	(None, textDAT, None, {}),  # the node's extension, filled per spec
	('par_exec', parameterexecuteDAT, 'td/callbacks/par_exec.py',
		{'op': '..', 'pars': '*', 'custom': True, 'builtin': False, 'valuechange': True, 'onpulse': True}),
	('frame_tick', executeDAT, 'td/callbacks/frame_tick.py', {'frameend': True}),
	('ws_callbacks', textDAT, 'td/callbacks/ws_callbacks.py', {}),
	('script_chop', textDAT, 'td/callbacks/script_chop.py', {}),
]


def _add_par(page, kind, name, label, default=None, low=None, high=None):
	par = getattr(page, 'append' + kind)(name, label=label)[0]
	if default is not None:
		par.default = default
		par.val = default
	if low is not None:
		par.min = par.normMin = low
		par.clampMin = True
	if high is not None:
		par.max = par.normMax = high
		par.clampMax = True


def _place(o, x, y):
	o.nodeX, o.nodeY = x, y
	o.viewer = True
	return o


def _drop_docked(o):
	for d in list(o.docked):
		d.destroy()


def _code(comp, spec):
	for row, (name, kind, path, extra) in enumerate(CODE_DATS):
		name = name or spec['ext']
		path = path or 'td/ext/{}.py'.format(spec['ext'])
		dat = _place(comp.create(kind, name), CODE_X, row * ROW)
		dat.par.file = path
		dat.par.syncfile = True
		dat.par.language = 'python'
		for par, value in extra.items():
			setattr(dat.par, par, value)


def _inputs(comp, kind):
	if kind == 'video':
		src = _place(comp.create(inTOP, 'in_video'), SRC_X, 0)
		fit = _place(comp.create(fitTOP, 'fit_send'), OUT_X, 0)
		fit.inputConnectors[0].connect(src)
		fit.par.fit = 'fill'
		fit.par.outputresolution = 'custom'
		fit.par.resolutionw.expr = 'parent().par.Size'
		fit.par.resolutionh.expr = ('int(round(parent().par.Size * me.inputs[0].height / max(me.inputs[0].width, 1)))'
			' if me.inputs else parent().par.Size')
	elif kind == 'audio':
		_place(comp.create(inCHOP, 'in_audio'), SRC_X, 0)
	elif kind in ('text', 'scene'):
		_place(comp.create(inDAT, 'in_' + kind), SRC_X, 0)


def _outputs(comp, outputs):
	for row, (kind, name) in enumerate(outputs + [('chop', 'status')], start=1):
		y = row * ROW
		if kind == 'chop':
			src = _place(comp.create(scriptCHOP, 'script_' + name), SRC_X, y)
			src.par.callbacks = 'script_chop'
			_drop_docked(src)
			out = _place(comp.create(outCHOP, 'out_' + name), OUT_X, y)
		else:
			src = _place(comp.create(tableDAT if kind == 'table' else textDAT, '{}_{}'.format(kind, name)), SRC_X, y)
			out = _place(comp.create(outDAT, 'out_' + name), OUT_X, y)
		out.inputConnectors[0].connect(src)
	return (len(outputs) + 2) * ROW


def build(name, parent_path=PARENT):
	spec = SPECS[name]
	parent = op(parent_path)
	if parent.op(name) is not None:
		parent.op(name).destroy()
	comp = parent.create(baseCOMP, name)
	comp.nodeX, comp.nodeY = spec['pos']
	server = comp.appendCustomPage('Server')
	for par in SERVER_PARS:
		_add_par(server, *par)
	if spec['page']:
		page = comp.appendCustomPage(spec['page'])
		for par in spec['pars']:
			_add_par(page, *par)
	_code(comp, spec)
	_inputs(comp, spec['input'])
	ws_y = _outputs(comp, spec['outputs'])
	ws = _place(comp.create(websocketDAT, 'websocket'), SRC_X, ws_y)
	ws.par.active = False
	ws.par.callbacks = 'ws_callbacks'
	_drop_docked(ws)
	comp.par.ext0object.mode = ParMode.CONSTANT
	comp.par.ext0object = "op('./{0}').module.{0}(me)".format(spec['ext'])
	comp.par.ext0promote = True
	comp.par.reinitextensions.pulse()
	return comp


def build_all(names=None):
	return [build(name) for name in (names or SPECS)]
