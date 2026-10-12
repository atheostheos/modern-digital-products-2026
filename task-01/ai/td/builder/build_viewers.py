"""Builds the view_* COMPs (visualisation of the ml_* node outputs) in TouchDesigner.

    exec(open(project.folder + '/td/builder/build_viewers.py', encoding='utf-8').read())
    build_view('view_gesture')    # or build_all_views(); then wire inputs (see WIRING)

Each viewer: In OPs -> data prep -> glslmultiTOP (td/glsl/<name>.frag, #include <glsl_common>) ->
Text TOP overlays -> null_view -> out_view (640x480). Rebuilding replaces the COMP.
"""

PARENT = '/project1'
W, H = 640, 480
DASH_W, DASH_H = 1280, 640  # TouchDesigner Non-Commercial caps textures at 1280x1280
ALWAYS = {'cooktype': 'always'}  # time-based CHOPs (lag/trail) must cook every frame, even when not displayed
IN_X, PREP_X, GLSL_X = -350, -175, 0
ROW = -125
UI_FONT, MONO_FONT = 'Avenir Next', 'Menlo'
TITLE = dict(font=UI_FONT, size=17, color=(0.45, 0.48, 0.53), align=('left', 'top'), pos=(24, -18))


def ch(opname, chan, default=0):
	"""Expression reading one CHOP channel value safely."""
	return "(op('{0}')['{1}'][0] if op('{0}') is not None and op('{0}')['{1}'] is not None else {2})".format(
		opname, chan, default)


VIEWS = {
	'view_gesture': dict(pos=(525, -25), title='GESTURE  ·  MediaPipe Gesture Recognizer',
		inputs=[('TOP', 'in_video'), ('CHOP', 'in_landmarks'), ('CHOP', 'in_gestures'), ('DAT', 'in_hands')],
		prep=[('fitTOP', 'fit_video', 'in_video', {'fit': 'fitbest', 'outputresolution': 'custom',
				'resolutionw': W, 'resolutionh': H}),
			('choptoTOP', 'chopto_landmarks', None, {'chop': 'in_landmarks', 'dataformat': 'r', 'format': 'rgba32float'})],
		glsl_inputs=['fit_video', 'chopto_landmarks'],
		uniforms=[('uHands', [ch('in_gestures', 'hands')])],
		texts=[dict(name='text_hands', font=UI_FONT, size=26, color=(1, 1, 1), align=('left', 'bottom'), pos=(24, 20),
			expr="'   '.join('{} hand  {}  {:.0%}'.format(op('in_hands')[r, 'handedness'].val, "
				"op('in_hands')[r, 'gesture'].val.replace('_', ' '), float(op('in_hands')[r, 'gesture_score'].val)) "
				"for r in range(1, op('in_hands').numRows)) or 'no hands'")]),
	'view_classify': dict(pos=(700, -25), title='IMAGE  ·  MobileNetV3-Small (ImageNet-1k)',
		inputs=[('DAT', 'in_top'), ('CHOP', 'in_classes')],
		prep=[('lagCHOP', 'lag_classes', 'in_classes', {'lag1': 0.25, 'lag2': 0.25}),
			('nullCHOP', 'null_classes', 'lag_classes', ALWAYS)],
		uniforms=[('uScores', [ch('null_classes', 'score%d' % i) for i in range(1, 5)]),
			('uScore5', [ch('null_classes', 'score5')])],
		texts=[dict(name='text_labels', font=UI_FONT, size=24, color=(0.92, 0.93, 0.95), align=('left', 'top'),
			pos=(32, -92), linespacing=28,
			expr="'\\n'.join('{}   {:.0%}'.format(op('in_top')[r, 'label'].val, float(op('in_top')[r, 'score'].val)) "
				"for r in range(1, op('in_top').numRows))")]),
	'view_speech': dict(pos=(525, -200), title='SPEECH  ·  faster-whisper tiny.en + Silero VAD',
		inputs=[('DAT', 'in_transcript'), ('DAT', 'in_partial'), ('CHOP', 'in_speech')],
		prep=[('lfoCHOP', 'lfo_pulse', None, {'frequency': 1.5, 'amp': 0.5, 'offset': 0.5})],
		uniforms=[('uSpeaking', [ch('in_speech', 'speaking')]), ('uPulse', [ch('lfo_pulse', 'chan1')])],
		texts=[dict(name='text_transcript', font=UI_FONT, size=24, color=(0.95, 0.95, 0.97), align=('left', 'bottom'),
				pos=(36, 92), wordwrap=True, expr="'\\n'.join(op('in_transcript').text.splitlines()[-3:])"),
			dict(name='text_partial', font=UI_FONT, size=20, color=(0.55, 0.58, 0.63), align=('left', 'bottom'),
				pos=(36, 26), expr="op('in_partial').text or ('listening…' if " + ch('in_speech', 'speaking') + " else '')")]),
	'view_sentiment': dict(pos=(700, -200), title='SENTIMENT  ·  DistilBERT SST-2',
		inputs=[('CHOP', 'in_sentiment'), ('DAT', 'in_label'), ('DAT', 'in_text')],
		prep=[('selectCHOP', 'select_valence', 'in_sentiment', {'channames': 'valence'}),
			('lagCHOP', 'lag_valence', 'select_valence', {'lag1': 0.4, 'lag2': 0.4}),
			('trailCHOP', 'trail_valence', 'lag_valence', {'wlength': 600, 'wlengthunit': 'frames'}),
			('nullCHOP', 'null_trail', 'trail_valence', ALWAYS),
			('choptoTOP', 'chopto_trail', None, {'chop': 'null_trail', 'dataformat': 'r', 'format': 'rgba32float'})],
		glsl_inputs=['chopto_trail'],
		uniforms=[('uValence', [ch('lag_valence', 'valence')])],
		texts=[dict(name='text_value', font=UI_FONT, size=30, color=(1, 1, 1), align=('left', 'bottom'), pos=(32, 20),
				expr="'{}   {:+.2f}'.format(op('in_label').text or '—', " + ch('in_sentiment', 'valence') + ")"),
			dict(name='text_sentence', font=UI_FONT, size=18, color=(0.75, 0.77, 0.8), align=('left', 'top'),
				pos=(32, -52), wordwrap=True, expr="'“{}”'.format(op('in_text').text) if op('in_text').text else ''")]),
	'view_llm': dict(pos=(525, -375), title='NARRATOR  ·  gemma3:1b via Ollama (CPU)',
		inputs=[('DAT', 'in_caption'), ('CHOP', 'in_llm'), ('CHOP', 'in_sentiment')],
		prep=[('datexecuteDAT', 'typewriter', 'in_caption',
			{'file': 'td/viewers/typewriter.py', 'syncfile': True, 'language': 'python', 'tablechange': True})],
		uniforms=[('uValence', [ch('in_sentiment', 'valence')])],
		texts=[dict(name='text_caption', font=UI_FONT, size=36, color=(1, 1, 1), align=('center', 'center'), pos=(0, 10),
				wordwrap=True, expr="op('in_caption').text[:max(0, int((absTime.frame - parent().fetch('typed_start', 0)) * 0.7))]"),
			dict(name='text_stats', font=UI_FONT, size=16, color=(0.6, 0.63, 0.68), align=('right', 'bottom'),
				pos=(-24, 52), expr="'{:.0f} tok/s   ·   {:.0f} ms'.format(" + ch('in_llm', 'tokens_per_s') + ", "
				+ ch('in_llm', 'latency_ms') + ")")]),
	'view_metrics': dict(pos=(700, -375), title='SERVER  ·  latency p50 per node (log scale, last 20 s)',
		inputs=[('CHOP', 'in_metrics'), ('DAT', 'in_nodes')],
		prep=[('selectCHOP', 'select_p50', 'in_metrics',
				{'channames': 'gesture_p50_ms classify_p50_ms speech_p50_ms sentiment_p50_ms llm_p50_ms'}),
			('trailCHOP', 'trail_p50', 'select_p50', {'wlength': 1200, 'wlengthunit': 'frames'}),  # <= 1280 px
			('nullCHOP', 'null_trail', 'trail_p50', ALWAYS),
			('choptoTOP', 'chopto_p50', None, {'chop': 'null_trail', 'dataformat': 'r', 'format': 'rgba32float'}),
			('textDAT', 'fmt', None, {'file': 'td/viewers/metrics_fmt.py', 'syncfile': True, 'language': 'python'})],
		glsl_inputs=['chopto_p50'],
		uniforms=[],
		texts=[dict(name='text_table', font=MONO_FONT, size=15, color=(0.88, 0.9, 0.93), align=('left', 'top'),
				pos=(32, -52), expr="op('fmt').module.table_text(op('in_nodes'), op('in_metrics'))"),
			dict(name='text_legend', font=UI_FONT, size=12, color=(0.55, 0.58, 0.63), align=('left', 'bottom'),
				pos=(32, 8), expr="'cyan gesture · amber classify · magenta speech · green sentiment · violet llm'")]),
}

WIRING = {  # viewer input -> (source COMP, source Out OP)
	'view_gesture': [('ml_input', 'out_video'), ('ml_gesture', 'out_landmarks'), ('ml_gesture', 'out_gestures'),
		('ml_gesture', 'out_hands')],
	'view_classify': [('ml_classify', 'out_top'), ('ml_classify', 'out_classes')],
	'view_speech': [('ml_speech', 'out_transcript'), ('ml_speech', 'out_partial'), ('ml_speech', 'out_speech')],
	'view_sentiment': [('ml_sentiment', 'out_sentiment'), ('ml_sentiment', 'out_label'), ('ml_speech', 'out_final')],
	'view_llm': [('ml_llm', 'out_caption'), ('ml_llm', 'out_llm'), ('ml_sentiment', 'out_sentiment')],
	'view_metrics': [('ml_server', 'out_metrics'), ('ml_server', 'out_nodes')],
}

FAMILY_IN = {'TOP': inTOP, 'CHOP': inCHOP, 'DAT': inDAT}


def _place(o, x, y):
	o.nodeX, o.nodeY = x, y
	o.viewer = True
	return o


def _set(o, params):
	for name, value in params.items():
		setattr(o.par, name, value)


def _glsl(comp, name, spec):
	common = _place(comp.create(textDAT, 'glsl_common'), GLSL_X, 2 * -ROW)
	common.par.file = 'td/glsl/common.glsl'
	common.par.syncfile = True
	common.par.language = 'glsl'
	glsl = _place(comp.create(glslmultiTOP, 'glsl_view'), GLSL_X, 0)
	for dock in list(glsl.docked):
		if dock.name.endswith('_compute'):
			dock.destroy()
	pixel = comp.op('glsl_view_pixel')
	pixel.par.file = 'td/glsl/{}.frag'.format(name)
	pixel.par.syncfile = True
	pixel.par.language = 'glsl'
	glsl.par.outputresolution = 'custom'
	glsl.par.resolutionw, glsl.par.resolutionh = W, H
	for i, src in enumerate(spec.get('glsl_inputs', [])):
		glsl.inputConnectors[i].connect(comp.op(src))
	if spec['uniforms']:
		glsl.seq.vec.numBlocks = len(spec['uniforms'])
	for i, (uniform, exprs) in enumerate(spec['uniforms']):
		setattr(glsl.par, 'vec{}name'.format(i), uniform)
		for axis, expr in zip('xyzw', exprs):
			getattr(glsl.par, 'vec{}value{}'.format(i, axis)).expr = expr
	return glsl


def _text(comp, prev, row, cfg):
	text = _place(comp.create(textTOP, cfg['name']), GLSL_X + 175 * row, 0)
	text.inputConnectors[0].connect(prev)
	text.par.text.expr = cfg['expr']
	text.par.font = cfg['font']
	text.par.fontsizex = cfg['size']
	text.par.fontcolorr, text.par.fontcolorg, text.par.fontcolorb = cfg['color']
	text.par.alignx, text.par.aligny = cfg['align']
	text.par.positionunit = 'pixels'
	text.par.positionx, text.par.positiony = cfg['pos']
	text.par.bgalpha = 0
	if cfg.get('wordwrap'):
		text.par.wordwrap = True
	if cfg.get('linespacing') is not None:
		text.par.linespacingunit = 'pixels'
		text.par.linespacing = cfg['linespacing']
	return text


def build_view(name, parent_path=PARENT):
	spec = VIEWS[name]
	parent = op(parent_path)
	if parent.op(name) is not None:
		parent.op(name).destroy()
	comp = parent.create(baseCOMP, name)
	comp.nodeX, comp.nodeY = spec['pos']
	for row, (family, in_name) in enumerate(spec['inputs']):
		_place(comp.create(FAMILY_IN[family], in_name), IN_X, row * ROW)
	for row, (kind, prep_name, source, params) in enumerate(spec['prep']):
		prep = _place(comp.create(getattr(td, kind), prep_name), PREP_X, row * ROW)
		if source is not None and kind != 'datexecuteDAT':
			prep.inputConnectors[0].connect(comp.op(source))
		if kind == 'datexecuteDAT':
			prep.par.dat = source
		_set(prep, params)
	prev = _glsl(comp, name, spec)
	texts = [dict(TITLE, name='text_title', expr=repr(spec['title']))] + spec['texts']
	for row, cfg in enumerate(texts, start=1):
		prev = _text(comp, prev, row, cfg)
	null = _place(comp.create(nullTOP, 'null_view'), GLSL_X + 175 * (len(texts) + 1), 0)
	null.inputConnectors[0].connect(prev)
	out = _place(comp.create(outTOP, 'out_view'), GLSL_X + 175 * (len(texts) + 2), 0)
	out.inputConnectors[0].connect(null)
	return comp


def _connector(connectors, attr, op_name):
	"""COMP connectors are not ordered by build order: find the one whose In/Out OP has this name."""
	return next(c for c in connectors if getattr(c, attr).name == op_name)


def wire_view(name, parent_path=PARENT):
	parent, comp = op(parent_path), op(parent_path).op(name)
	for (_, in_name), (src_comp, out_name) in zip(VIEWS[name]['inputs'], WIRING[name]):
		source = _connector(parent.op(src_comp).outputConnectors, 'outOP', out_name)
		source.connect(_connector(comp.inputConnectors, 'inOP', in_name))


def build_all_views():
	for name in VIEWS:
		build_view(name)
		wire_view(name)


DASHBOARD_ORDER = ['view_gesture', 'view_classify', 'view_speech', 'view_sentiment', 'view_llm', 'view_metrics']


def build_dashboard(name='dashboard', parent_path=PARENT, pos=(900, -200)):
	"""3x2 grid of all viewers (each 640x480, scaled into 1280x640) via a tiling GLSL shader; out_dashboard."""
	parent = op(parent_path)
	if parent.op(name) is not None:
		parent.op(name).destroy()
	comp = parent.create(baseCOMP, name)
	comp.nodeX, comp.nodeY = pos
	common = _place(comp.create(textDAT, 'glsl_common'), GLSL_X, 2 * -ROW)
	common.par.file, common.par.syncfile, common.par.language = 'td/glsl/common.glsl', True, 'glsl'
	glsl = _place(comp.create(glslmultiTOP, 'glsl_tile'), GLSL_X, 0)
	for dock in list(glsl.docked):
		if dock.name.endswith('_compute'):
			dock.destroy()
	pixel = comp.op('glsl_tile_pixel')
	pixel.par.file, pixel.par.syncfile, pixel.par.language = 'td/glsl/dashboard.frag', True, 'glsl'
	glsl.par.outputresolution = 'custom'
	glsl.par.resolutionw, glsl.par.resolutionh = DASH_W, DASH_H
	for row, view in enumerate(DASHBOARD_ORDER):
		inp = _place(comp.create(inTOP, 'in_' + view.replace('view_', '')), IN_X, row * ROW)
		glsl.inputConnectors[row].connect(inp)
		source = parent.op(view).outputConnectors[0]
		source.connect(_connector(comp.inputConnectors, 'inOP', inp.name))
	null = _place(comp.create(nullTOP, 'null_dashboard'), GLSL_X + 175, 0)
	null.inputConnectors[0].connect(glsl)
	out = _place(comp.create(outTOP, 'out_dashboard'), GLSL_X + 350, 0)
	out.inputConnectors[0].connect(null)
	return comp
