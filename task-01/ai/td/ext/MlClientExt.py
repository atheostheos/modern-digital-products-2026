"""Shared client logic for every ml_* node: connection, sequence numbers, latency and status.

Subclasses set NODE (the server node name) and override OnResult(); image nodes call SendFrame(top).
All inputs are captured in TouchDesigner and streamed to the local ML server (tdml-server).
"""

import json
import struct
import time

import numpy as np

FRAME_HEADER = struct.Struct('<Id')  # seq, t_send
AUDIO_HEADER = struct.Struct('<IdI')  # seq, t_send, sample_rate
JPEG_QUALITY = 85
RATE_WINDOW_S = 2.0
RECONNECT_DELAY_FRAMES = 2


class MlClientExt:
	"""Base extension: subclass per node."""

	NODE = ''

	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.seq = 0
		self.lastSend = 0.0
		self.replyTimes = []
		self.Status = {'connected': 0, 'sent': 0, 'received': 0, 'errors': 0,
			'rtt_ms': 0.0, 'inference_ms': 0.0, 'replies_per_s': 0.0}
		self.LastError = ''
		# Connection state is inferred from traffic (a send that succeeds or a reply), not only from
		# onConnect: extension re-init (e.g. after a code edit) wipes attributes while the socket stays open.
		self.Status['connected'] = int(self.ownerComp.fetch('connected', 0, storeDefault=False))
		self.Configure()

	# ------------------------------------------------------------ connection
	@property
	def ws(self):
		return self.ownerComp.op('websocket')

	def Configure(self):
		"""Apply Host/Port/Active parameters to the WebSocket DAT."""
		par = self.ownerComp.par
		self.ws.par.netaddress = 'ws://{}/ws/{}'.format(par.Host.eval(), self.NODE)
		self.ws.par.port = int(par.Port.eval())
		self.ws.par.active = bool(par.Active.eval())
		if not par.Active.eval():
			self.setConnected(False)

	def OnPulse(self, name):
		"""Pulse parameters: Reconnect for all nodes; Send for nodes that implement SendNow()."""
		if name == 'Reconnect':
			self.Reconnect()
		elif name == 'Send':
			self.SendNow()

	def SendNow(self):
		"""Override: send immediately (text nodes)."""

	def Reconnect(self):
		"""Drop the socket now and reopen it a couple of frames later (same-frame toggles are ignored)."""
		self.ws.par.active = False
		self.setConnected(False)
		run("op({!r}).Configure()".format(self.ownerComp.path), delayFrames=RECONNECT_DELAY_FRAMES)

	def setConnected(self, connected):
		if self.Status['connected'] == int(connected):
			return
		self.ownerComp.store('connected', int(connected))
		self.Status['connected'] = int(connected)
		self.refreshStatus()

	def OnConnect(self):
		self.setConnected(True)

	def OnDisconnect(self):
		self.setConnected(False)

	# ------------------------------------------------------------ sending
	def nextSeq(self):
		self.seq += 1
		return self.seq

	def ready(self):
		par = self.ownerComp.par
		if not par.Active.eval():
			return False
		interval = 1.0 / max(float(par.Rate.eval()), 0.1) if hasattr(par, 'Rate') else 0.0
		return time.perf_counter() - self.lastSend >= interval

	def sendBinary(self, payload):
		return self.afterSend(self.ws.sendBinary(payload))

	def afterSend(self, result):
		"""A negative result means the socket is not open: mark disconnected and retry next interval."""
		self.lastSend = time.perf_counter()
		if result < 0:
			self.setConnected(False)
			return False
		self.setConnected(True)
		self.Status['sent'] += 1
		return True

	def SendFrame(self, top):
		"""Encode the TOP as JPEG and send it. numpyArray(delayed=True) avoids a GPU stall (1 frame old)."""
		arr = top.numpyArray(delayed=True)
		if arr is None:
			return False
		import cv2  # bundled with TouchDesigner
		rgb = np.clip(arr[:, :, :3], 0.0, 1.0)
		bgr = (np.flipud(rgb[:, :, ::-1]) * 255).astype(np.uint8)  # TD origin is bottom-left
		ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
		if not ok:
			return False
		return self.sendBinary(FRAME_HEADER.pack(self.nextSeq(), time.perf_counter()) + buf.tobytes())

	def SendText(self, text):
		if not self.ownerComp.par.Active.eval() or not text:
			return False
		message = json.dumps({'seq': self.nextSeq(), 't_send': time.perf_counter(), 'text': text})
		return self.afterSend(self.ws.sendText(message))

	# ------------------------------------------------------------ receiving
	def OnReceiveText(self, message):
		try:
			reply = json.loads(message)
		except ValueError:
			self.recordError('unreadable reply from server')
			return
		now = time.perf_counter()
		self.setConnected(True)
		self.Status['received'] += 1
		self.replyTimes = [t for t in self.replyTimes if now - t < RATE_WINDOW_S] + [now]
		self.Status['replies_per_s'] = round(len(self.replyTimes) / RATE_WINDOW_S, 2)
		if reply.get('t_send') is not None:
			self.Status['rtt_ms'] = round((now - reply['t_send']) * 1000, 2)
		if reply.get('timing'):
			self.Status['inference_ms'] = reply['timing']['inference_ms']
		if reply.get('ok'):
			self.OnResult(reply.get('data') or {}, reply)
		else:
			self.recordError(reply.get('error', 'unknown error'))
		self.refreshStatus()

	def OnResult(self, data, reply):
		"""Override in subclasses."""

	def recordError(self, error):
		self.Status['errors'] += 1
		self.LastError = str(error)
		self.refreshStatus()

	# ------------------------------------------------------------ outputs
	def refreshStatus(self):
		chop = self.ownerComp.op('script_status')
		if chop is not None:
			chop.cook(force=True)

	def FillChop(self, scriptOp):
		"""Script CHOP callback entry point: dispatch on the CHOP's name."""
		scriptOp.clear()
		filler = getattr(self, 'fill_' + scriptOp.name.replace('script_', ''), None)
		if filler is not None:
			filler(scriptOp)

	def FillTop(self, scriptOp):
		"""Script TOP callback entry point: dispatch on the TOP's name (copyNumpyArray only works in onCook)."""
		filler = getattr(self, 'filltop_' + scriptOp.name.replace('script_', ''), None)
		if filler is not None:
			filler(scriptOp)

	def fill_status(self, scriptOp):
		scriptOp.numSamples = 1
		for name, value in self.Status.items():
			scriptOp.appendChan(name)[0] = value

	def Tick(self, frame):
		"""Called every frame (Execute DAT). Override for periodic sending."""
