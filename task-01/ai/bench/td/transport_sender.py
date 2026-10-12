# Execute DAT callbacks (TouchDesigner): send the source TOP to the echo server every N frames.
# Settings live in the parent COMP's storage so they can be changed without editing code:
#   mode: 'bytearray' (TOP.saveByteArray JPEG) | 'numpy' (numpyArray(delayed=True) + cv2.imencode)
#   every: send every N frames; quality: JPEG quality 0..1; active: bool
import struct
import time

import cv2
import numpy as np

HEADER = struct.Struct('<Id')
DEFAULTS = {'mode': 'bytearray', 'every': 2, 'quality': 0.8, 'active': False}


def _setting(name):
	return parent().fetch(name, DEFAULTS[name])


def _encode_bytearray(top, quality):
	return bytes(top.saveByteArray('.jpg', quality=quality))


def _encode_numpy(top, quality):
	arr = top.numpyArray(delayed=True)
	if arr is None:
		return None
	# TD arrays are float32 RGBA, origin bottom-left; JPEG wants uint8 BGR, origin top-left.
	bgr = (np.flipud(arr[:, :, 2::-1]) * 255).astype(np.uint8)
	ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality * 100)])
	return buf.tobytes() if ok else None


ENCODERS = {'bytearray': _encode_bytearray, 'numpy': _encode_numpy}


def onFrameEnd(frame):
	if not _setting('active') or frame % _setting('every'):
		return
	mode = _setting('mode')
	t0 = time.perf_counter()
	jpeg = ENCODERS[mode](op('null_source'), _setting('quality'))
	encode_ms = (time.perf_counter() - t0) * 1000
	if jpeg is None:
		return
	seq = parent().fetch('seq', 0) + 1
	parent().store('seq', seq)
	pending = parent().fetch('pending', {})
	pending[seq] = (mode, encode_ms)
	parent().store('pending', pending)
	sent = op('websocket_echo').sendBinary(HEADER.pack(seq, time.perf_counter()) + jpeg)
	if sent < 0:
		pending.pop(seq, None)
		parent().store('send_errors', parent().fetch('send_errors', 0) + 1)
	return
