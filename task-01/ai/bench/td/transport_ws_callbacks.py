# WebSocket DAT callbacks (TouchDesigner): record round-trip time for each echoed frame.
# Appends one row per reply to table_rtt: seq, mode, encode_ms, bytes, decode_ms, server_ms, rtt_ms.
import json
import time

COLUMNS = ['seq', 'mode', 'encode_ms', 'bytes', 'decode_ms', 'server_ms', 'rtt_ms']


def onConnect(dat):
	table = op('table_rtt')
	if table.numRows == 0:
		table.appendRow(COLUMNS)
	return


def onDisconnect(dat):
	return


def onReceiveText(dat, rowIndex, message):
	t_now = time.perf_counter()
	reply = json.loads(message)
	if not reply.get('ok'):
		debug('echo error:', reply.get('error'))
		return
	pending = parent().fetch('pending', {})
	mode, encode_ms = pending.pop(reply['seq'], ('?', -1.0))
	parent().store('pending', pending)
	rtt_ms = (t_now - reply['t_send']) * 1000
	op('table_rtt').appendRow([reply['seq'], mode, round(encode_ms, 3), reply['bytes'],
		reply['decode_ms'], reply['server_ms'], round(rtt_ms, 3)])
	return


def onReceiveBinary(dat, contents):
	return


def onReceivePing(dat, contents):
	dat.sendPong(contents)
	return


def onReceivePong(dat, contents):
	return


def onMonitorMessage(dat, message):
	return
