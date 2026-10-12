"""ml_server: subscribes to the server's /ws/metrics stream (pushed every 0.5 s).

Outputs: table_nodes (node, requests, processed, dropped, errors, p50_ms, p95_ms, fps),
script_metrics (<node>_p50_ms, <node>_p95_ms, <node>_fps, <node>_dropped ... + cpu_percent, rss_mb).
"""

import json

from MlClientExt import MlClientExt, isNumber

NODE_FIELDS = ['requests', 'processed', 'dropped', 'errors', 'p50_ms', 'p95_ms', 'fps']
CHOP_FIELDS = ['p50_ms', 'p95_ms', 'fps', 'dropped', 'errors']


def parseSnapshot(snapshot):
	"""{'nodes': [{'node': str, ...}], 'process': {...}} with unusable parts dropped; None if not a snapshot."""
	if not isinstance(snapshot, dict):
		return None
	nodes = snapshot.get('nodes')
	process = snapshot.get('process')
	return {
		'nodes': [n for n in nodes if isinstance(n, dict) and isinstance(n.get('node'), str)]
			if isinstance(nodes, list) else [],
		'process': {str(k): v for k, v in process.items()} if isinstance(process, dict) else {},
	}


class ServerExt(MlClientExt):
	NODE = 'metrics'

	def __init__(self, ownerComp):
		self.Snapshot = {'nodes': [], 'process': {}}
		super().__init__(ownerComp)

	def OnReceiveText(self, message):
		try:
			snapshot = parseSnapshot(json.loads(message))
		except ValueError:
			snapshot = None
		if snapshot is None:
			self.recordError('unreadable metrics from server')
			return
		self.Snapshot = snapshot
		self.setConnected(True)
		self.Status['received'] += 1
		table = self.ownerComp.op('table_nodes')
		table.clear()
		table.appendRow(['node'] + NODE_FIELDS)
		for node in self.Snapshot['nodes']:
			table.appendRow([node['node']] + ['' if node.get(f) is None else node[f] for f in NODE_FIELDS])
		self.ownerComp.op('script_metrics').cook(force=True)
		self.refreshStatus()

	def fill_metrics(self, scriptOp):
		scriptOp.numSamples = 1
		for node in self.Snapshot['nodes']:
			for field in CHOP_FIELDS:
				value = node.get(field)
				scriptOp.appendChan('{}_{}'.format(node['node'], field))[0] = value if isNumber(value) else 0.0
		for key, value in self.Snapshot['process'].items():
			scriptOp.appendChan(key)[0] = value if isNumber(value) else 0.0
