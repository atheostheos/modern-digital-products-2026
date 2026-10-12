# Formats the ml_server node table for view_metrics (monospace Text TOP).

FIELDS = [('node', 10), ('p50_ms', 9), ('p95_ms', 9), ('fps', 6), ('dropped', 8), ('errors', 7)]


def _cell(table, row, field):
	cell = table[row, field] if table is not None else None
	return '' if cell is None else cell.val


def table_text(table, metrics):
	header = ''.join(name.ljust(width) for name, width in FIELDS)
	lines = [header]
	for row in range(1, table.numRows if table is not None else 0):
		values = []
		for name, width in FIELDS:
			value = _cell(table, row, name)
			try:
				value = '{:.1f}'.format(float(value)) if name in ('p50_ms', 'p95_ms', 'fps') else value
			except ValueError:
				pass
			values.append(str(value).ljust(width))
		lines.append(''.join(values))
	cpu = metrics['cpu_percent'] if metrics is not None and metrics['cpu_percent'] is not None else None
	rss = metrics['rss_mb'] if metrics is not None and metrics['rss_mb'] is not None else None
	if cpu is not None and rss is not None:
		lines.append('')
		lines.append('server  cpu {:.0f} %   ram {:.0f} MB'.format(cpu[0], rss[0]))
	return '\n'.join(lines)
