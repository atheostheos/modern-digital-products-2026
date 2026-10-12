# ml_server — server metrics

Subscribes to `/ws/metrics`, which the server pushes every 0.5 s. It sends no input. Use it to
watch latency, throughput, dropped frames and resource use.

| | |
|---|---|
| Inputs | none |
| Parameters | Server page only (Host, Port, Active, Reconnect) |

## Outputs

| Output | Type | Content |
|---|---|---|
| `out_nodes` | table DAT | `node requests processed dropped errors p50_ms p95_ms fps`, one row per model |
| `out_metrics` | CHOP, 1 sample | `<node>_p50_ms`, `<node>_p95_ms`, `<node>_fps`, `<node>_dropped`, `<node>_errors` for gesture, classify, speech, sentiment and llm, plus `cpu_percent`, `rss_mb`, `threads` for the server process |
| `out_status` | CHOP | see [README](README.md) |

`p50_ms` / `p95_ms` are model inference times over a rolling window. `dropped` counts requests
replaced by a newer one before the model reached them (latest-wins). The same data is available
from `GET /metrics`.

## Wiring

```
out_metrics ─▶ view_metrics.in_metrics      out_nodes ─▶ view_metrics.in_nodes
```

To record a benchmark, feed `out_metrics` into a Trail CHOP, or log `GET /metrics` from a terminal.
