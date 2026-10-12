# Serving latency (up to 30 single requests per model, 30 s budget; fit on the last 730 days; warm val users)

| model | fit_seconds | requests | single_p50_ms | single_p95_ms | single_max_ms | batch_users | batch_seconds | batch_ms_per_user |
|---|---|---|---|---|---|---|---|---|
| trending | 3.44 | 30 | 9.29 | 10.38 | 10.64 | 1000 | 0.0690 | 0.0692 |
| also_bought | 8.14 | 3 | 17,377 | 42,238 | 45,000 | 1000 | 12.46 | 12.46 |
| user_knn | 28.93 | 30 | 4.37 | 6.30 | 15.25 | 1000 | 0.0210 | 0.0212 |
| als | 4.91 | 30 | 4.94 | 6.23 | 8.69 | 1000 | 0.0690 | 0.0692 |
| content | 15.06 | 30 | 33.22 | 35.17 | 91.86 | 1000 | 1.87 | 1.87 |
| hybrid (precomputed test lists, lookup only) | 0.0000 | 30 | 7.29e-04 | 0.0028 | 0.0050 | - | - | - |
