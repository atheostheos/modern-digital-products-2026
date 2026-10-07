# Serving latency (30 single requests, fit on the last 730 days)

| model | fit_seconds | requests | single_p50_ms | single_p95_ms | single_max_ms | batch_users | batch_seconds | batch_ms_per_user |
|---|---|---|---|---|---|---|---|---|
| trending | 3.3400 | 30 | 7.8038 | 8.5810 | 10.3585 | 1000 | 0.0710 | 0.0709 |
| also_bought | 6.9900 | 4 | 9944.3652 | 10584.1878 | 10674.0180 | 1000 | 10.1820 | 10.1817 |
| user_knn | 27.4800 | 30 | 3.1494 | 3.9408 | 21.2494 | 1000 | 0.0180 | 0.0183 |
| als | 4.6000 | 30 | 4.3147 | 4.6375 | 6.2822 | 1000 | 0.0670 | 0.0673 |
| content | 14.5300 | 30 | 33.1811 | 34.6616 | 37.0209 | 1000 | 1.7380 | 1.7382 |
| hybrid (precomputed lookup) | 0.0000 | 30 | 0.0009 | 0.0038 | 0.0106 | None | None | None |
