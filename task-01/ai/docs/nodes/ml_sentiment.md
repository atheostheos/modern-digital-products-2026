# ml_sentiment — text sentiment (DistilBERT SST-2, Hugging Face)

Sends the input DAT's text to `/ws/sentiment` whenever it changes, at most **Max Rate** times per
second. It returns positive/negative probabilities and a single `valence` value in −1…1.
`POST /api/sentiment` answers the same request over REST.

| | |
|---|---|
| Input | `in_text` — any DAT (its whole text is sent; truncated to 512 tokens) |
| Server | `/ws/sentiment`, about 13 ms p50 |

## Parameters

| Page | Parameter | Default | Meaning |
|---|---|---|---|
| Sentiment | Max Rate (msg/s) | 4 | throttle for fast-changing text |
| Sentiment | Send Now | pulse | send the current text even if it did not change |
| Server | Host, Port, Active, Reconnect | | see [README](README.md) |

## Outputs

| Output | Type | Content |
|---|---|---|
| `out_sentiment` | CHOP, 1 sample | `positive`, `negative` (0–1), `valence` = positive − negative |
| `out_label` | text DAT | `positive` or `negative` |
| `out_status` | CHOP | see [README](README.md) |

## Wiring

```
ml_speech.out_final ─▶ in_text           (or a Text DAT you type into, or a Web/OSC/chat DAT)
out_sentiment ─▶ view_sentiment.in_sentiment, view_llm.in_sentiment
out_label     ─▶ view_sentiment.in_label
```

`valence` works well with a Lag CHOP and a Lookup TOP or ramp, for example as a colour grade driven
by mood.
