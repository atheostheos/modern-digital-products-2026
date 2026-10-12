# ml_speech — speech to text (faster-whisper tiny.en + Silero VAD)

Streams the input audio to `/ws/speech` in short float32 chunks. The server resamples to 16 kHz,
finds speech with Silero VAD, cuts it into utterances and transcribes them with whisper `tiny.en`
(English only). It sends a **partial** guess while someone talks and a **final** sentence after a
pause.

| | |
|---|---|
| Input | `in_audio` — any audio CHOP, any sample rate; several channels are mixed to mono |
| Server | `/ws/speech`, one session per connection; final about 0.3 s after the speaker stops |

## Parameters

| Page | Parameter | Default | Meaning |
|---|---|---|---|
| Speech | Chunk (s) | 0.25 | audio sent per message |
| Speech | Transcript Lines | 10 | lines kept in `out_transcript` |
| Server | Host, Port, Active, Reconnect | | see [README](README.md) |

## Outputs

| Output | Type | Content |
|---|---|---|
| `out_final` | text DAT | last finished sentence |
| `out_partial` | text DAT | current guess while speaking (empty otherwise) |
| `out_transcript` | text DAT | last N finished sentences, one per line |
| `out_speech` | CHOP, 1 sample | `speaking` (0/1), `latency_ms` (last chunk of the utterance sent → final received), `finals` (count) |
| `out_status` | CHOP | see [README](README.md) |

## Wiring

```
ml_input.out_audio ─▶ in_audio
out_final ─▶ ml_sentiment.in_text, view_sentiment.in_text, view_llm.in_quote
out_partial ─▶ view_speech.in_partial  out_transcript ─▶ view_speech.in_transcript  out_speech ─▶ view_speech.in_speech
```

To feed your own audio, connect any Audio Device In, Audio File In or Audio Stream In CHOP to
`in_audio`. The CHOP must be time-sliced (the default), so that each frame carries only that frame's
new samples.
