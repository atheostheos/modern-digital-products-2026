# ml_classify — image classification (MobileNetV3-Small, torchvision)

Sends the input image to `/ws/classify` a few times per second and gets back the top-5 of the
1000 ImageNet classes.

| | |
|---|---|
| Input | `in_video` — any TOP (resized to **Send Size**) |
| Server | `/ws/classify`, about 30 ms p50 on CPU; also `POST /api/classify` with a JPEG body |

## Parameters

| Page | Parameter | Default | Meaning |
|---|---|---|---|
| Classify | Send Rate (fps) | 2 | images sent per second |
| Classify | Send Size (px) | 224 | the model's native input size |
| Server | Host, Port, Active, Reconnect | | see [README](README.md) |

## Outputs

| Output | Type | Content |
|---|---|---|
| `out_top` | table DAT | `rank label score`, 5 rows |
| `out_label` | text DAT | top-1 label |
| `out_classes` | CHOP, 1 sample | `top1_score`, `score1` … `score5` |
| `out_status` | CHOP | see [README](README.md) |

## Wiring

```
ml_input.out_video ─▶ in_video
out_classes ─▶ view_classify.in_classes      out_top ─▶ view_classify.in_top
```

ImageNet labels describe objects (banana, coffee mug, laptop…), not people or hands. Point the
camera at objects to get confident scores.
