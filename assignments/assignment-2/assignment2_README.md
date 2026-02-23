# Assignment 2: Image-to-Video Semantic Retrieval

**Course:** AI Foundations — Spring 2026  
**Dataset:** [naenile40/rav4-detections](https://huggingface.co/datasets/naenile40/rav4-detections)  
**Video:** [Toyota RAV4 2026 Exterior Review](https://www.youtube.com/watch?v=YcvECxtXoxQ)

---

## Overview

A visual retrieval system that answers the following query:

> Given a single image of a car exterior component, retrieve the video clip(s) in which that component appears.

The system operates through detected semantic structure — no manual labels, hardcoded timestamps, or query-specific heuristics are used.

---

## Files

| File | Description |
|------|-------------|
| `assignment2.ipynb` | Full pipeline notebook (run in Google Colab with T4 GPU) |
| `detections.parquet` | Video detection index — 1,270 detections, 17 classes, CLIP embeddings |
| `retrieval_results.parquet` | Retrieval results — query images matched to temporal video segments |
| `assignment2_report.docx` | 4-page report covering detector choice, sampling, matching logic, and failure cases |

---

## Pipeline

```
Video → Frame Extraction (0.5 fps) → YOLOv8l Detection → CLIP Embeddings → detections.parquet
                                                                                      ↓
Query Image → YOLOv8l Detection → CLIP Embedding → Two-Stage Retrieval → Temporal Segments
```

**Stage 1 — Class label matching:** detected part class in query is matched against the video index  
**Stage 2 — CLIP re-ranking:** cosine similarity between query crop embedding and video detection embeddings filters and ranks segments

---

## Model

- **Detector:** YOLOv8l-seg fine-tuned on [Ultralytics carparts-seg](https://docs.ultralytics.com/datasets/segment/carparts-seg/) (30 epochs, AdamW)
- **Embeddings:** CLIP ViT-B/32 — 512-dim L2-normalised embedding per detection crop
- **Classes detected (17):** back_bumper, back_glass, back_left_door, back_left_light, back_light, back_right_door, front_bumper, front_glass, front_left_door, front_right_door, front_right_light, hood, left_mirror, right_mirror, tailgate, trunk, wheel

---

## Reproducing Results

Open `assignment2.ipynb` in [Google Colab](https://colab.research.google.com) and set runtime to **T4 GPU**, then run all cells in order.

```
Cell 1  — Verify GPU
Cell 2  — Install dependencies
Cell 3  — Download video + extract frames
Cell PRE— Download yolov8l weights + install CLIP
Cell 4a — Fine-tune YOLOv8l on carparts dataset
Cell 4b — Run detection on all frames → detections.parquet
Cell 5  — Two-stage retrieval → retrieval_results.parquet
Cell 6  — Spot-check results on YouTube
Cell 7  — Upload to Hugging Face
Cell 8  — Download output files
```

---

## Verifying a Result

Each row in `retrieval_results.parquet` includes a `youtube_verify_url` field. Open it in a browser to confirm the segment contains the queried component. Example:

```
https://www.youtube.com/embed/YcvECxtXoxQ?start=120&end=165
```
