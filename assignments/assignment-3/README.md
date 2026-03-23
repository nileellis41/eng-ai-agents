# Assignment 3 — UAV Drone Detection and Tracking

## Output Tracking Videos

**Video 1:**
[drone_video_1_tracked](https://www.youtube.com/watch?v=YOUR_VIDEO_1_ID)

**Video 2:**
[drone_video_2_tracked](https://www.youtube.com/watch?v=YOUR_VIDEO_2_ID)

---

## HuggingFace Dataset

Detection frames (Parquet format): [naenile40/drone-detections](https://huggingface.co/datasets/naenile40/drone-detections)

---

## Dataset Choice

### Primary — Seraphim Drone Detection Dataset (HuggingFace)

**Source:** `lgrzybowski/seraphim-drone-detection-dataset`
**License:** CC BY 4.0
**Size:** 83,483 images (75,134 train / 8,349 test)
**Format:** YOLO (640×640, single class: drone)

This dataset was selected as the primary training source because it is the largest and most carefully curated open-source drone detection dataset publicly available. It was assembled from 23 independent source datasets and processed through a custom cleaning pipeline including exact-duplicate removal, near-duplicate filtering via perceptual hashing, and resolution standardization to 640×640. The single-class annotation (drone only) aligns directly with the assignment objective.

The dataset includes a size taxonomy based on COCO-style bounding box area buckets — tiny (below 16×16 px), small, medium, and large — which directly informed Kalman filter noise tuning, since small and tiny drones are significantly harder to track consistently across frames.

### Supplementary — drones_new (Roboflow Universe)

**Source:** `tracker-qjlj1/drones_new` via Roboflow Universe
**License:** CC BY 4.0
**Size:** ~9,500 images
**Format:** YOLOv8

Added to supplement Seraphim with additional visual diversity. This dataset includes drone images captured alongside confusable aerial objects (birds, helicopters, planes), which improves the detector's ability to reduce false positives in cluttered sky backgrounds — a known failure mode for drone detection systems.

### Note on VisDrone

VisDrone was explicitly considered and excluded. Despite its name, VisDrone is a dataset of objects detected *from* drone-mounted cameras (pedestrians, vehicles, bicycles) — not a dataset for detecting drones themselves. Using it would have trained the model on the wrong task entirely.

### Final Merged Dataset

| Split | Images |
|-------|--------|
| Train | 80,363 |
| Val   | 16,706 |
| Test  |  9,412 |

---

## Detector Configuration

**Architecture:** YOLOv8n (nano)
**Pretrained weights:** `yolov8n.pt` (COCO pretrained, fine-tuned on merged drone dataset)
**Training:** 10 epochs, image size 640×640, batch size 16, device: NVIDIA Tesla T4 GPU
**Training subset:** 15,000 images sampled from the merged train split (random seed 42)
**Inference confidence threshold:** 0.25
**Framework:** Ultralytics 8.4.25, PyTorch 2.10.0, CUDA 13.0

### Final Training Metrics (Epoch 10)

| Metric | Value |
|--------|-------|
| Precision | 0.845 |
| Recall | 0.680 |
| mAP@50 | 0.765 |
| mAP@50-95 | 0.417 |

The pipeline processes all `.mp4` files found in a given input directory, making it fully generalizable beyond the two provided test videos.

---

## Kalman Filter State Design

### State Vector

The tracker uses a 4-dimensional state vector representing 2D position and velocity of the bounding box center:

```
x = [cx, cy, vx, vy]
```

where `cx, cy` is the estimated center of the drone in pixel coordinates and `vx, vy` is the estimated velocity in pixels per frame.

### Measurement Vector

Only position is directly observed from the detector output:

```
z = [cx, cy]
```

### Motion Model

A constant velocity model is used as the state transition matrix:

```
F = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
```

The measurement matrix observes only position:

```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

### Noise Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Measurement noise R | 10.0 × I₂ | Accounts for detector bounding box jitter across frames |
| Process noise Q (position) | 0.1 | Drones move smoothly in short intervals |
| Process noise Q (velocity) | 1.0 | Velocity can change more abruptly due to maneuvers |
| Initial covariance P | 50.0 × I₄ | High uncertainty at track initialization |

### Detection Association

At each frame the filter predicts the next state. If detections are present, the detection closest to the predicted center is selected via nearest-neighbor association, provided it falls within a 150-pixel radius. Detections outside this radius are treated as missed rather than associated with the current track.

---

## Handling Missed Detections

The tracker maintains a `miss_count` that increments each frame the detector produces no associated detection. During missed frames:

- The Kalman filter continues to **predict** forward using its motion model
- The predicted position is rendered as an orange circle on the output frame labeled "predicted"
- If `miss_count` exceeds **5 consecutive frames**, the track is terminated and the tracker re-initializes on the next available detection

This design maintains trajectory continuity through brief occlusions, fast motion blur, and frames where the drone falls below the detector's confidence threshold.

---

## Failure Cases

**Small and distant drones.** Detector confidence drops significantly when the drone occupies fewer than 16×16 pixels in the frame. The tracker relies on Kalman prediction in these cases, which can drift if the drone changes direction rapidly.

**Background clutter.** Sky backgrounds with clouds, birds, or other aerial objects occasionally trigger false positives. The multi-class supplementary training data (drones_new) partially mitigates this but does not eliminate it entirely.

**Track re-initialization.** When a track terminates after 5 missed frames and the drone reappears, the tracker re-initializes from scratch, creating a discontinuity in the trajectory polyline. A more robust system would apply Hungarian algorithm matching across multiple candidate tracks to handle re-identification.

**Fixed camera assumption.** The Kalman filter assumes a stationary camera. If the recording device is moving, the observed pixel-space motion of the drone combines drone motion and camera motion, inflating apparent velocity and degrading tracking accuracy.

---

## Repository Structure

```
assignment-3/
├── dataset_pipeline.py         # Local dataset download and merge pipeline
├── drone_dataset/              # Merged YOLO dataset (local)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
├── runs/
│   └── detect/
│       └── drone_runs/
│           └── train1/
│               └── weights/
│                   └── best.pt
├── videos/                     # Downloaded test videos
├── frames/                     # Extracted frames at 5fps
├── detections/                 # Frames with at least one drone detection
├── output_videos/              # Final tracked output videos
├── submission_note.txt         # Late submission explanation
└── README.md
```

---

## Development Environment

This project was executed in Google Colaboratory with a Tesla T4 GPU (15GB VRAM). The dataset pipeline was built and validated locally on Windows before being migrated to Colab for model training, inference, and video rendering. The migration was a deliberate engineering decision to meet the computational demands of fine-tuning on 80k+ images — training on CPU would have required an estimated 4–8 hours for a single run.
