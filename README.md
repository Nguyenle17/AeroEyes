# AeroEyes – Spatio-Temporal Object Localization with Drones

This repository contains the implementation for the **AeroEyes** challenge, focusing on spatio-temporal localization of a target object in drone-captured videos using deep learning.

The proposed system combines:
- **YOLO** for real-time object detection
- **ResNet50** for feature embedding
- **Cosine similarity** for instance-level matching
- **DeepSORT** for temporal tracking

The pipeline is designed to be efficient and suitable for deployment on **Jetson-based drones**.

---

## 1. Project Overview

### Problem Definition
Given:
- Three reference images of a target object
- A drone video scanning a large area

The goal is to determine **when and where** the target object appears in the video by predicting bounding boxes for each detected frame.

This is a **spatio-temporal localization** problem evaluated using the **Spatio–Temporal IoU (ST-IoU)** metric.

---

## 2. Repository Structure

aeroeyes/
├── README.md
├── requirements.txt
├── dataset/
│ ├── samples/
│ └── annotations/
├── frames/
│ └── <video_id>/
├── yolo_dataset/
│ ├── images/
│ └── labels/
├── weights/
│ └── yolo.pt
├── scripts/
│ ├── extract_frames.py
│ ├── prepare_yolo_dataset.py
│ ├── inference.py
└── results/
└── predictions.json


---

## 3. Environment Setup

### 3.1 Python Version


Python 3.9+


### 3.2 Create Virtual Environment

python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

### 3.3 Install Dependencies

- pip install -r requirements.txt

## 4. Dataset Preparation

### 4.1 Dataset Structure
dataset/
├── samples/
│   ├── drone_video_001/
│   │   ├── object_images/
│   │   │   ├── img_1.jpg
│   │   │   ├── img_2.jpg
│   │   │   └── img_3.jpg
│   │   └── drone_video.mp4
│   └── ...
└── annotations/
    └── annotations.json

### 4.2 Extract Frames from Videos
python scripts/extract_frames.py \
  --video_root dataset/samples \
  --save_root frames/

## 5. YOLO Dataset Preparation

- Convert annotations to YOLO format and split into train/validation sets:

python scripts/prepare_yolo_dataset.py \
  --annotation_path dataset/annotations/annotations.json \
  --dataset_frames frames/ \
  --save_path yolo_dataset/

## 6. YOLO Training (Optional)

- Fine-tune YOLO on the prepared dataset:

yolo detect train \
  data=yolo_dataset.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640


The trained model should be saved under:

weights/yolo.pt

## 7. Inference Pipeline

Run inference on extracted frames using:

YOLO detection

DeepSORT tracking

ResNet50 embedding

Cosine similarity matching

python scripts/inference.py \
  --frames_root frames/ \
  --weights weights/yolo.pt \
  --output results/predictions.json

## 8. Output Format

The prediction file follows the official AeroEyes submission format:

[
  {
    "video_id": "drone_video_001",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355}
        ]
      }
    ]
  }
]

## 9. Notes

External connectivity (cloud inference) is not used

All models run locally using PyTorch

Large files (datasets, weights) are not committed

This repository is provided for reproducibility and evaluation purposes only