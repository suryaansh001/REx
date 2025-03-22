# YOLO Smart Interaction
Real-time furniture detection with voice feedback for Airbnb guests.

## Problem
Guests struggle to identify room items. This AI detects furniture (Chair, Sofa, Table) and speaks them aloud.

## Approach
- **Dataset**:  "yolo_furniture" dataset (3 classes: Chair, Sofa, Table) .You may use a dataset with more classes.
- **Model**: Fine-tuned YOLOv8n on custom dataset
- **Features**: Real-time detection, TTS interaction
- **Optimization**: Frame skipping (every 15 frames) for low-resource devices

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train: `python src/train.py`
3. Run: `python src/detect.py`

## Results
- **Training**: mAP@0.5 = 0.995 
- **FPS**: ~25-30 on CPU (inference 31.9ms-39.1ms)
- **Demo**: [https://github.com/terausername/YOLO_Smart_Interaction/blob/main/demo.mp4] (Replace with your link)
t")
