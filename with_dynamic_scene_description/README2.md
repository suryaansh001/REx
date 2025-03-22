# YOLO + Dynamic Talk-Back System
A smart vision system that detects objects, describes scenes, and talks back dynamically using YOLO, BLIP, and TTS.

## What It Does
This project combines real-time object detection (YOLO), scene captioning (BLIP), and text-to-speech (TTS) to create an interactive assistant. It:
- Detects objects like chairs, sofas, tables (or whatever YOLO is trained on).
- Generates scene descriptions (e.g., "a room with a table").
- Responds intelligently with dynamic messages (e.g., "I see a chair for the first time!" or "Careful! A sofa is nearby.").


## How It Works
- **YOLO**: Detects objects in real-time from webcam feed.
- **BLIP**: Creates a caption for the entire scene.
- **DynamicResponseGenerator**: Keeps track of what’s seen before and generates smart responses based on detected objects and scene changes.
- **TTS**: Speaks out the responses using `pyttsx3`.

## Setup
1. **Install Dependencies**:
   pip install -r req.txt
