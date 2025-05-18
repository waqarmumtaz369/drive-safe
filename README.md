# Violation Detection System

## Overview
This project is a real-time Violation Detection System designed to detect seatbelt and phone usage violations using computer vision on the OAK-D CM4 device (powered by Raspberry Pi OS and DepthAI). The system leverages AI models to process video streams from camera or video files, providing live detection and a user-friendly interface for monitoring and reviewing violations.

## Features
- **Real-time detection** of seatbelt usage and phone-in-hand violations.
- **Optimized for OAK-D CM4** (Raspberry Pi Compute Module 4) with DepthAI hardware acceleration.
- **User Interface** built with Tkinter for easy operation and visualization.
- **Supports both live camera and video file input**.
- **Centralized configuration** for easy tuning of thresholds and model paths.
- **Violation queue logic** to reduce false positives by requiring consistent detection over time.
- **Visual feedback** with bounding boxes and detection images.

## System Architecture
- **DepthAI Pipeline**: Runs YOLOv8 for person/phone detection and a custom seatbelt classifier.
- **Python Application**: Handles video input, detection logic, and UI.
- **Tkinter UI**: Allows users to select video/camera, view detections, and review violation events.

## Requirements
- OAK-D CM4 device (with camera)
- Raspberry Pi OS (or compatible Linux)
- Python 3.7+
- DepthAI Python SDK
- OpenCV, NumPy, Pillow, Tkinter

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Connect the OAK-D CM4 to your Raspberry Pi.
2. Place the required model files in the `models/` directory (see below).
3. Run the application:
```bash
python main.py
```
4. Use the UI to select a video file or start the camera.
5. Violations will be detected and displayed in real time.

## Model Files
- `models/yolov8n_coco_416x416_openvino_2022.1_8shave.blob` (YOLOv8 for person/phone)
- `models/seatbelt_nchw.blob` (Seatbelt classifier)

## Configuration
All key parameters (thresholds, model paths, UI colors, etc.) are set in `config.py` for easy adjustment.

## Project Structure
- `main.py` - Entry point, runs the detection loop and UI
- `model_loader.py` - Loads and configures DepthAI pipeline
- `detectors.py` - Detection logic for seatbelt and phone
- `detection_ui.py` - Tkinter-based user interface
- `visualization.py` - Drawing and image utilities
- `config.py` - Centralized configuration
- `project_utils.py` - Helper functions
- `models/` - Model files
- `images/` - UI images
- `sample/` - Sample video files

## Business Value
- **Safety Compliance**: Automates detection of critical road safety violations.
- **Edge AI**: Runs entirely on-device, no cloud required.
- **Scalable**: Can be deployed in vehicles, checkpoints, or monitoring stations.

## License
This project is provided for demonstration and PoC purposes. Contact the Applied Innovation Lab Team for licensing or commercial use.
