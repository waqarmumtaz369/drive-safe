# Violation Detection System - Business & Technical Summary

## Project Purpose
The Violation Detection System is designed to automatically detect seatbelt and phone usage violations in real time using AI-powered computer vision. It is optimized for the OAK-D CM4 platform (Raspberry Pi Compute Module 4 with DepthAI), enabling edge-based, low-latency safety monitoring for vehicles or checkpoints.

## Key Capabilities
- **Real-Time Detection**: Identifies if a driver is not wearing a seatbelt or is using a phone while driving.
- **Edge AI**: All processing is performed on-device, ensuring privacy and low latency.
- **User Interface**: Simple, intuitive UI for operators to monitor live video or review video files.
- **Configurable**: Detection thresholds, model paths, and UI settings are easily adjustable.
- **False Positive Reduction**: Uses a time-based queue system to confirm violations only if detected consistently over several seconds.

## Technical Architecture
- **Hardware**: OAK-D CM4 (DepthAI, Raspberry Pi CM4)
- **Software Stack**:
  - Python 3.7+
  - DepthAI Python SDK
  - OpenCV, NumPy, Pillow, Tkinter
- **AI Models**:
  - YOLOv8 (person/phone detection)
  - Custom seatbelt classifier
- **Pipeline**:
  1. Video input from camera or file
  2. YOLOv8 detects persons and phones
  3. Closest person is analyzed for seatbelt status
  4. Results are queued to confirm violations
  5. UI displays live video, bounding boxes, and violation events

## Business Benefits
- **Improved Safety**: Automates enforcement of seatbelt and phone usage laws.
- **Operational Efficiency**: Reduces need for manual monitoring.
- **Scalable & Flexible**: Can be deployed in vehicles, at checkpoints, or in control rooms.
- **Privacy-Preserving**: No cloud upload; all data processed locally.

## Customization
- **Thresholds and logic** can be tuned in `config.py`.

## Deployment Notes
- Requires OAK-D CM4 hardware and compatible Raspberry Pi OS.
- Place model files in the `models/` directory.
- Install dependencies with `pip install -r requirements.txt`.
- Run with `python main.py`.

## Next Steps for Presentation
- Use this summary to explain both the technical and business value.
- Highlight real-time, on-device AI and privacy benefits.
- Demonstrate the UI and detection results live or with sample videos.

---
*For more details, see the README.md or contact the Applied Innovation Lab Team.*
