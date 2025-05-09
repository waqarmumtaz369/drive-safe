# Seatbelt and Phone Usage Detection using YOLOv5 and Keras Models

This project detects whether a person is wearing a seatbelt and using a phone. It utilizes a custom YOLOv5 model for person detection, a standard YOLOv5s model for phone detection, and a Keras model for seatbelt classification.

## Prerequisites

Before running the project, ensure that you have Python installed (preferably version 3.8 or higher). You'll also need to install required dependencies and set up a virtual environment for better package management.

1. Set Up a Virtual Environment (Optional but Recommended)
To create a virtual environment, follow these steps:

    ```bash
    # Create virtual environment if you don't have it
    python -m venv venv

    # Activate the virtual environment (Windows)
    .\venv\Scripts\activate

    # Activate the virtual environment (MacOS/Linux)
    source venv/bin/activate
    ```

2. Install Required Packages

    Once the virtual environment is activated, install the necessary packages listed in requirements.txt:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

The main script offers flexible options for real-time detection using either a camera feed or video files:

```bash
# To use your default webcam (camera ID 0)
python main.py

# To use a specific camera (e.g., camera ID 1)
python main.py --camera_id 1

# To list all available cameras on your system
python main.py --list_cameras

# To process a specific video file
python main.py --video sample/test_1.mp4
```

### What the Detector Does

The detection system performs:

- Person detection using the custom YOLOv5 model (best.pt)
- Seatbelt status classification for detected persons
- Phone usage detection using YOLOv5s model
- Visualization with color-coded bounding boxes:
  - Green: Seatbelt worn
  - Red: No seatbelt worn
  - Yellow text indicator when phone usage is detected

During execution, press 'q' to quit the detection process.

## Camera Identification

Not sure which camera to use? The script now uses the OAK-D camera via DepthAI for live detection. Video files are still supported using OpenCV.

## Notes

- The video preview is now shown in a standalone window (OpenCV), not inside the Tkinter UI.
- Detection results are displayed in the Tkinter interface.
- On the initial run, the script may take extra time as it prepares the YOLOv5 models.
- The confidence score threshold for seatbelt detection is set to 0.99, ensuring high accuracy.

## Workflow Overview

1. ### Person Detection

   - The custom YOLOv5 model (best.pt) detects people in each frame
   - Each person is isolated for further analysis

2. ### Seatbelt Classification

   - For each detected person, the Keras model determines if they're wearing a seatbelt
   - Results are displayed with appropriate color coding

3. ### Phone Usage Detection

   - The YOLOv5s model identifies phones in the frame
   - The system checks if a detected phone is associated with a person
   - Phone usage is indicated with yellow text

## Sample Videos

The project includes sample videos for testing:

- sample/test_1.mp4
- sample/test_2.mp4
- sample/test_3.mp4

## Troubleshooting

If you encounter any issues, ensure that:

- Your virtual environment is activated before installing packages or running the script
- The required packages in requirements.txt are installed correctly
- The model files are in their correct locations:
  - Person detection: models/best.pt
  - Seatbelt classification: models/keras_model.h5
  - Phone detection: yolov5s.pt (will use the one in models/ folder if available)
