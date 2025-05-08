import os

# Configuration constants

# Model paths
PERSON_MODEL_PATH = os.path.join('blob', 'yolov8n_coco_416x416_openvino_2022.1_6shave.blob')
SEATBELT_MODEL_PATH = os.path.join('blob', 'seatbelt.blob')

# Detection thresholds
THRESHOLD_PHONE = 0.5
THRESHOLD_SCORE_SEATBELT = 0.8
RELATIVE_PHONE_AREA_THRESHOLD = 0.05

# Input image sizes
IMG_SIZE_SEATBELT = (224, 224)  # Input size for the seatbelt classification model
YOLO_INPUT_SIZE = (416, 416)    # Input size for YOLO detection model

# Visualization colors (BGR format)
COLOR_GREEN = (0, 255, 0)       # Seatbelt ON
COLOR_RED = (0, 0, 255)         # Seatbelt OFF
COLOR_YELLOW = (0, 255, 255)    # Phone Detected
COLOR_PERSON_BOX = (255, 255, 0) # Default box for person if seatbelt unclear

# Classification labels
CLASS_NAMES_SEATBELT = {0: "No Seatbelt worn", 1: "Seatbelt Worn"}

# Frame processing
RESIZE_WIDTH = 640  # Resize width for the frame if larger than this