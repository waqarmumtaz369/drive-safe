import os

# Configuration constants

# Model paths
PERSON_MODEL_PATH = os.path.join('models', 'yolov8n_coco_416x416_openvino_2022.1_8shave.blob')
SEATBELT_MODEL_PATH = os.path.join('models', 'seatbelt.blob')

# Detection thresholds
THRESHOLD_PHONE = 0.20
THRESHOLD_SCORE_SEATBELT = 1.0
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
CLASS_NAMES_SEATBELT = {0: "Not Worn", 1: "Worn"}

# Frame processing
RESIZE_WIDTH = 800