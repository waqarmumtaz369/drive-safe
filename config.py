# Configuration constants

# Model paths
PERSON_MODEL_PATH = "models/yolov5s.pt"    # Custom model for person detection
PHONE_MODEL_PATH = 'models/yolov5s.pt' # Path to your trained phone detection model
SEATBELT_MODEL_PATH = "models/keras_model.h5"  # Seatbelt classifier model
SEATBELT_LABELS_PATH = 'models/labels.txt' # Path to the labels file for the seatbelt model

# Detection thresholds
CONFIDENCE_THRESHOLD_PERSON = 0.45 # Confidence threshold for person detection
THRESHOLD_PHONE = 0.15
IOU_THRESHOLD = 0.45            # IoU threshold for Non-Maximum Suppression (NMS)
THRESHOLD_SCORE_SEATBELT = 0.99 # Minimum confidence score to consider seatbelt classification valid

# Add a new config for relative phone area threshold
RELATIVE_PHONE_AREA_THRESHOLD = 0.0500  # 5.00% of person box area (tune as needed)

# Image processing
IMG_SIZE_SEATBELT = (224, 224) # Input size for the seatbelt classification model

# Visualization colors (BGR format)
COLOR_GREEN = (0, 255, 0) # Seatbelt ON
COLOR_RED = (255, 0, 0)   # Seatbelt OFF
COLOR_YELLOW = (0, 255, 255) # Phone Detected
COLOR_PERSON_BOX = (255, 255, 0) # Default box for person if seatbelt unclear

CLASS_NAMES_SEATBELT = {0: "No Seatbelt worn", 1: "Seatbelt Worn"}

RESIZE_WIDTH = 800 # Resize width for the frame if larger than this