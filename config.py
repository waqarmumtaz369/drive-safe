# Configuration constants

# Model paths
PERSON_MODEL_PATH = 'yolov5s.pt'  # Assuming YOLOv5s for person detection
PHONE_MODEL_PATH = 'models/best.pt' # Path to your trained phone detection model
SEATBELT_MODEL_PATH = 'models/keras_model.h5' # Path to your trained seatbelt classification model
SEATBELT_LABELS_PATH = 'models/labels.txt' # Path to the labels file for the seatbelt model

# Detection thresholds
CONFIDENCE_THRESHOLD_PERSON = 0.45 # Confidence threshold for person detection
CONFIDENCE_THRESHOLD_PHONE = 0.50 # Confidence threshold for phone detection
IOU_THRESHOLD = 0.45            # IoU threshold for Non-Maximum Suppression (NMS)
THRESHOLD_SCORE_SEATBELT = 0.60 # Minimum confidence score to consider seatbelt classification valid

# Image processing
IMG_SIZE_SEATBELT = (224, 224) # Input size for the seatbelt classification model

# Visualization colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_PERSON_BOX = (255, 0, 0) # Blue for person box by default

# Class names (Ensure these match your model's output)
# Example for seatbelt model (adjust based on your labels.txt)
# Assumes labels.txt has "Not Worn" on the first line, "Worn" on the second
CLASS_NAMES_SEATBELT = ["Not Worn", "Worn"]
