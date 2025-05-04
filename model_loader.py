import torch
from keras.models import load_model
import config

def load_models():
    """Loads the YOLOv5 and Keras seatbelt predictor models."""
    try:
        # Load seatbelt classifier
        seatbelt_predictor = load_model(config.SEATBELT_MODEL_PATH, compile=False)
        print("Seatbelt predictor loaded.")
        
        # Check if person and phone models are the same
        if config.PERSON_MODEL_PATH == config.PHONE_MODEL_PATH:
            # Load YOLOv5 model once for both person and phone detection
            yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path=config.PERSON_MODEL_PATH, force_reload=False)
            print("YOLOv5 model loaded once for both person and phone detection.")
            return yolo_model, yolo_model, seatbelt_predictor
        else:
            # If models are different, load them separately
            person_model = torch.hub.load("ultralytics/yolov5", "custom", path=config.PERSON_MODEL_PATH, force_reload=False)
            print("Person detection model loaded.")
            
            phone_model = torch.hub.load("ultralytics/yolov5", "custom", path=config.PHONE_MODEL_PATH, force_reload=False)
            print("Phone detection model loaded.")
            
            return person_model, phone_model, seatbelt_predictor
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()
