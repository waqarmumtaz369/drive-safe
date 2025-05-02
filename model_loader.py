import torch
from keras.models import load_model
import config

def load_models():
    """Loads the YOLOv5 and Keras seatbelt predictor models."""
    try:
        # Load seatbelt classifier
        seatbelt_predictor = load_model(config.SEATBELT_MODEL_PATH, compile=False)
        print("Seatbelt predictor loaded.")
        
        # Load person detection model (custom)
        person_model = torch.hub.load("ultralytics/yolov5", "custom", path=config.PERSON_MODEL_PATH, force_reload=True)
        print("Person detection model loaded.")
        
        # Load phone detection model (use local yolov5s.pt file)
        phone_model = torch.hub.load("ultralytics/yolov5", "custom", path=config.PHONE_MODEL_PATH, force_reload=True)
        print("Phone detection model loaded.")
        
        return person_model, phone_model, seatbelt_predictor
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()
