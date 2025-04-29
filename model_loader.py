import torch
import tensorflow as tf
import numpy as np
import cv2 # Only needed if preprocessing is done here, otherwise remove
import os
import config # Import constants from config.py

def load_seatbelt_model_and_labels():
    """Loads the Keras seatbelt classification model and its labels."""
    if not os.path.exists(config.SEATBELT_MODEL_PATH):
        print(f"Error: Seatbelt model file not found at {config.SEATBELT_MODEL_PATH}")
        exit()
    if not os.path.exists(config.SEATBELT_LABELS_PATH):
        print(f"Error: Seatbelt labels file not found at {config.SEATBELT_LABELS_PATH}")
        exit()

    try:
        seatbelt_model = tf.keras.models.load_model(config.SEATBELT_MODEL_PATH, compile=False)
        with open(config.SEATBELT_LABELS_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print("Seatbelt model and labels loaded successfully.")
        return seatbelt_model, class_names
    except Exception as e:
        print(f"Error loading seatbelt model or labels: {e}")
        exit()

def load_yolo_model(model_path):
    """Loads a YOLOv5 model from the specified path."""
    if not os.path.exists(model_path):
        print(f"Error: YOLO model file not found at {model_path}")
        exit()
    try:
        # Load the YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False) # force_reload=False is often faster
        print(f"YOLO model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        exit()

def load_models():
    """Loads all required models."""
    print("Loading models...")
    person_model = load_yolo_model(config.PERSON_MODEL_PATH)
    phone_model = load_yolo_model(config.PHONE_MODEL_PATH)
    seatbelt_model, _ = load_seatbelt_model_and_labels() # Labels are also in config now
    print("All models loaded.")
    return person_model, phone_model, seatbelt_model
