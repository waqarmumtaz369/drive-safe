import cv2
import numpy as np
import tensorflow as tf
import config

def list_available_cameras(max_cameras=3):
    """Lists all available camera devices by trying to open each one."""
    print("Checking for available cameras...")
    available_cameras = []
    
    # Try camera indices 0-max_cameras
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            pass
        else:
            print(f"Camera ID {i} is available")
            available_cameras.append(i)
        cap.release()
    
    if not available_cameras:
        print("No cameras found!")
    else:
        print(f"Found {len(available_cameras)} camera(s): {available_cameras}")
        print("Use --camera_id [number] to select a specific camera")
    
    return available_cameras

def prediction_func(img, predictor):
    """Runs the seatbelt prediction on a BGR image crop."""
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_normalized = (img_resized / 127.5) - 1
    img_expanded = tf.expand_dims(img_normalized, axis=0)
    pred = predictor.predict(img_expanded, verbose=0)
    index = np.argmax(pred)
    class_name = config.CLASS_NAMES_SEATBELT.get(index, "Unknown")
    confidence_score = pred[0][index]
    return class_name, confidence_score
