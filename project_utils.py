import cv2
import numpy as np
import tensorflow as tf
import config

def list_available_cameras(max_cameras=3):
    """Lists all available camera devices. (Note: OAK-D camera is now used via DepthAI, not OpenCV)"""
    print("OAK-D camera is used via DepthAI. Listing USB cameras is deprecated in this mode.")
    return [0]  # Always return 0 for compatibility

def prediction_func(img_array, model):
    """
    Process an image through the seatbelt detection model and return the prediction.
    Adapted for DepthAI model output format.
    """
    try:
        # For DAI model, the preprocessing is done in detectors.py
        # This function now just processes the model output
        predictions = model
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        return config.CLASS_NAMES_SEATBELT[predicted_class], confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Unknown", 0.0

def preprocess_frame(frame, target_size=(416, 416)):
    """
    Preprocess frame for DepthAI model input.
    Always resize if width > config.RESIZE_WIDTH, maintaining aspect ratio.
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > config.RESIZE_WIDTH:
        new_w = config.RESIZE_WIDTH
        new_h = int(h * (config.RESIZE_WIDTH / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def resize_image(frame):
    """
    Resize the frame if width > config.RESIZE_WIDTH, maintaining aspect ratio, and convert to RGB.
    Args:
        frame: Input image in BGR format (OpenCV).
    Returns:
        The processed image in BGR format.
    """
    height, width = frame.shape[:2]
    if width <= config.RESIZE_WIDTH:
        resized = frame
    else:
        new_width = config.RESIZE_WIDTH
        new_height = int(height * (config.RESIZE_WIDTH / width))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized