import cv2
import numpy as np
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
    """
    if frame is None:
        return None
        
    # Resize while maintaining aspect ratio
    h, w = frame.shape[:2]
    if w > config.RESIZE_WIDTH:
        h = int(h * (config.RESIZE_WIDTH / w))
        w = config.RESIZE_WIDTH
        frame = cv2.resize(frame, (w, h))
        
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