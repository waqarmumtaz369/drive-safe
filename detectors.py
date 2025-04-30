import cv2
import numpy as np
import config
from project_utils import prediction_func

def detect_objects_and_seatbelt(frame, person_model, phone_model, seatbelt_predictor):
    """
    Detects persons using custom model, phones using YOLOv5s, and seatbelt status.

    Args:
        frame: The input frame (BGR format).
        person_model: Loaded custom person detection model.
        phone_model: Loaded YOLOv5s model for phone detection.
        seatbelt_predictor: Loaded Keras seatbelt predictor model.

    Returns:
        A list of dictionaries with detection info for each person.
    """
    detection_results = []
    
    # Convert to RGB for YOLO models
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect persons with custom model
    person_results = person_model(img_rgb)
    person_detections = person_results.xyxy[0].cpu().numpy()
    
    # Detect phones with YOLOv5s model
    phone_results = phone_model(img_rgb)
    phone_detections = phone_results.xyxy[0].cpu().numpy()
    
    # Filter phones (class index 67 in COCO)
    phones = []
    for det in phone_detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 67 and conf >= config.THRESHOLD_PHONE:
            phones.append([int(x1), int(y1), int(x2), int(y2)])
    
    # Process each detected person
    for person_det in person_detections:
        px1, py1, px2, py2, p_conf, _ = person_det
        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

        # Ensure coordinates are valid before cropping
        crop_py1, crop_py2 = max(0, py1), min(frame.shape[0], py2)
        crop_px1, crop_px2 = max(0, px1), min(frame.shape[1], px2)
        person_crop_bgr = frame[crop_py1:crop_py2, crop_px1:crop_px2]

        seatbelt_status = "Unknown"
        seatbelt_score = 0.0
        phone_detected = False
        phone_box = None

        # Check if the crop is valid for seatbelt prediction
        if person_crop_bgr.shape[0] > 0 and person_crop_bgr.shape[1] > 0:
            seatbelt_status, seatbelt_score = prediction_func(person_crop_bgr, seatbelt_predictor)

        # Check for phones near this person
        for phone_box_coords in phones:
            phx1, phy1, phx2, phy2 = phone_box_coords
            phone_center_x = (phx1 + phx2) / 2
            phone_center_y = (phy1 + phy2) / 2
            
            # Check if phone center is within the person's bounding box
            if (phone_center_x >= px1 and phone_center_x <= px2 and
                phone_center_y >= py1 and phone_center_y <= py2):
                phone_detected = True
                phone_box = [phx1, phy1, phx2, phy2]
                break

        detection_results.append({
            'person_box': [px1, py1, px2, py2],
            'seatbelt_status': seatbelt_status,
            'seatbelt_score': float(seatbelt_score),
            'phone_detected': phone_detected,
            'phone_box': phone_box
        })

    return detection_results
