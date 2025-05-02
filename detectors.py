import cv2
import numpy as np
import config
from project_utils import prediction_func

def expand_bbox(x1, y1, x2, y2, frame_height, frame_width, expand_percent=0.5):
    """
    Expands a bounding box by the specified percentage while respecting frame boundaries.
    Args:
        x1, y1, x2, y2: Original bounding box coordinates
        frame_height, frame_width: Frame dimensions
        expand_percent: Percentage to expand (0.5 = 50% expansion)
    Returns:
        Expanded bounding box coordinates
    """
    width = x2 - x1
    height = y2 - y1
    
    # Calculate expansion amounts (half on each side)
    width_expand = int(width * expand_percent / 2)
    height_expand = int(height * expand_percent / 2)
    
    # Expand coordinates
    new_x1 = max(0, x1 - width_expand)
    new_y1 = max(0, y1 - height_expand)
    new_x2 = min(frame_width, x2 + width_expand)
    new_y2 = min(frame_height, y2 + height_expand)
    
    return new_x1, new_y1, new_x2, new_y2

def detect_objects_and_seatbelt(frame, person_model, phone_model, seatbelt_predictor):
    """
    Detects persons using custom model, phones using YOLOv5s, and seatbelt status.
    Phone detection is performed in the expanded bounding box area.
    Seatbelt detection is performed in the original bounding box area.
    """
    detection_results = []
    frame_height, frame_width = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    person_results = person_model(img_rgb)
    person_detections = person_results.xyxy[0].cpu().numpy()

    for person_idx, person_det in enumerate(person_detections):
        px1, py1, px2, py2, p_conf, _ = person_det
        px1, py1, px2, py2 = map(int, (px1, py1, px2, py2))
        ex1, ey1, ex2, ey2 = expand_bbox(px1, py1, px2, py2, frame_height, frame_width)

        # Seatbelt detection in original bounding box
        person_box_crop_bgr = frame[py1:py2, px1:px2]
        seatbelt_status = "Unknown"
        seatbelt_score = 0.0
        if person_box_crop_bgr.shape[0] > 0 and person_box_crop_bgr.shape[1] > 0:
            seatbelt_status, seatbelt_score = prediction_func(person_box_crop_bgr, seatbelt_predictor)

        # Phone detection in expanded bounding box
        phone_detected = False
        phone_box = None
        phone_score = 0.0
        person_crop_bgr = frame[ey1:ey2, ex1:ex2]
        if person_crop_bgr.shape[0] > 0 and person_crop_bgr.shape[1] > 0:
            try:
                person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
                phone_results = phone_model(person_crop_rgb)
                for det in phone_results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 67 and conf >= config.THRESHOLD_PHONE:
                        phone_detected = True
                        phone_score = float(conf)
                        phone_box = [
                            int(x1) + ex1,
                            int(y1) + ey1,
                            int(x2) + ex1,
                            int(y2) + ey1
                        ]
                        break
            except Exception as e:
                print(f"Error in person crop phone detection: {e}")

        detection_results.append({
            'person_box': [px1, py1, px2, py2],
            'expanded_box': [ex1, ey1, ex2, ey2],
            'seatbelt_status': seatbelt_status,
            'seatbelt_score': float(seatbelt_score),
            'phone_detected': phone_detected,
            'phone_box': phone_box,
            'phone_score': float(phone_score)
        })

    return detection_results