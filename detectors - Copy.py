import cv2
import numpy as np
import config
from project_utils import prediction_func

# Add a global dictionary to track phone detections across frames for smoothing
phone_detection_history = {}
MAX_HISTORY_FRAMES = 5  # Track detections for N frames for stability

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
    global phone_detection_history
    detection_results = []
    
    # Convert to RGB for YOLO models
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect persons with custom model
    person_results = person_model(img_rgb)
    person_detections = person_results.xyxy[0].cpu().numpy()
    
    # Detect phones with YOLOv5s model - use a higher confidence for the whole frame scan
    phone_results = phone_model(img_rgb)
    phone_detections = phone_results.xyxy[0].cpu().numpy()
    
    # Log all potential phone detections for debugging
    print("\n--- Phone Detections in Frame ---")
    for det in phone_detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 67:  # COCO class 67 is 'cell phone'
            print(f"  Phone detected: confidence={conf:.4f}, box=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Filter phones (class index 67 in COCO dataset is 'cell phone')
    phones = []
    for det in phone_detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Classes 67 (cell phone), 73 (laptop), 65 (remote) can all be mistaken for phones in cars
        # Adjust these if the model is detecting other objects as phones
        if (int(cls) == 67 and conf >= config.THRESHOLD_PHONE):
            # Store phones with their confidence scores
            phones.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
    
    # Process each detected person
    for person_idx, person_det in enumerate(person_detections):
        px1, py1, px2, py2, p_conf, _ = person_det
        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
        person_id = f"person_{person_idx}"  # Simple ID for tracking

        # Ensure coordinates are valid before cropping
        crop_py1, crop_py2 = max(0, py1), min(frame.shape[0], py2)
        crop_px1, crop_px2 = max(0, px1), min(frame.shape[1], px2)
        person_crop_bgr = frame[crop_py1:crop_py2, crop_px1:crop_px2]

        seatbelt_status = "Unknown"
        seatbelt_score = 0.0
        phone_detected = False
        phone_box = None
        best_phone_conf = 0.0  # Track best phone confidence

        # Check if the crop is valid for seatbelt prediction
        if person_crop_bgr.shape[0] > 0 and person_crop_bgr.shape[1] > 0:
            seatbelt_status, seatbelt_score = prediction_func(person_crop_bgr, seatbelt_predictor)
            
            # Detect phones specifically in the person's crop with a lower threshold
            # This helps detect phones that might be missed in the whole frame
            try:
                person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
                phone_results_crop = phone_model(person_crop_rgb)
                
                # Check for phones in crop with lower threshold for close-up detection
                lower_threshold = config.THRESHOLD_PHONE * 0.7
                
                for det in phone_results_crop.xyxy[0].cpu().numpy():
                    _, _, _, _, conf, cls = det
                    if int(cls) == 67 and conf > lower_threshold:
                        phone_detected = True
                        best_phone_conf = max(best_phone_conf, float(conf))
                        print(f"  Phone in person crop: confidence={conf:.4f}, person_idx={person_idx}")
                        break
            except Exception as e:
                print(f"Error in person crop phone detection: {e}")

        # Check for phones near this person (from whole-frame detection)
        for phone_box_coords in phones:
            phx1, phy1, phx2, phy2, ph_conf = phone_box_coords
            phone_center_x = (phx1 + phx2) / 2
            phone_center_y = (phy1 + phy2) / 2
            
            # More flexible matching - check if phone is near the person
            # Check if phone center is within the person's bounding box (standard check)
            if (phone_center_x >= px1 and phone_center_x <= px2 and
                phone_center_y >= py1 and phone_center_y <= py2):
                phone_detected = True
                phone_box = [phx1, phy1, phx2, phy2]
                best_phone_conf = max(best_phone_conf, ph_conf)
                print(f"  Phone matched to person {person_idx}: confidence={ph_conf:.4f} (center inside)")
                break
                
            # Also check for proximity (phone may be held at edge of bounding box)
            # Calculate how much the phone bounding box overlaps with person bbox
            person_area = (px2 - px1) * (py2 - py1)
            overlap_x1 = max(px1, phx1)
            overlap_y1 = max(py1, phy1)
            overlap_x2 = min(px2, phx2)
            overlap_y2 = min(py2, phy2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:  # There is overlap
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                phone_area = (phx2 - phx1) * (phy2 - phy1)
                
                # If significant overlap or phone is mostly inside person box
                if (overlap_area > 0.3 * phone_area or 
                    overlap_area > 0.1 * person_area):
                    phone_detected = True
                    phone_box = [phx1, phy1, phx2, phy2]
                    best_phone_conf = max(best_phone_conf, ph_conf)
                    print(f"  Phone matched to person {person_idx}: confidence={ph_conf:.4f} (overlap area)")
                    break

        # Apply temporal smoothing for phone detection
        if person_id not in phone_detection_history:
            phone_detection_history[person_id] = []
        
        # Add current detection to history (True/False and confidence)
        phone_detection_history[person_id].append((phone_detected, best_phone_conf))
        
        # Keep only recent history
        if len(phone_detection_history[person_id]) > MAX_HISTORY_FRAMES:
            phone_detection_history[person_id].pop(0)
        
        # Apply smoothing rule: if majority of recent frames had phone, keep showing it
        if len(phone_detection_history[person_id]) >= 2:
            # Count how many frames had phone detected
            detected_count = sum(1 for detected, _ in phone_detection_history[person_id] if detected)
            
            # If over 40% of recent frames had phone, mark as detected for stability
            if detected_count >= len(phone_detection_history[person_id]) * 0.4:
                phone_detected = True
                # Get the max confidence from recent history
                historical_confs = [conf for detected, conf in phone_detection_history[person_id] if detected and conf > 0]
                if historical_confs:
                    best_phone_conf = max(best_phone_conf, max(historical_confs))
            # Only clear phone if multiple consecutive frames have no detection
            elif detected_count == 0:
                phone_detected = False
        
        detection_results.append({
            'person_box': [px1, py1, px2, py2],
            'seatbelt_status': seatbelt_status,
            'seatbelt_score': float(seatbelt_score),
            'phone_detected': phone_detected,
            'phone_box': phone_box,
            'phone_score': float(best_phone_conf)  # Add phone detection confidence to results
        })
        
        # Clean up person IDs that haven't been seen for a while (every 100 frames or so)
        # This would require frame counting, omitted for simplicity

    return detection_results