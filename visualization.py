import cv2
import numpy as np
import config

def draw_bounding_box(frame, x1, y1, x2, y2, color, thickness=3, text=None):
    """Draw a bounding box on the frame with optional text."""
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if text:
        cv2.putText(frame, text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, thickness)

def draw_text(frame, x, y, text, color, font_scale=0.85, thickness=3):
    """Draws text on the frame."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_fps(frame, fps):
    """Draws FPS counter on the frame."""
    fps_text = f"FPS: {fps:.1f}"
    draw_text(frame, 10, 30, fps_text, config.COLOR_GREEN)

def draw_detection_info(frame, detection):
    """Draws all detection information on the frame."""
    # Draw person box
    x1, y1, x2, y2 = detection['person_box']
    seatbelt_status = detection['seatbelt_status']
    seatbelt_score = detection['seatbelt_score']
    
    # Determine box color based on seatbelt status
    if seatbelt_status == "Worn" and seatbelt_score >= config.THRESHOLD_SCORE_SEATBELT:
        box_color = config.COLOR_GREEN
    elif seatbelt_status == "Not Worn":
        box_color = config.COLOR_RED
    else:
        box_color = config.COLOR_PERSON_BOX
    
    # Draw person box
    draw_bounding_box(frame, x1, y1, x2, y2, box_color)
    
    # Draw seatbelt status
    seatbelt_text = f"Seatbelt {seatbelt_status}"
    draw_text(frame, x1 + 5, y1 + 50, seatbelt_text, box_color)
    
    # Draw phone detection if present
    if detection['phone_detected'] and detection['phone_box'] is not None:
        px1, py1, px2, py2 = detection['phone_box']
        phone_score = detection['phone_score']
        draw_bounding_box(frame, px1, py1, px2, py2, config.COLOR_YELLOW)
        phone_text = f"Phone Detected: {phone_score:.2f}"
        draw_text(frame, px1, py1 - 10, phone_text, config.COLOR_YELLOW)

def create_detection_image(frame, x1, y1, x2, y2, width=200):
    """Create a small visualization image of the detection region."""
    try:
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return None
            
        # Calculate height maintaining aspect ratio
        aspect_ratio = crop.shape[0] / crop.shape[1]
        height = int(width * aspect_ratio)
        
        # Resize the crop
        resized = cv2.resize(crop, (width, height))
        return resized
    except Exception as e:
        print(f"Error creating detection image: {e}")
        return None

