import cv2
import numpy as np

def draw_bounding_box(frame, x1, y1, x2, y2, color, thickness=2, text=None):
    """Draw a bounding box on the frame with optional text."""
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if text:
        cv2.putText(frame, text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

def draw_text(frame, x, y, text, color, font_scale=0.7, thickness=2):
    """Draws text on the frame."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

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

