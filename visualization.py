import cv2
import config # Import colors

def draw_bounding_box(frame, x1, y1, x2, y2, color, thickness=2):
    """Draws a bounding box on the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def draw_text(frame, x, y, text, color, font_scale=0.7, thickness=2):
    """Draws text on the frame."""
    # Use a filled rectangle background for better visibility
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), color, -1) # Filled background
    cv2.putText(frame, text, (x, y - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA) # Black text

