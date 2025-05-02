import cv2

def draw_bounding_box(frame, x1, y1, x2, y2, color, thickness=2):
    """Draws a bounding box on the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def draw_text(frame, x, y, text, color, font_scale=0.7, thickness=2):
    """Draws text on the frame."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

