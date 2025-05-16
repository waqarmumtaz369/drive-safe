import cv2
import numpy as np
import config
import depthai as dai

def expand_bbox(x1, y1, x2, y2, frame_height, frame_width, expand_percent=0.6):
    """Expands a bounding box by the specified percentage"""
    width = x2 - x1
    height = y2 - y1
    x_expand = width * expand_percent
    y_expand = height * expand_percent
    
    ex1 = max(0, int(x1 - x_expand))
    ey1 = max(0, int(y1 - y_expand))
    ex2 = min(frame_width, int(x2 + x_expand))
    ey2 = min(frame_height, int(y2 + y_expand))
    
    return ex1, ey1, ex2, ey2

def detect_objects_and_seatbelt(frame, device, q_in, q_rgb, q_nn, q_seatbelt_in, q_seatbelt_out):
    """
    Detects only the closest person in the frame, and performs seatbelt and phone detection.
    Uses DepthAI pipeline for inference.
    """
    detection_results = []
    frame_height, frame_width = frame.shape[:2]
    
    # Camera vs video file mode handling
    if q_in is not None:
        # Video file mode - send frame to device
        img = dai.ImgFrame()
        resized_frame = cv2.resize(frame, config.PERSON_MODEL_SIZE)
        img.setData(resized_frame.transpose(2, 0, 1).flatten())
        img.setType(dai.RawImgFrame.Type.BGR888p)
        img.setWidth(resized_frame.shape[1])
        img.setHeight(resized_frame.shape[0])
        img.setTimestamp(dai.Clock.now())
        q_in.send(img)
    
    # Get detections - works for both camera and video modes
    in_nn = q_nn.tryGet()
    
    if in_nn is not None:
        # Process person detections from new model format (1, 1, 200, 7)
        detections_data = np.array(in_nn.getFirstLayerFp16()).reshape((200, 7))
        
        # Find person with largest bounding box (closest)
        max_area = -1
        closest_person_det = None
        phone_detected = False
        phone_box = None
        phone_score = 0.0
        
        # Filter detections with confidence > 0.5
        for detection in detections_data:
            image_id, label, conf, xmin, ymin, xmax, ymax = detection
            
            if conf < 0.5:  # Skip low confidence detections
                continue
                
            if label == 0:  # Person class (0 for person in new model)
                area = (xmax - xmin) * (ymax - ymin)
                if area > max_area:
                    max_area = area
                    closest_person_det = {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'conf': conf
                    }
            # Note: Phone detection will be handled separately through the existing YOLO model

        if closest_person_det is not None:
            # Get person coordinates scaled to frame size
            px1 = int(closest_person_det['xmin'] * frame_width)
            py1 = int(closest_person_det['ymin'] * frame_height)
            px2 = int(closest_person_det['xmax'] * frame_width)
            py2 = int(closest_person_det['ymax'] * frame_height)
            
            # Get expanded box for phone detection
            ex1, ey1, ex2, ey2 = expand_bbox(px1, py1, px2, py2, frame_height, frame_width)
            
            # Perform seatbelt detection
            person_box_crop = frame[py1:py2, px1:px2]
            seatbelt_status = "Unknown"
            seatbelt_score = 0.0
            
            if person_box_crop.shape[0] > 0 and person_box_crop.shape[1] > 0:
                try:
                    # Resize for seatbelt model
                    resized_person = cv2.resize(person_box_crop, config.IMG_SIZE_SEATBELT)
                    
                    # Send to seatbelt classifier
                    seatbelt_img = dai.ImgFrame()
                    seatbelt_img.setData(resized_person.transpose(2, 0, 1).flatten())
                    seatbelt_img.setType(dai.ImgFrame.Type.BGR888p)
                    seatbelt_img.setWidth(config.IMG_SIZE_SEATBELT[0])
                    seatbelt_img.setHeight(config.IMG_SIZE_SEATBELT[1])
                    seatbelt_img.setTimestamp(dai.Clock.now())
                    q_seatbelt_in.send(seatbelt_img)
                    
                    # Get seatbelt result
                    seatbelt_result = q_seatbelt_out.tryGet()
                    if seatbelt_result is not None:
                        seatbelt_data = np.array(seatbelt_result.getFirstLayerFp16())
                        seatbelt_class = int(np.argmax(seatbelt_data))
                        seatbelt_score = float(seatbelt_data[seatbelt_class])
                        
                        if seatbelt_class == 1 and seatbelt_score < 1.0:
                            seatbelt_status = config.CLASS_NAMES_SEATBELT[0]  # Not Worn
                        else:
                            seatbelt_status = config.CLASS_NAMES_SEATBELT[seatbelt_class]
                    
                except Exception as e:
                    print(f"Error in seatbelt detection: {e}")

            # Append detection results
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