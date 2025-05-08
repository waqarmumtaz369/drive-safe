import cv2
import depthai as dai
import numpy as np
import time
import os
import argparse
from model_loader import load_models
from detectors import detect_objects_and_seatbelt
from visualization import draw_bounding_box
import config

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Script loaded. Import complete")

def run_detection_loop(video_source, ui_callback=None):
    # Create and load pipeline
    pipeline = load_models()
    
    # Initialize video source
    if isinstance(video_source, str):
        if not os.path.exists(video_source):
            print(f"Error: Video file not found at {video_source}")
            return
        cap = cv2.VideoCapture(video_source)
        source_name = video_source
    else:
        cap = cv2.VideoCapture(video_source)
        source_name = f"Camera ID {video_source}"
        if not cap.isOpened():
            print(f"Error: Could not open camera {video_source}.")
            return
    print(f"Processing source: {source_name}")
    
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get input/output queues
        q_in = device.getInputQueue(name="frame")
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_seatbelt_in = device.getInputQueue(name="seatbelt_in")
        q_seatbelt_out = device.getOutputQueue(name="seatbelt_out", maxSize=4, blocking=False)

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Finished processing video or cannot read frame from camera.")
                break
            
            # Process frame
            detections = detect_objects_and_seatbelt(
                frame, device, q_in, q_rgb, q_nn, q_seatbelt_in, q_seatbelt_out
            )
            
            # Draw results on frame
            for det in detections:
                px1, py1, px2, py2 = det['person_box']
                seatbelt_status = det['seatbelt_status']
                seatbelt_score = det['seatbelt_score']
                phone_detected = det['phone_detected']
                phone_score = det.get('phone_score', 0.0)
                
                # Draw person box with color based on seatbelt status
                box_color = config.COLOR_PERSON_BOX
                seatbelt_text = f"{seatbelt_status} ({seatbelt_score:.2f})"
                if seatbelt_score >= config.THRESHOLD_SCORE_SEATBELT:
                    if seatbelt_status == config.CLASS_NAMES_SEATBELT[1]:
                        box_color = config.COLOR_GREEN
                    elif seatbelt_status == config.CLASS_NAMES_SEATBELT[0]:
                        box_color = config.COLOR_RED
                else:
                    seatbelt_text = f"No Seatbelt Worn ({seatbelt_score:.2f})"
                
                draw_bounding_box(frame, px1, py1, px2, py2, box_color, thickness=2)
                cv2.putText(frame, seatbelt_text, (px1, py1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                # Draw phone detection if found
                if phone_detected and det.get('phone_box'):
                    bx1, by1, bx2, by2 = det['phone_box']
                    draw_bounding_box(frame, bx1, by1, bx2, by2, config.COLOR_YELLOW, thickness=2)
                    phone_text = f"Phone ({phone_score:.2f})"
                    cv2.putText(frame, phone_text, (bx1, by1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 2)
            
            # Calculate and show FPS
            frame_time = time.time() - start_time
            fps_text = f"FPS: {1/frame_time:.1f}" if frame_time > 0 else "FPS: N/A"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update UI if callback provided
            if ui_callback:
                ui_callback(frame, detections)
            
            # Show frame
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Seatbelt and Phone Detection using DepthAI pipeline.")
    parser.add_argument("--video", type=str, help="Path to the video file. If not provided, webcam 0 is used.")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID to use (default: 0).")
    parser.add_argument("--list_cameras", action="store_true", help="List available cameras and exit.")
    args = parser.parse_args()

    if args.list_cameras:
        print("Listing available cameras is not supported with DepthAI pipeline.")
        exit(0)

    video_source = args.video if args.video else args.camera_id
    run_detection_loop(video_source)

