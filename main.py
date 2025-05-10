import cv2
import depthai as dai
import numpy as np
import time
import os
import argparse
from model_loader import load_models
from detectors import detect_objects_and_seatbelt
from visualization import draw_bounding_box, draw_fps
from detection_ui import DetectionUI
import config
from PIL import Image, ImageTk
from project_utils import resize_image

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Script loaded. Import complete")

def run_detection_loop(video_source, ui):
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
    
    # Open video window in UI
    ui.open_video_window()
    
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
            
            # Resize frame if needed
            frame = resize_image(frame)
            
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
                
                draw_bounding_box(frame, px1, py1, px2, py2, box_color)
                cv2.putText(frame, seatbelt_text, (px1, py1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.85, box_color, 3)
                
                # Draw phone detection if found
                if phone_detected and det.get('phone_box'):
                    bx1, by1, bx2, by2 = det['phone_box']
                    draw_bounding_box(frame, bx1, by1, bx2, by2, config.COLOR_YELLOW)
                    phone_text = f"Phone ({phone_score:.2f})"
                    cv2.putText(frame, phone_text, (bx1, by1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.85, config.COLOR_YELLOW, 3)
            
            # Calculate and show FPS
            frame_time = time.time() - start_time
            fps = 1/frame_time if frame_time > 0 else 0
            draw_fps(frame, fps)
            
            # Convert frame for Tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            
            # Update UI with frame size
            ui.update_video_frame(frame_tk, frame.shape[1], frame.shape[0])
            ui.update_detections(detections)
            
            # Process Tkinter events
            ui.video_window.update()
            
            if not ui.video_window.winfo_exists():
                break
                
        cap.release()

def main():
    def on_video_selected(video_path):
        run_detection_loop(video_path, ui)
        
    def on_camera_selected():
        run_detection_loop(0, ui)  # Use default camera (0)
        
    def on_exit():
        print("Exiting application...")
        
    # Create and run UI
    ui = DetectionUI(on_video_selected, on_camera_selected, on_exit)
    ui.run()

if __name__ == "__main__":
    main()
