import cv2
import depthai as dai
import numpy as np
import time
import os
import argparse
from model_loader import load_models
from detectors import detect_objects_and_seatbelt
from visualization import draw_bounding_box
from detection_ui import DetectionUI
import config
from PIL import Image, ImageTk

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Script loaded. Import complete")

def run_detection_loop(video_source, ui):
    # Initialize video source
    if isinstance(video_source, str):
        if video_source == "depthai":
            # Use OAK-D camera directly
            print("Using OAK-D onboard camera")
            use_depthai_camera = True
            source_name = "OAK-D Camera"
            # Create pipeline with onboard camera
            pipeline = load_models(use_onboard_camera=True)
        else:
            # Video file input
            if not os.path.exists(video_source):
                print(f"Error: Video file not found at {video_source}")
                return
            cap = cv2.VideoCapture(video_source)
            source_name = video_source
            use_depthai_camera = False
            # Create pipeline for external input
            pipeline = load_models(use_onboard_camera=False)
    else:
        # OpenCV camera
        cap = cv2.VideoCapture(video_source)
        source_name = f"Camera ID {video_source}"
        use_depthai_camera = False
        if not cap.isOpened():
            print(f"Error: Could not open camera {video_source}.")
            return
        # Create pipeline for external input
        pipeline = load_models(use_onboard_camera=False)
    
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

        # For DepthAI camera, set up direct camera stream
        if use_depthai_camera:
            # Create camera node in the pipeline
            q_cam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            start_time = time.time()
            
            if use_depthai_camera:
                # Get frame directly from the OAK-D camera
                in_rgb = q_cam.tryGet()
                if in_rgb is None:
                    continue
                frame = in_rgb.getCvFrame()  # This will be in RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            else:
                # Get frame from OpenCV camera
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
            
            # Convert frame for Tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            
            # Update UI
            ui.update_video_frame(frame_tk)
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
        # First try to use the OAK-D camera directly through DepthAI
        try:
            print("Attempting to use OAK-D camera through DepthAI")
            run_detection_loop("depthai", ui)
            return
        except Exception as e:
            print(f"Error using DepthAI camera: {e}")
            
        # If DepthAI direct approach fails, fall back to regular OpenCV
        # Try multiple camera indices
        for camera_index in [0, 1, 2]:
            try:
                print(f"Attempting to open camera at index {camera_index}")
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    cap.release()
                    print(f"Successfully found camera at index {camera_index}")
                    run_detection_loop(camera_index, ui)
                    return
                cap.release()
            except Exception as e:
                print(f"Error with camera index {camera_index}: {e}")
        
        # If we get here, no camera was found
        print("No camera could be accessed. Please check your connections and permissions.")
        ui.window.deiconify()  # Show the main window again
        
    def on_exit():
        print("Exiting application...")
        
    # Create and run UI
    ui = DetectionUI(on_video_selected, on_camera_selected, on_exit)
    ui.run()

if __name__ == "__main__":
    main()

