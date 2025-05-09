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
import sys

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Script loaded. Import complete")

def run_oak_headless():
    """
    Run the OAK-D camera in headless mode (without Tkinter UI)
    Useful for Raspberry Pi environments where Tkinter might have issues
    """
    print("Running in headless mode with OAK-D camera")
    try:
        # Check for available devices
        available_devices = dai.Device.getAllAvailableDevices()
        if len(available_devices) == 0:
            print("No OAK-D devices found. Exiting headless mode.")
            return
            
        print(f"Found {len(available_devices)} DepthAI device(s):")
        for i, device_info in enumerate(available_devices):
            print(f"  {i+1}: {device_info.getMxId()} (state: {device_info.state})")
        
        # Create pipeline with onboard camera
        pipeline = load_models(use_onboard_camera=True)
        
        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:
            # Get the RGB camera output queue
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            
            print("Device connected. Starting detection loop...")
            while True:
                # Get frame from the device
                in_rgb = q_rgb.tryGet()
                if in_rgb is None:
                    continue
                    
                frame = in_rgb.getCvFrame()
                
                # Process detections (simplified)
                if q_nn.has():
                    in_nn = q_nn.get()
                    detections = in_nn.detections
                    
                    # Display results on the frame
                    for detection in detections:
                        # Only process person detections (class 0 in COCO dataset)
                        if detection.label == 0:  # Person
                            bbox = detection.boundingBox
                            x1 = int(bbox.xmin)
                            y1 = int(bbox.ymin)
                            x2 = int(bbox.xmax)
                            y2 = int(bbox.ymax)
                            
                            # Draw person bounding box
                            draw_bounding_box(frame, x1, y1, x2, y2, config.COLOR_PERSON_BOX, thickness=2)
                            
                            # Add detection text
                            cv2.putText(frame, f"Person: {detection.confidence:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_PERSON_BOX, 2)
                
                # Show frame
                cv2.imshow("OAK-D Detection (Headless Mode)", frame)
                
                # Exit on 'q' pressed
                if cv2.waitKey(1) == ord('q'):
                    break
                    
        cv2.destroyAllWindows()
        print("Headless mode exited")
            
    except Exception as e:
        print(f"Error in headless mode: {e}")
        cv2.destroyAllWindows()

def run_detection_loop(video_source, ui):
    use_depthai_camera = False
    
    # Initialize video source
    if isinstance(video_source, str):
        if video_source == "depthai":
            # Use OAK-D camera directly
            print("Using OAK-D onboard camera")
            use_depthai_camera = True
            source_name = "OAK-D Camera"
            cap = None  # We'll use the DepthAI API directly
            
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
    try:
        ui.open_video_window()
    except Exception as e:
        print(f"Error opening video window: {e}")
        # Return to main menu
        ui.window.deiconify()
        return
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Drive Safe - Seatbelt & Phone Detection')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without Tkinter UI (for Raspberry Pi)')
    parser.add_argument('--video', type=str, help='Path to video file to process')
    args = parser.parse_args()
    
    # Run in headless mode if specified (for Raspberry Pi where Tkinter might not work well)
    if args.headless:
        if args.video:
            print(f"Headless mode with video not implemented yet. Using OAK-D camera instead.")
        run_oak_headless()
        return
        
    # If video file is specified, run with that
    if args.video:
        if os.path.exists(args.video):
            # Create UI but immediately process the video
            ui = DetectionUI(lambda: None, lambda: None, lambda: None)
            run_detection_loop(args.video, ui)
            return
        else:
            print(f"Video file not found: {args.video}")
    
    # Normal UI mode
    def on_video_selected(video_path):
        run_detection_loop(video_path, ui)
        
    def on_camera_selected():
        # First check if any OAK-D devices are connected
        try:
            print("Checking for connected OAK-D devices...")
            available_devices = dai.Device.getAllAvailableDevices()
            if len(available_devices) > 0:
                print(f"Found {len(available_devices)} DepthAI device(s):")
                for i, device_info in enumerate(available_devices):
                    print(f"  {i+1}: {device_info.getMxId()} (state: {device_info.state})")
                
                # Try to use the first available device
                print(f"Attempting to use OAK-D device: {available_devices[0].getMxId()}")
                run_detection_loop("depthai", ui)
                return
            else:
                print("No OAK-D devices found")
        except Exception as e:
            print(f"Error checking for OAK-D devices: {e}")
            
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

