import cv2
import depthai as dai
import time
import os
from model_loader import load_models, get_available_cameras
from detectors import detect_objects_and_seatbelt
from visualization import draw_bounding_box, draw_fps
from detection_ui import DetectionUI
from PIL import Image, ImageTk
from project_utils import resize_image
import config
import collections

def run_detection_loop(video_source, ui):
    # --- Time-based Violation Queue System Config ---
    is_violation = False
    queue_duration = config.QUEUE_DURATION
    times_seatbelt_detected = config.TIMES_SEATBELT_DETECTED
    times_phone_detected = config.TIMES_PHONE_DETECTED
    seatbelt_queue = collections.deque()
    phone_queue = collections.deque()
    fps_estimate = config.FPS_ESTIMATE_DEFAULT
    frame_times = collections.deque(maxlen=config.FRAME_TIMES_MAXLEN)

    # Special handling for DepthAI camera
    use_camera = video_source == "dai_camera"
    
    # Create and load pipeline with appropriate mode
    pipeline = load_models(use_camera=use_camera)
    
    # Initialize video source
    if video_source == "dai_camera":
        # We'll use the DepthAI API directly - no need for OpenCV capture
        cap = None
        source_name = "DepthAI OAK-D Camera"
    elif isinstance(video_source, str):
        # Regular video file
        if not os.path.exists(video_source):
            print(f"Error: Video file not found at {video_source}")
            return
        cap = cv2.VideoCapture(video_source)
        source_name = video_source
    else:
        # Fallback to regular OpenCV camera (shouldn't be used with OAK-D)
        print("Warning: Using standard OpenCV camera API which may not work with OAK-D devices")
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
        if use_camera:
            # For camera mode, we get frames from the RGB output queue
            q_in = None  # Not needed for camera mode
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        else:
            # For video file mode, we send frames to the input queue
            q_in = device.getInputQueue(name="frame")
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_seatbelt_in = device.getInputQueue(name="seatbelt_in")
        q_seatbelt_out = device.getOutputQueue(name="seatbelt_out", maxSize=4, blocking=False)

        # --- Calculate queue length based on FPS and duration ---
        queue_length = None
        while True:
            start_time = time.time()
            
            # Get frame - different method depending on source
            if use_camera:
                # Get frame directly from DepthAI
                in_rgb = q_rgb.get()
                if in_rgb is None:
                    continue
                frame = in_rgb.getCvFrame()  # Convert DepthAI frame to OpenCV format
            else:
                if cap is None:
                    break
                ret, frame = cap.read()
                if not ret:
                    print("Finished processing video or cannot read frame.")
                    break
            
            # Resize frame if needed
            frame = resize_image(frame)
            
            # --- FPS estimation for queue length ---
            frame_times.append(time.time())
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                fps_estimate = max(1, int(fps))
            if queue_length is None:
                queue_length = max(1, int(fps_estimate * queue_duration))
            
            # Process frame - adjust for camera mode
            if use_camera:
                # For camera mode, detections come directly from the pipeline
                # No need to send frames via q_in
                detections = detect_objects_and_seatbelt(
                    frame, device, None, q_rgb, q_nn, q_seatbelt_in, q_seatbelt_out
                )
            else:
                # For video mode, send frames to the pipeline
                detections = detect_objects_and_seatbelt(
                    frame, device, q_in, q_rgb, q_nn, q_seatbelt_in, q_seatbelt_out
                )
            
            # --- Only process the closest person (first detection) ---
            if detections:
                det = detections[0]
                seatbelt_status = det['seatbelt_status']
                seatbelt_score = det['seatbelt_score']
                phone_detected = det['phone_detected']
                # --- Update queues ---
                seatbelt_queue.append(1 if seatbelt_status == config.CLASS_NAMES_SEATBELT[1] and seatbelt_score >= config.THRESHOLD_SCORE_SEATBELT else 0)
                phone_queue.append(1 if phone_detected else 0)
                if len(seatbelt_queue) > queue_length:
                    seatbelt_queue.popleft()
                if len(phone_queue) > queue_length:
                    phone_queue.popleft()
            
            # --- Violation logic ---
            violation_now = False
            show_detections = False
            if len(seatbelt_queue) == queue_length and len(phone_queue) == queue_length:
                seatbelt_not_worn_pct = 100 * (1 - sum(seatbelt_queue) / queue_length)
                phone_detected_pct = 100 * (sum(phone_queue) / queue_length)
                if seatbelt_not_worn_pct > (100 - times_seatbelt_detected) or phone_detected_pct > times_phone_detected:
                    violation_now = True
                # --- UI update logic ---
                if violation_now:
                    is_violation = True
                    show_detections = True
                elif is_violation:
                    is_violation = False
                    show_detections = False
                    # Clear right UI pane
                    ui.update_detections([])
            
            # --- Draw results on frame ---
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
            # --- Only update detections in UI if violation is confirmed ---
            if show_detections:
                ui.update_detections(detections)
            
            # Process Tkinter events
            ui.video_window.update()
            
            if not ui.video_window.winfo_exists():
                break
                
        if cap:
            cap.release()

def main():
    def on_video_selected(video_path):
        run_detection_loop(video_path, ui)
        
    def on_camera_selected():
        # Check for available DepthAI cameras
        devices_info, has_camera = get_available_cameras()
        
        if has_camera:
            # Pass "dai_camera" as a special identifier for DepthAI camera mode
            run_detection_loop("dai_camera", ui)
        else:
            # Show error message
            from tkinter import messagebox
            messagebox.showerror("Camera Error", "No DepthAI cameras (OAK-D CM4) detected.")
        
    def on_exit():
        print("Exiting application...")
        
    # Create and run UI
    ui = DetectionUI(on_video_selected, on_camera_selected, on_exit)
    ui.run()

if __name__ == "__main__":
    main()
