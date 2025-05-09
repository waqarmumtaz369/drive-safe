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
from project_utils import resize_image

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Script loaded. Import complete")

def run_detection_loop(video_source, ui):
    pipeline = load_models()
    
    # Determine if using camera or video file
    use_depthai_camera = not isinstance(video_source, str)
    
    if use_depthai_camera:
        # Use DepthAI ColorCamera for live camera
        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            q_seatbelt_in = device.getInputQueue(name="seatbelt_in")
            q_seatbelt_out = device.getOutputQueue(name="seatbelt_out", maxSize=4, blocking=False)

            while True:
                start_time = time.time()
                in_rgb = q_rgb.tryGet()
                if in_rgb is None:
                    continue
                frame = in_rgb.getCvFrame()
                frame = resize_image(frame)  # Ensure frame is resized before processing
                
                detections = detect_objects_and_seatbelt(
                    frame, device, None, q_rgb, q_nn, q_seatbelt_in, q_seatbelt_out
                )
                # Draw results on frame (same as before)
                for det in detections:
                    px1, py1, px2, py2 = det['person_box']
                    seatbelt_status = det['seatbelt_status']
                    seatbelt_score = det['seatbelt_score']
                    phone_detected = det['phone_detected']
                    phone_score = det.get('phone_score', 0.0)
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
                    if phone_detected and det.get('phone_box'):
                        bx1, by1, bx2, by2 = det['phone_box']
                        draw_bounding_box(frame, bx1, by1, bx2, by2, config.COLOR_YELLOW, thickness=2)
                        phone_text = f"Phone ({phone_score:.2f})"
                        cv2.putText(frame, phone_text, (bx1, by1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 2)
                frame_time = time.time() - start_time
                fps_text = f"FPS: {1/frame_time:.1f}" if frame_time > 0 else "FPS: N/A"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Show video in a standalone OpenCV window
                cv2.imshow("OAK-D Camera Preview", frame)
                # Update detection results in Tkinter UI
                ui.update_detections(detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
    else:
        # Use OpenCV for video file
        if not os.path.exists(video_source):
            print(f"Error: Video file not found at {video_source}")
            return
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_source}.")
            return
        with dai.Device(pipeline) as device:
            q_in = device.getInputQueue(name="frame")
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            q_seatbelt_in = device.getInputQueue(name="seatbelt_in")
            q_seatbelt_out = device.getOutputQueue(name="seatbelt_out", maxSize=4, blocking=False)
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Finished processing video or cannot read frame from file.")
                    break
                frame = resize_image(frame)  # Ensure frame is resized before processing
                detections = detect_objects_and_seatbelt(
                    frame, device, q_in, q_rgb, q_nn, q_seatbelt_in, q_seatbelt_out
                )
                for det in detections:
                    px1, py1, px2, py2 = det['person_box']
                    seatbelt_status = det['seatbelt_status']
                    seatbelt_score = det['seatbelt_score']
                    phone_detected = det['phone_detected']
                    phone_score = det.get('phone_score', 0.0)
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
                    if phone_detected and det.get('phone_box'):
                        bx1, by1, bx2, by2 = det['phone_box']
                        draw_bounding_box(frame, bx1, by1, bx2, by2, config.COLOR_YELLOW, thickness=2)
                        phone_text = f"Phone ({phone_score:.2f})"
                        cv2.putText(frame, phone_text, (bx1, by1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 2)
                frame_time = time.time() - start_time
                fps_text = f"FPS: {1/frame_time:.1f}" if frame_time > 0 else "FPS: N/A"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Show video in a standalone OpenCV window
                cv2.imshow("Video File Preview", frame)
                # Update detection results in Tkinter UI
                ui.update_detections(detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

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
