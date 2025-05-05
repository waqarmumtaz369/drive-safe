import cv2
import os
import argparse
import tensorflow as tf
import time
from detection_ui import DetectionUI
from PIL import Image, ImageTk
import threading

# Import from modules
import config
from model_loader import load_models
from project_utils import list_available_cameras
from project_utils import resize_image
from detectors import detect_objects_and_seatbelt
from visualization import draw_bounding_box, draw_text

# Disable oneDNN custom operations warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Tensorflow version:", tf.__version__)
print("Script loaded. Import complete")

def run_detection_loop(video_source, ui_callback=None):
    person_model, phone_model, seatbelt_model = load_models()
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
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video or cannot read frame from camera.")
            break
        frame = resize_image(frame)
        detections = detect_objects_and_seatbelt(frame, person_model, phone_model, seatbelt_model)
        frame_processing_time = time.time() - start_time
        print(f"Frame Processing time: {frame_processing_time:.3f} sec")
        # Draw results on the frame
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
            text_y = py1 - 10 if py1 > 20 else py1 + 20
            draw_text(frame, px1, text_y, seatbelt_text, box_color)
            if phone_detected:
                phone_text = f"Phone Detected ({phone_score:.4f})"
                draw_text(frame, px1, text_y - 25, phone_text, config.COLOR_YELLOW)
                if det['phone_box']:
                    phx1, phy1, phx2, phy2 = det['phone_box']
                    draw_bounding_box(frame, phx1, phy1, phx2, phy2, config.COLOR_YELLOW, thickness=1)
        # UI callback for updating UI
        if ui_callback:
            # Convert frame to ImageTk
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb_frame)
            im = im.resize((640, 360))
            imgtk = ImageTk.PhotoImage(image=im)
            ui_callback(imgtk, detections)
        else:
            cv2.imshow("Seatbelt and Phone Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    cap.release()
    if not ui_callback:
        cv2.destroyAllWindows()
    print("Resources released.")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Seatbelt and Phone Detection using Webcam or Video File.")
    parser.add_argument("--video", type=str, help="Path to the video file. If not provided, webcam 0 is used.")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID to use (default: 0).")
    parser.add_argument("--list_cameras", action="store_true", help="List available cameras and exit.")
    args = parser.parse_args()

    if args.list_cameras:
        list_available_cameras()
        exit(0)

    if not args.video and not (args.camera_id != 0):
        # No CLI args: launch UI
        def on_video_selected(video_path):
            ui.open_video_window()
            def update_ui(imgtk, detections):
                ui.update_video_frame(imgtk)
                ui.update_detections(detections)
            threading.Thread(target=run_detection_loop, args=(video_path, update_ui), daemon=True).start()
        def on_camera_selected():
            ui.open_video_window()
            def update_ui(imgtk, detections):
                ui.update_video_frame(imgtk)
                ui.update_detections(detections)
            threading.Thread(target=run_detection_loop, args=(0, update_ui), daemon=True).start()
        def on_exit():
            pass
        ui = DetectionUI(on_video_selected, on_camera_selected, on_exit)
        ui.run()
    else:
        # CLI mode
        video_source = args.video if args.video else args.camera_id
        run_detection_loop(video_source)

