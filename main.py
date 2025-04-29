import cv2
import os
import argparse
import tensorflow as tf # Keep for version print, though model loading moved

# Import from new modules
import config
from model_loader import load_models
from project_utils import list_available_cameras # Renamed from utils
from detectors import detect_objects_and_seatbelt
from visualization import draw_bounding_box, draw_text

# Disable oneDNN custom operations warning (can stay here or move to model_loader)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Tensorflow version:", tf.__version__)
print("Script loaded. Import complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Seatbelt and Phone Detection using Webcam or Video File.")
    parser.add_argument("--video", type=str, help="Path to the video file. If not provided, webcam 0 is used.")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID to use (default: 0).")
    parser.add_argument("--list_cameras", action="store_true", help="List available cameras and exit.")
    args = parser.parse_args()

    # If user just wants to list cameras
    if args.list_cameras:
        list_available_cameras()
        exit(0)

    # List available cameras anyway for reference
    list_available_cameras()

    # Load models using the function from model_loader
    person_model, phone_model, seatbelt_model = load_models()

    # Initialize video source
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found at {args.video}")
            exit()
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        cap = cv2.VideoCapture(args.camera_id)
        source_name = f"Camera ID {args.camera_id}"
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera_id}.")
            exit()

    print(f"Processing source: {source_name}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video or cannot read frame from camera.")
            break

        frame_count += 1

        # Perform detection using the function from detectors
        detections = detect_objects_and_seatbelt(frame, person_model, phone_model, seatbelt_model)

        # Log results to terminal
        print(f"--- Frame {frame_count} Detections ---")
        if not detections:
            print("  No persons detected.")
        for i, det in enumerate(detections):
            print(f"  Person {i+1}: Box={det['person_box']}, Seatbelt='{det['seatbelt_status']}' (Score: {det['seatbelt_score']:.2f}), Phone Detected={det['phone_detected']}")

        # Draw results on the frame using functions from visualization
        for det in detections:
            px1, py1, px2, py2 = det['person_box']
            seatbelt_status = det['seatbelt_status']
            seatbelt_score = det['seatbelt_score']
            phone_detected = det['phone_detected']

            # Determine box color based on seatbelt status (if score is high enough)
            box_color = config.COLOR_PERSON_BOX # Default
            seatbelt_text = f"{seatbelt_status} ({seatbelt_score:.2f})"
            if seatbelt_score >= config.THRESHOLD_SCORE_SEATBELT:
                if seatbelt_status == config.CLASS_NAMES_SEATBELT[1]: # Worn
                    box_color = config.COLOR_GREEN
                elif seatbelt_status == config.CLASS_NAMES_SEATBELT[0]: # Not worn
                    box_color = config.COLOR_RED
            else:
                 seatbelt_text = f"No Seatbelt Worn ({seatbelt_score:.2f})" # Indicate lower confidence


            # Draw person bounding box
            draw_bounding_box(frame, px1, py1, px2, py2, box_color)

            # Prepare text labels
            text_y = py1 - 10 if py1 > 20 else py1 + 20 # Position text above box, or below if too close to top
            draw_text(frame, px1, text_y, seatbelt_text, box_color)

            if phone_detected:
                # Draw phone indicator text
                phone_text = "Phone Detected"
                draw_text(frame, px1, text_y - 25, phone_text, config.COLOR_YELLOW)
                # Optionally draw phone box if needed
                if det['phone_box']:
                    phx1, phy1, phx2, phy2 = det['phone_box']
                    draw_bounding_box(frame, phx1, phy1, phx2, phy2, config.COLOR_YELLOW, thickness=1)


        # Display the frame
        cv2.imshow("Seatbelt and Phone Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

