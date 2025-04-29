import cv2

def list_available_cameras(max_cameras_to_check=10):
    """Lists available camera devices and their IDs."""
    print("\nAvailable Camera Devices:")
    available_cameras = []
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"  Camera ID {i}: Found")
            available_cameras.append(i)
            cap.release()
        # else:
            # print(f"  Camera ID {i}: Not found or cannot be opened.") # Optional: uncomment for more verbose output
    if not available_cameras:
        print("  No cameras found.")
    print("") # Add a newline for spacing
    return available_cameras

# Add other general utility functions here if needed in the future.
