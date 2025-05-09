import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image
from PIL import ImageTk
import threading
import cv2
import os
from datetime import datetime

class DetectionUI:
    def __init__(self, on_video_selected, on_camera_selected, on_exit):
        self.on_video_selected = on_video_selected
        self.on_camera_selected = on_camera_selected
        self.on_exit = on_exit
        self.window = tk.Tk()
        self.window.title("Seatbelt & Phone Detection Demo")
        self.setup_welcome_screen()

    def setup_welcome_screen(self):
        # Title
        tk.Label(self.window, text="Drive Safe App", font=("Arial", 18)).pack(padx=5, pady=5)

        # Welcome Image
        try:
            welcome_image = Image.open("images/welcome_image.jpg")
            welcome_image = welcome_image.resize((300, 300))
            self.photo = ImageTk.PhotoImage(welcome_image)
            tk.Label(self.window, image=self.photo).pack(pady=10, padx=20)
        except:
            tk.Label(self.window, text="[Welcome Image Placeholder]", font=("Arial", 14)).pack(pady=10)

        # Status Text
        tk.Label(self.window, text="No video loaded.\nPlease load a video or start the camera.", font=("Arial", 12)).pack(padx=5, pady=5)

        # Buttons
        tk.Button(self.window, text="Load Video", font=("Arial", 14), command=self.load_video).pack(pady=10)
        tk.Button(self.window, text="Start Camera", font=("Arial", 14), command=self.start_camera).pack(pady=10)
        tk.Button(self.window, text="Exit", font=("Arial", 14), command=self.exit_app).pack(pady=10)

        self.center_window()

    def center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"+{x}+{y}")

    def load_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if file_path:
            self.window.withdraw()
            self.on_video_selected(file_path)

    def start_camera(self):
        self.window.withdraw()
        self.on_camera_selected()

    def exit_app(self):
        self.on_exit()
        self.window.destroy()

    def open_video_window(self, video_title="Seatbelt & Phone Detection Interface"):
        self.video_window = tk.Toplevel()
        self.video_window.title(video_title)
        
        # Get screen dimensions
        screen_width = self.video_window.winfo_screenwidth()
        screen_height = self.video_window.winfo_screenheight()
        
        # Set window size to 90% of screen size
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        self.video_window.geometry(f"{window_width}x{window_height}")
        
        self.video_window.grid_rowconfigure(0, weight=1)
        self.video_window.grid_columnconfigure(0, weight=3)
        self.video_window.grid_columnconfigure(1, weight=1)

        # Video Display (Left Pane)
        self.video_frame = tk.Frame(self.video_window, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)

        # Detections Display (Right Pane)
        detections_frame = tk.Frame(self.video_window)
        detections_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.canvas = tk.Canvas(detections_frame)
        self.scrollbar = tk.Scrollbar(detections_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        tk.Button(self.video_window, text="Close", command=self.close_video_window).grid(row=1, column=0, columnspan=2, pady=10)
        self.video_window.protocol("WM_DELETE_WINDOW", self.close_video_window)

    def update_video_frame(self, frame_imgtk, frame_width=None, frame_height=None):
        if frame_width and frame_height:
            # Calculate scaling to fit the video frame while maintaining aspect ratio
            frame_aspect = frame_width / frame_height
            video_frame_width = self.video_frame.winfo_width()
            video_frame_height = self.video_frame.winfo_height()
            frame_aspect = frame_width / frame_height
            
            if video_frame_width / video_frame_height > frame_aspect:
                # Window is wider than video
                display_height = video_frame_height
                display_width = int(display_height * frame_aspect)
            else:
                # Window is taller than video
                display_width = video_frame_width
                display_height = int(display_width / frame_aspect)
            
            # Update label size
            self.video_label.config(width=display_width, height=display_height)
            
        self.video_label.configure(image=frame_imgtk)
        self.video_label.image = frame_imgtk

    def update_detections(self, detections):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Add current date time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tk.Label(self.scrollable_frame, text=current_time, font=("Arial", 12)).pack(anchor="w", padx=5, pady=5)
            
        for i, det in enumerate(detections):
            entry_frame = tk.Frame(self.scrollable_frame, bd=2, relief="groove")
            entry_frame.pack(fill="x", padx=5, pady=5)
            
            seatbelt_status = det.get('seatbelt_status', 'Unknown')
            seatbelt_score = det.get('seatbelt_score', 0.0)
            phone_detected = det.get('phone_detected', False)
            phone_score = det.get('phone_score', 0.0)
            
            seatbelt_text = f"Seatbelt: {seatbelt_status} ({seatbelt_score:.2f})"
            seatbelt_color = "green" if seatbelt_status == "Worn" else "red"
            tk.Label(entry_frame, text=seatbelt_text, fg=seatbelt_color, font=("Arial", 11)).pack(anchor="w")
            
            phone_text = f"Phone: {'Detected' if phone_detected else 'Not Detected'}"
            if phone_detected:
                phone_text += f" ({phone_score:.2f})"
            tk.Label(entry_frame, text=phone_text, fg="orange" if phone_detected else "gray", font=("Arial", 11)).pack(anchor="w")

            # Display detection image
            if 'detection_image' in det:
                image_label = tk.Label(entry_frame, image=det['detection_image'])
                image_label.image = det['detection_image']  # Keep a reference
                image_label.pack(pady=5)

    def close_video_window(self):
        self.video_window.destroy()
        self.window.deiconify()

    def run(self):
        self.window.mainloop()
