import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import cv2
import os

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
        tk.Label(self.window, text="Welcome to Seatbelt & Phone Detection", font=("Arial", 16)).pack(padx=10, pady=10)

        # Welcome Image
        try:
            welcome_image = Image.open("welcome_image.jpg")
            welcome_image = welcome_image.resize((300, 300))
            self.photo = ImageTk.PhotoImage(welcome_image)
            tk.Label(self.window, image=self.photo).pack(pady=10)
        except:
            tk.Label(self.window, text="[Welcome Image Placeholder]", font=("Arial", 14)).pack(pady=10)

        # Status Text
        tk.Label(self.window, text="No video loaded. Please load a video or start the camera.", font=("Arial", 12)).pack(pady=10)

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
        self.video_window.state('zoomed')
        self.video_window.grid_rowconfigure(0, weight=1)
        self.video_window.grid_columnconfigure(0, weight=3)
        self.video_window.grid_columnconfigure(1, weight=1)

        # Video Display (Left Pane)
        self.video_frame = tk.Frame(self.video_window, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.video_label = tk.Label(self.video_frame, text="[Video Feed Placeholder]", fg="white", bg="black", font=("Arial", 16))
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

    def update_video_frame(self, frame_imgtk):
        self.video_label.configure(image=frame_imgtk)
        self.video_label.image = frame_imgtk

    def update_detections(self, detections):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        for i, det in enumerate(detections):
            entry_frame = tk.Frame(self.scrollable_frame, bd=2, relief="groove")
            entry_frame.pack(fill="x", padx=5, pady=5)
            tk.Label(entry_frame, text=f"Person {i+1}", font=("Arial", 12, "bold")).pack(anchor="w")
            seatbelt_status = det.get('seatbelt_status', 'Unknown')
            seatbelt_score = det.get('seatbelt_score', 0.0)
            phone_detected = det.get('phone_detected', False)
            phone_score = det.get('phone_score', 0.0)
            seatbelt_text = f"Seatbelt: {seatbelt_status} ({seatbelt_score:.2f})"
            seatbelt_color = "green" if seatbelt_status == "Seatbelt Worn" else "red"
            tk.Label(entry_frame, text=seatbelt_text, fg=seatbelt_color, font=("Arial", 11)).pack(anchor="w")
            phone_text = f"Phone: {'Detected' if phone_detected else 'Not Detected'}"
            if phone_detected:
                phone_text += f" ({phone_score:.2f})"
            tk.Label(entry_frame, text=phone_text, fg="orange" if phone_detected else "gray", font=("Arial", 11)).pack(anchor="w")
            tk.Label(entry_frame, text="[Detection Image Placeholder]", relief="solid").pack(pady=5)

    def close_video_window(self):
        self.video_window.destroy()
        self.window.deiconify()

    def run(self):
        self.window.mainloop()
