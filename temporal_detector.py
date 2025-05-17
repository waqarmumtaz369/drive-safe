import time
from collections import deque
import numpy as np

class TemporalDetector:
    def __init__(self, queue_duration=5.0, seatbelt_threshold=50, phone_threshold=50):
        """
        Initialize the temporal detection system.
        
        Args:
            queue_duration (float): Duration in seconds to maintain detection history
            seatbelt_threshold (float): Percentage threshold for seatbelt violation detection
            phone_threshold (float): Percentage threshold for phone violation detection
        """
        self.queue_duration = queue_duration
        self.times_seatbelt_detected = seatbelt_threshold
        self.times_phone_detected = phone_threshold
        
        # Initialize queues with timestamps
        self.seatbelt_queue = deque()  # Stores tuples of (timestamp, value)
        self.phone_queue = deque()  # Stores tuples of (timestamp, value)
        
        # Violation state
        self.is_violation = False
        self.violation_types = set()  # Can contain 'seatbelt', 'phone' or both
        
        # Queue capacity calculation will be based on FPS
        self.last_fps_update = time.time()
        self.frame_times = deque(maxlen=30)  # Store last 30 frame times for FPS calculation
        self.current_fps = None
    
    def update_fps(self):
        """Calculate current FPS based on frame times"""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_fps_update)
        self.last_fps_update = current_time
        
        if len(self.frame_times) >= 10:  # Wait for at least 10 frames
            self.current_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def _clean_old_entries(self, current_time):
        """Remove entries older than queue_duration from both queues"""
        cutoff_time = current_time - self.queue_duration
        
        # Clean seatbelt queue
        while self.seatbelt_queue and self.seatbelt_queue[0][0] < cutoff_time:
            self.seatbelt_queue.popleft()
            
        # Clean phone queue    
        while self.phone_queue and self.phone_queue[0][0] < cutoff_time:
            self.phone_queue.popleft()
    
    def process_detection(self, detection):
        """
        Process a new detection frame and update violation state.
        
        Args:
            detection (dict): Detection result containing seatbelt and phone info
        
        Returns:
            tuple: (is_violation, violation_types)
        """
        current_time = time.time()
        self.update_fps()
        
        # Process seatbelt detection
        seatbelt_value = 1 if detection['seatbelt_status'] == "Worn" else 0
        self.seatbelt_queue.append((current_time, seatbelt_value))
        
        # Process phone detection
        phone_value = 1 if detection['phone_detected'] else 0
        self.phone_queue.append((current_time, phone_value))
        
        # Clean old entries
        self._clean_old_entries(current_time)
        
        # Only evaluate if we have enough data (queue is populated)
        if self.current_fps and len(self.seatbelt_queue) >= max(10, self.current_fps * 2):  # At least 2 seconds of data
            # Calculate violation percentages
            seatbelt_not_worn_percent = (1 - sum(v for _, v in self.seatbelt_queue) / len(self.seatbelt_queue)) * 100
            phone_detected_percent = sum(v for _, v in self.phone_queue) / len(self.phone_queue) * 100
            
            # Update violation types
            self.violation_types.clear()
            
            if seatbelt_not_worn_percent > (100 - self.times_seatbelt_detected):
                self.violation_types.add('seatbelt')
            
            if phone_detected_percent > self.times_phone_detected:
                self.violation_types.add('phone')
            
            # Update overall violation state
            self.is_violation = len(self.violation_types) > 0
            
        return self.is_violation, self.violation_types.copy()
