import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime
import threading
import queue

class LiveVirtualTryOn:
    def __init__(self):
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Clothing items storage
        self.clothing_items = []
        self.current_clothing_index = 0
        self.load_clothing_items()
        
        # UI elements
        self.show_instructions = True
        self.recording = False
        self.frame_count = 0
        
    def load_clothing_items(self):
        """Load clothing items from the clothes directory"""
        clothes_dir = "clothes"
        if not os.path.exists(clothes_dir):
            os.makedirs(clothes_dir)
            print(f"Created {clothes_dir} directory. Please add clothing images there.")
        
        # Load all clothing images
        for filename in os.listdir(clothes_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(clothes_dir, filename)
                clothing_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if clothing_img is not None:
                    self.clothing_items.append({
                        'name': filename,
                        'image': clothing_img,
                        'path': img_path
                    })
        
        if not self.clothing_items:
            # Create a default shirt if no clothes found
            default_shirt = self.create_default_shirt()
            self.clothing_items.append({
                'name': 'Default Shirt',
                'image': default_shirt,
                'path': 'default'
            })
    
    def create_default_shirt(self):
        """Create a default colored shirt for demo purposes"""
        shirt = np.zeros((300, 250, 4), dtype=np.uint8)
        
        # Create shirt shape
        cv2.rectangle(shirt, (50, 50), (200, 250), (0, 100, 200, 200), -1)  # Red shirt
        cv2.rectangle(shirt, (75, 50), (175, 100), (0, 80, 180, 200), -1)   # Collar area
        
        # Add some texture
        for i in range(5):
            y = 80 + i * 30
            cv2.line(shirt, (60, y), (190, y), (0, 120, 220, 100), 2)
        
        return shirt
    
    def get_body_landmarks(self, frame):
        """Extract body pose landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = []
            h, w = frame.shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            return landmarks, results
        return None, None
    
    def get_person_mask(self, frame):
        """Get person segmentation mask"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            return mask
        return None
    
    def fit_clothing_to_body(self, clothing_img, landmarks):
        """Fit clothing item to detected body landmarks"""
        if landmarks is None or len(landmarks) < 25:
            return None, None
        
        # Key body points
        left_shoulder = landmarks[11]   # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        left_hip = landmarks[23]        # Left hip
        right_hip = landmarks[24]       # Right hip
        
        # Calculate clothing dimensions
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        torso_height = abs(left_hip[1] - left_shoulder[1])
        
        if shoulder_width < 50 or torso_height < 50:
            return None, None
        
        # Resize clothing to fit body
        clothing_resized = cv2.resize(clothing_img, (shoulder_width, torso_height))
        
        # Calculate position
        center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        center_y = (left_shoulder[1] + left_hip[1]) // 2
        
        # Position coordinates
        x1 = center_x - shoulder_width // 2
        y1 = center_y - torso_height // 2
        
        return clothing_resized, (x1, y1, shoulder_width, torso_height)
    
    def apply_clothing_to_frame(self, frame, clothing_img, position):
        """Apply clothing to the frame with realistic blending"""
        if position is None:
            return frame
        
        x1, y1, width, height = position
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, w - width))
        y1 = max(0, min(y1, h - height))
        x2 = min(w, x1 + width)
        y2 = min(h, y1 + height)
        
        # Adjust clothing size if needed
        actual_width = x2 - x1
        actual_height = y2 - y1
        
        if actual_width <= 0 or actual_height <= 0:
            return frame
        
        clothing_fitted = cv2.resize(clothing_img, (actual_width, actual_height))
        
        # Handle transparency if clothing has alpha channel
        if clothing_fitted.shape[2] == 4:
            # Extract alpha channel
            alpha = clothing_fitted[:, :, 3] / 255.0
            clothing_rgb = clothing_fitted[:, :, :3]
            
            # Apply alpha blending
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * clothing_rgb[:, :, c] + 
                    (1 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            # Simple overlay without transparency
            # Create a soft mask for better blending
            mask = np.ones((actual_height, actual_width), dtype=np.float32) * 0.8
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    mask * clothing_fitted[:, :, c] + 
                    (1 - mask) * frame[y1:y2, x1:x2, c]
                )
        
        return frame
    
    def draw_ui_elements(self, frame):
        """Draw user interface elements"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for UI
        overlay = frame.copy()
        
        # Current clothing info
        if self.clothing_items:
            current_item = self.clothing_items[self.current_clothing_index]
            text = f"Current: {current_item['name']} ({self.current_clothing_index + 1}/{len(self.clothing_items)})"
            cv2.rectangle(overlay, (10, 10), (500, 50), (0, 0, 0), -1)
            cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        if self.show_instructions:
            instructions = [
                "Controls:",
                "A/D - Change clothes",
                "S - Save screenshot",
                "R - Start/Stop recording",
                "H - Hide/Show instructions",
                "Q - Quit"
            ]
            
            y_start = h - 200
            cv2.rectangle(overlay, (10, y_start - 20), (300, h - 10), (0, 0, 0), -1)
            
            for i, instruction in enumerate(instructions):
                y = y_start + i * 25
                cv2.putText(overlay, instruction, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(overlay, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(overlay, "REC", (w - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Blend overlay with frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input for controls"""
        if key == ord('q') or key == 27:  # Q or ESC to quit
            return False
        elif key == ord('a'):  # Previous clothing
            if self.clothing_items:
                self.current_clothing_index = (self.current_clothing_index - 1) % len(self.clothing_items)
        elif key == ord('d'):  # Next clothing
            if self.clothing_items:
                self.current_clothing_index = (self.current_clothing_index + 1) % len(self.clothing_items)
        elif key == ord('s'):  # Save screenshot
            self.save_screenshot()
        elif key == ord('r'):  # Toggle recording
            self.toggle_recording()
        elif key == ord('h'):  # Toggle instructions
            self.show_instructions = not self.show_instructions
        
        return True
    
    def save_screenshot(self):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tryon_screenshot_{timestamp}.jpg"
        # This will be saved in the next frame capture
        self.save_next_frame = filename
    
    def toggle_recording(self):
        """Toggle video recording"""
        self.recording = not self.recording
        if self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"tryon_recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (1280, 720))
            print(f"Started recording: {self.video_filename}")
        else:
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
                print(f"Stopped recording: {self.video_filename}")
    
    def run(self):
        """Main application loop"""
        print("Live Virtual Try-On Started!")
        print("Make sure you have clothing images in the 'clothes' directory")
        print("Controls: A/D - change clothes, S - screenshot, R - record, H - toggle help, Q - quit")
        
        self.save_next_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Get body landmarks
            landmarks, pose_results = self.get_body_landmarks(frame)
            
            # Apply clothing if available and body is detected
            if landmarks and self.clothing_items:
                current_clothing = self.clothing_items[self.current_clothing_index]['image']
                fitted_clothing, position = self.fit_clothing_to_body(current_clothing, landmarks)
                
                if fitted_clothing is not None:
                    frame = self.apply_clothing_to_frame(frame, fitted_clothing, position)
            
            # Draw pose landmarks (optional, for debugging)
            if pose_results and pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            # Draw UI elements
            frame = self.draw_ui_elements(frame)
            
            # Save screenshot if requested
            if self.save_next_frame:
                cv2.imwrite(self.save_next_frame, frame)
                print(f"Screenshot saved: {self.save_next_frame}")
                self.save_next_frame = None
            
            # Record video if recording
            if self.recording and hasattr(self, 'video_writer'):
                self.video_writer.write(frame)
            
            # Display frame
            cv2.imshow('Live Virtual Try-On', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard_input(key):
                break
            
            self.frame_count += 1
        
        # Cleanup
        self.cap.release()
        if self.recording and hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()

# Enhanced version with gesture controls
class GestureControlledTryOn(LiveVirtualTryOn):
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.gesture_cooldown = 0
    
    def detect_gestures(self, frame):
        """Detect hand gestures for clothing control"""
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand landmarks
                landmarks = []
                h, w = frame.shape[:2]
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                # Detect swipe gestures
                if len(landmarks) >= 21:
                    # Check for left swipe (next clothing)
                    if landmarks[8][0] < landmarks[5][0] - 50:  # Index finger moving left
                        self.current_clothing_index = (self.current_clothing_index + 1) % len(self.clothing_items)
                        self.gesture_cooldown = 30  # Cooldown to prevent rapid switching
                    
                    # Check for right swipe (previous clothing)
                    elif landmarks[8][0] > landmarks[5][0] + 50:  # Index finger moving right
                        self.current_clothing_index = (self.current_clothing_index - 1) % len(self.clothing_items)
                        self.gesture_cooldown = 30
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
    
    def run(self):
        """Enhanced run method with gesture control"""
        print("Gesture-Controlled Live Virtual Try-On Started!")
        print("Use hand gestures to control clothing or keyboard controls")
        print("Swipe left/right with index finger to change clothes")
        
        self.save_next_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            self.detect_gestures(frame)
            
            # Get body landmarks and apply clothing
            landmarks, pose_results = self.get_body_landmarks(frame)
            
            if landmarks and self.clothing_items:
                current_clothing = self.clothing_items[self.current_clothing_index]['image']
                fitted_clothing, position = self.fit_clothing_to_body(current_clothing, landmarks)
                
                if fitted_clothing is not None:
                    frame = self.apply_clothing_to_frame(frame, fitted_clothing, position)
            
            # Draw UI elements
            frame = self.draw_ui_elements(frame)
            
            # Save screenshot if requested
            if self.save_next_frame:
                cv2.imwrite(self.save_next_frame, frame)
                print(f"Screenshot saved: {self.save_next_frame}")
                self.save_next_frame = None
            
            # Record video if recording
            if self.recording and hasattr(self, 'video_writer'):
                self.video_writer.write(frame)
            
            cv2.imshow('Gesture-Controlled Virtual Try-On', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard_input(key):
                break
        
        # Cleanup
        self.cap.release()
        if self.recording and hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Choose your preferred version
    print("Choose your virtual try-on mode:")
    print("1. Basic Live Try-On")
    print("2. Gesture-Controlled Try-On")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        app = GestureControlledTryOn()
    else:
        app = LiveVirtualTryOn()
    
    app.run()
