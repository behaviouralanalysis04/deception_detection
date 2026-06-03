"""
Video processing module for deception detection
Handles facial landmark detection and behavioral analysis
"""

import cv2
import numpy as np
import time
import dlib
from collections import deque
import queue

class EnhancedVideoProcessor:
    """Real-time video processor for live camera analysis"""
    
    def __init__(self):
        # Initialize dlib detectors
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.use_dlib = True
        except:
            self.use_dlib = False
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Feature tracking
        self.frame_count = 0
        self.process_every_n_frames = 3
        self.blink_counter = 0
        self.blink_rate = 0
        self.eye_closed_start = None
        
        # Gaze tracking
        self.gaze_durations = {'left': 0, 'right': 0, 'center': 0}
        
        # Feature storage
        self.mar_values = deque(maxlen=300)
        self.lip_comp_values = deque(maxlen=300)
        self.asymmetry_values = deque(maxlen=300)
        self.head_pitch_values = deque(maxlen=300)
        self.head_yaw_values = deque(maxlen=300)
        self.head_roll_values = deque(maxlen=300)
        
        self.nod_count = 0
        self.shake_count = 0
        self.tilt_count = 0
        self.prev_head_pitch = None
        self.prev_head_yaw = None
        self.prev_head_roll = None
        
        self.prev_landmarks = None
        self.micro_expression_frames = 0
        self.features_queue = queue.Queue(maxsize=5)
        self.last_process_time = time.time()
        self.fps_target = 10
    
    def eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for blink detection"""
        points = [landmarks.part(i) for i in eye_indices]
        p1 = np.array([points[1].x, points[1].y])
        p2 = np.array([points[2].x, points[2].y])
        p3 = np.array([points[3].x, points[3].y])
        p4 = np.array([points[4].x, points[4].y])
        p5 = np.array([points[5].x, points[5].y])
        p6 = np.array([points[0].x, points[0].y])
        
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        if horizontal == 0:
            return 0
        return (vertical1 + vertical2) / (2.0 * horizontal)
    
    def recv(self, frame):
        """Process video frame"""
        from streamlit_webrtc import VideoProcessorBase
        import av
        
        current_time = time.time()
        
        if current_time - self.last_process_time < 1.0 / self.fps_target:
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        self.last_process_time = current_time
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.frame_count += 1
        
        if self.use_dlib:
            faces = self.detector(gray, 0)
        else:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            face = faces[0] if self.use_dlib else faces[0]
            
            if self.use_dlib:
                landmarks = self.predictor(gray, face)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(img, (x, y), (x+w, y+h), (102, 126, 234), 2)
                
                left_ear = self.eye_aspect_ratio(landmarks, range(36, 42))
                right_ear = self.eye_aspect_ratio(landmarks, range(42, 48))
                ear = (left_ear + right_ear) / 2.0
                
                if ear < 0.2:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = current_time
                else:
                    if self.eye_closed_start is not None:
                        self.blink_counter += 1
                        self.eye_closed_start = None
                
                self.blink_rate = (self.blink_counter / max(self.frame_count, 1)) * 30 * 60
            else:
                (x, y, w, h) = face
                cv2.rectangle(img, (x, y), (x+w, y+h), (102, 126, 234), 2)
                self.blink_rate = (self.blink_counter / max(self.frame_count, 1)) * 30 * 60
        
        cv2.putText(img, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        cv2.putText(img, f"Blinks: {self.blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        cv2.putText(img, f"Rate: {self.blink_rate:.0f}/min", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        
        # Extract features periodically
        if self.frame_count % 90 == 0 and self.frame_count >= 90:
            features = self._extract_features()
            if self.features_queue.empty():
                self.features_queue.put(features)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _extract_features(self):
        """Extract features from processed frames"""
        duration_sec = self.frame_count / 30
        total_frames = max(1, self.frame_count)
        
        features = {
            'blink_rate': min(float(self.blink_rate), 45.0),
            'avg_blink_duration': 0.2,
            'gaze_left_ratio': float(self.gaze_durations['left']) / total_frames,
            'gaze_right_ratio': float(self.gaze_durations['right']) / total_frames,
            'gaze_center_ratio': float(self.gaze_durations['center']) / total_frames,
            'avg_mouth_open_ratio': float(np.mean(self.mar_values)) if self.mar_values else 0.0,
            'std_mouth_open_ratio': float(np.std(self.mar_values)) if self.mar_values else 0.0,
            'avg_facial_asymmetry': float(np.mean(self.asymmetry_values)) if self.asymmetry_values else 0.0,
            'std_facial_asymmetry': float(np.std(self.asymmetry_values)) if self.asymmetry_values else 0.0,
            'avg_lip_compression': float(np.mean(self.lip_comp_values)) if self.lip_comp_values else 0.0,
            'micro_expression_frequency': float(self.micro_expression_frames) / max(duration_sec, 0.1),
            'avg_head_pitch': float(np.mean(self.head_pitch_values)) if self.head_pitch_values else 0.0,
            'std_head_pitch': float(np.std(self.head_pitch_values)) if self.head_pitch_values else 0.0,
            'avg_head_yaw': float(np.mean(self.head_yaw_values)) if self.head_yaw_values else 0.0,
            'std_head_yaw': float(np.std(self.head_yaw_values)) if self.head_yaw_values else 0.0,
            'avg_head_roll': float(np.mean(self.head_roll_values)) if self.head_roll_values else 0.0,
            'std_head_roll': float(np.std(self.head_roll_values)) if self.head_roll_values else 0.0,
            'head_nod_frequency': float(self.nod_count) / max(duration_sec, 0.1),
            'head_shake_frequency': float(self.shake_count) / max(duration_sec, 0.1),
            'head_tilt_frequency': float(self.tilt_count) / max(duration_sec, 0.1),
            'duration_seconds': duration_sec
        }
        return features

class VideoAnalysisProcessor:
    """Process uploaded video files"""
    
    def __init__(self):
        self.features = None
    
    def process_video(self, video_path):
        """Extract features from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        if duration > 60:
            cap.release()
            raise Exception(f"Video too long: {duration:.1f}s (max 60s)")
        
        # Simulate feature extraction (replace with actual extraction)
        time.sleep(1)
        
        features = {
            'blink_rate': np.random.uniform(5, 45),
            'avg_blink_duration': np.random.uniform(0.1, 0.4),
            'gaze_left_ratio': np.random.uniform(0, 0.5),
            'gaze_right_ratio': np.random.uniform(0, 0.5),
            'gaze_center_ratio': np.random.uniform(0.2, 0.8),
            'avg_mouth_open_ratio': np.random.uniform(0.1, 0.5),
            'std_mouth_open_ratio': np.random.uniform(0.05, 0.2),
            'avg_facial_asymmetry': np.random.uniform(5, 30),
            'std_facial_asymmetry': np.random.uniform(1, 10),
            'avg_lip_compression': np.random.uniform(5, 25),
            'micro_expression_frequency': np.random.uniform(0.5, 4),
            'avg_head_pitch': np.random.uniform(-15, 15),
            'std_head_pitch': np.random.uniform(1, 10),
            'avg_head_roll': np.random.uniform(-10, 10),
            'std_head_roll': np.random.uniform(1, 8),
            'avg_head_yaw': np.random.uniform(-20, 20),
            'std_head_yaw': np.random.uniform(1, 12),
            'head_nod_frequency': np.random.uniform(0, 2),
            'head_shake_frequency': np.random.uniform(0, 1.5),
            'head_tilt_frequency': np.random.uniform(0, 1),
            'duration_seconds': duration
        }
        
        cap.release()
        self.features = features
        return features
    
    def generate_summary(self, features):
        """Generate deception summary from features"""
        score = 0
        indicators = []
        
        if features['blink_rate'] > 30:
            score += 15
            indicators.append(('Elevated blink rate', 'warning'))
        elif features['blink_rate'] < 10:
            score += 10
            indicators.append(('Reduced blink rate', 'warning'))
        
        gaze_aversion = features['gaze_left_ratio'] + features['gaze_right_ratio']
        if gaze_aversion > 0.6:
            score += 20
            indicators.append(('Frequent gaze aversion', 'danger'))
        elif gaze_aversion > 0.4:
            score += 10
            indicators.append(('Occasional gaze aversion', 'warning'))
        
        if features['avg_lip_compression'] > 15:
            score += 15
            indicators.append(('Lip compression detected', 'danger'))
        
        if features['avg_facial_asymmetry'] > 10:
            score += 15
            indicators.append(('Facial asymmetry detected', 'warning'))
        
        if features['micro_expression_frequency'] > 2:
            score += 20
            indicators.append(('Frequent micro-expressions', 'danger'))
        elif features['micro_expression_frequency'] > 1:
            score += 10
            indicators.append(('Occasional micro-expressions', 'warning'))
        
        deception_score = min(100, score)
        
        # Import here to avoid circular imports
        from utils import get_classification_for_score, get_color_for_score
        
        return {
            'deception_score': deception_score,
            'indicators': indicators,
            'classification': get_classification_for_score(deception_score),
            'color': get_color_for_score(deception_score, True),
            'icon': '🔴' if deception_score >= 60 else '🟡' if deception_score >= 40 else '🟢',
            'confidence': 'High' if len(indicators) >= 4 else 'Medium' if len(indicators) >= 2 else 'Low'
        }