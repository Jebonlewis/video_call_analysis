"""
Facial emotion detection module.
Uses DeepFace to detect emotions from video frames.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from deepface import DeepFace
import cv2
import os


class FacialEmotionDetector:
    """Detects emotions from facial expressions in video frames."""
    
    def __init__(self, model_name: str = "VGG-Face", backend: str = "opencv"):
        """
        Initialize emotion detector.
        
        Args:
            model_name: DeepFace model name (VGG-Face, OpenFace, etc.)
            backend: Backend for face detection (opencv, ssd, dlib, etc.)
        """
        self.model_name = model_name
        self.backend = backend
    
    def detect_emotion(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect emotion in a single frame.
        
        Args:
            frame: Frame as numpy array (RGB format)
        
        Returns:
            Dictionary with emotion predictions or None if no face detected
        """
        try:
            # DeepFace expects RGB, but we'll ensure it's correct
            # Convert to BGR if needed (OpenCV format)
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Analyze emotion
            result = DeepFace.analyze(
                img_path=frame_bgr,
                actions=['emotion'],
                enforce_detection=False,  # Don't fail if no face
                silent=True,
                model_name=self.model_name,
                detector_backend=self.backend
            )
            
            # Handle both single result and list results
            if isinstance(result, list):
                result = result[0]
            
            # Extract emotion scores
            emotion_scores = result.get('emotion', {})
            
            if not emotion_scores:
                return None
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = emotion_scores[dominant_emotion]
            
            return {
                "emotion": dominant_emotion,
                "confidence": float(confidence) / 100.0,  # Convert to 0-1 range
                "all_emotions": {k: float(v) / 100.0 for k, v in emotion_scores.items()}
            }
        
        except Exception as e:
            # No face detected or other error
            return None
    
    def detect_emotions_batch(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[Dict]:
        """
        Detect emotions in multiple frames.
        
        Args:
            frames: List of (timestamp, frame_array) tuples
        
        Returns:
            List of dictionaries with timestamp and emotion info
        """
        results = []
        
        for timestamp, frame in frames:
            emotion_result = self.detect_emotion(frame)
            
            if emotion_result:
                results.append({
                    "timestamp": timestamp,
                    "emotion": emotion_result["emotion"],
                    "confidence": emotion_result["confidence"],
                    "all_emotions": emotion_result["all_emotions"]
                })
            else:
                # No face detected
                results.append({
                    "timestamp": timestamp,
                    "emotion": None,
                    "confidence": 0.0,
                    "all_emotions": {}
                })
        
        return results
    
    def get_emotion_at_timestamp(
        self,
        frames: List[Tuple[float, np.ndarray]],
        target_timestamp: float,
        tolerance: float = 0.5
    ) -> Optional[Dict]:
        """
        Get emotion for frame closest to target timestamp.
        
        Args:
            frames: List of (timestamp, frame_array) tuples
            target_timestamp: Target timestamp in seconds
            tolerance: Maximum time difference to consider
        
        Returns:
            Emotion result or None
        """
        closest_frame = None
        min_diff = float('inf')
        
        for timestamp, frame in frames:
            diff = abs(timestamp - target_timestamp)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_frame = (timestamp, frame)
        
        if closest_frame:
            _, frame = closest_frame
            return self.detect_emotion(frame)
        
        return None


def detect_emotion_in_frame(frame: np.ndarray) -> Optional[Dict]:
    """
    Convenience function to detect emotion in a frame.
    
    Args:
        frame: Frame as numpy array (RGB)
    
    Returns:
        Emotion result or None
    """
    detector = FacialEmotionDetector()
    return detector.detect_emotion(frame)
