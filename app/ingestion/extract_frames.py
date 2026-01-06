"""
Frame extraction module for video files.
Extracts frames at specified FPS for visual emotion analysis.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def extract_frames(video_path: str, fps: float = 2.0, output_dir: Optional[str] = None) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input video file
        fps: Frames per second to extract (default 2.0)
        output_dir: Optional directory to save frames. If None, frames kept in memory.
    
    Returns:
        List of tuples: (timestamp_seconds, frame_array)
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Extract every Nth frame
    
    frames = []
    frame_count = 0
    timestamp = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append((timestamp, frame_rgb))
                
                # Optionally save to disk
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    frame_filename = os.path.join(
                        output_dir,
                        f"frame_{timestamp:.2f}.jpg"
                    )
                    cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
    
    finally:
        cap.release()
    
    return frames


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """
    Extract a single frame at specific timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
    
    Returns:
        Frame array (RGB) or None if timestamp is out of range
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video properties
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    try:
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        return info
    finally:
        cap.release()
