"""
Audio extraction module for video files.
Uses FFmpeg to extract audio from video files.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def extract_audio(video_path: str, output_path: Optional[str] = None, sample_rate: int = 16000) -> str:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path for output audio file. If None, creates temp file.
        sample_rate: Target sample rate (default 16000 for ASR)
    
    Returns:
        Path to extracted audio file
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        subprocess.CalledProcessError: If FFmpeg fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        video_name = Path(video_path).stem
        output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # FFmpeg command to extract audio
    # -i: input file
    # -vn: disable video
    # -ar: audio sample rate
    # -ac: audio channels (mono)
    # -y: overwrite output file
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono channel
        "-y",  # Overwrite
        output_path
    ]
    
    try:
        # Run FFmpeg (suppress output)
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to extract audio: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH. "
            "Visit https://ffmpeg.org/download.html"
        )


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using FFmpeg.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get video duration: {e}")
