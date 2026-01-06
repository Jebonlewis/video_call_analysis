"""
Whisper ASR module for speech transcription.
Uses OpenAI Whisper for high-quality speech-to-text.
"""

import whisper
import torch
from typing import List, Dict, Optional
import os


class WhisperTranscriber:
    """Wrapper for Whisper ASR model."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu, or None for auto)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en'). If None, auto-detect.
            task: 'transcribe' or 'translate'
        
        Returns:
            Dictionary with transcription results including segments with timestamps
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.model is None:
            self._load_model()
        
        try:
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=True,
                verbose=False
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def get_segments(self, transcription_result: Dict) -> List[Dict]:
        """
        Extract segments from transcription result.
        
        Args:
            transcription_result: Result from transcribe() method
        
        Returns:
            List of segment dictionaries with text, start, end, and words
        """
        segments = []
        for seg in transcription_result.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"],
                "words": seg.get("words", [])
            })
        return segments
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp as HH:MM:SS.
        
        Args:
            seconds: Timestamp in seconds
        
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None
) -> Dict:
    """
    Convenience function to transcribe audio.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        language: Language code or None for auto-detect
    
    Returns:
        Transcription result dictionary
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    return transcriber.transcribe(audio_path, language=language)
