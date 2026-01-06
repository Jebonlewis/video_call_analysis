"""
Sarvam ASR module for speech transcription.
Uses Sarvam AI API for high-quality speech-to-text.
"""

import os
import json
from typing import List, Dict, Optional
import requests
import dotenv
dotenv.load_dotenv()

# Try to import Sarvam SDK, fallback to direct API calls
try:
    from sarvamai import SarvamAI
    SARVAM_SDK_AVAILABLE = True
except ImportError:
    SARVAM_SDK_AVAILABLE = False


class SarvamTranscriber:
    """Wrapper for Sarvam AI ASR API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "saarika:v2.5"):
        """
        Initialize Sarvam API client.
        
        Args:
            api_key: Sarvam API key (or from SARVAM_API_KEY env var)
            model: Model version (default: saarika:v2.5)
        """
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY environment variable not set. "
                "Please set it in .env file or environment. "
                "Get key from: https://dashboard.sarvam.ai/signin"
            )
        
        self.model = model
        
        # Try to use SDK if available
        if SARVAM_SDK_AVAILABLE:
            try:
                self.client = SarvamAI(api_subscription_key=self.api_key)
                self.use_sdk = True
            except Exception:
                self.use_sdk = False
                self.api_url = "https://api.sarvam.ai/v1/speech-to-text"
        else:
            self.use_sdk = False
            self.api_url = "https://api.sarvam.ai/speech-to-text"
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio file using Sarvam API.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if not provided)
            task: 'transcribe' or 'translate' (default: transcribe)
        
        Returns:
            Dictionary with transcription results including segments with timestamps
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Use SDK if available
            if self.use_sdk:
                return self._transcribe_with_sdk(audio_path, language)
            else:
                return self._transcribe_with_api(audio_path, language)
        
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _transcribe_with_sdk(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using Sarvam SDK."""
        try:
            with open(audio_path, "rb") as audio_file:
                # Try different SDK method signatures
                try:
                    # Method 1: speech_to_text.transcribe
                    response = self.client.speech_to_text.transcribe(
                        file=audio_file,
                        model=self.model
                    )
                except AttributeError:
                    try:
                        # Method 2: speech_to_text
                        response = self.client.speech_to_text(
                            audio=audio_file,
                            language_code=language or "en-IN",
                            model=self.model
                        )
                    except AttributeError:
                        # Method 3: Direct call
                        response = self.client.transcribe(
                            audio_file=audio_file,
                            model=self.model
                        )
            
            # Get audio duration
            import librosa
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
            except:
                duration = 0.0
            
            return self._format_result(response, audio_path, duration)
        except Exception as e:
            # Fallback to API if SDK fails
            return self._transcribe_with_api(audio_path, language)
    
    def _transcribe_with_api(self, audio_path: str, language: Optional[str] = None) -> Dict:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_path), audio_file, "audio/wav")
            }

            data = {
                "model": "saarika:v2.5"
            }

            if language:
                data["language_code"] = language  # en-IN

            headers = {
                "api-subscription-key": self.api_key
            }

            response = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=300
            )

            response.raise_for_status()
            result = response.json()
            
            # Get audio duration for segment creation
            import librosa
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
            except:
                duration = 0.0
            
            # Format result to match Whisper-like structure
            return self._format_result(result, audio_path, duration)

    
    def _format_result(self, api_result: Dict, audio_path: str, duration: float = 0.0) -> Dict:
        """
        Format Sarvam API result to match Whisper-like structure.
        
        Args:
            api_result: Raw API response
            audio_path: Path to audio file (for reference)
            duration: Audio duration in seconds
        
        Returns:
            Formatted result dictionary
        """
        # Extract transcript text
        transcript_text = api_result.get("transcript", "") or api_result.get("text", "")
        
        if not transcript_text:
            # If no transcript, return empty result
            return {
                "text": "",
                "language": api_result.get("language_code", "unknown"),
                "segments": []
            }
        
        # Try to extract segments if available
        segments = api_result.get("segments", [])
        
        # If no segments, split transcript into sentences and create segments
        if not segments and transcript_text:
            # Split transcript into sentences (simple approach)
            sentences = self._split_into_sentences(transcript_text)
            
            # Distribute sentences evenly across duration
            if duration > 0 and len(sentences) > 0:
                time_per_sentence = duration / len(sentences)
                segments = []
                current_time = 0.0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        segment_duration = time_per_sentence
                        segments.append({
                            "text": sentence,
                            "start": current_time,
                            "end": current_time + segment_duration,
                            "words": []
                        })
                        current_time += segment_duration
            else:
                # Fallback: single segment
                segments = [{
                    "text": transcript_text,
                    "start": 0.0,
                    "end": duration if duration > 0 else 10.0,  # Default 10s if duration unknown
                    "words": []
                }]
        
        # Format segments to match Whisper structure
        formatted_segments = []
        for seg in segments:
            if isinstance(seg, dict):
                formatted_seg = {
                    "text": seg.get("text", "").strip(),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", seg.get("start", 0.0) + 1.0)),
                    "words": seg.get("words", [])
                }
            else:
                # Handle case where seg might be a string
                formatted_seg = {
                    "text": str(seg).strip(),
                    "start": 0.0,
                    "end": duration if duration > 0 else 1.0,
                    "words": []
                }
            
            if formatted_seg["text"]:  # Only add non-empty segments
                formatted_segments.append(formatted_seg)
        
        return {
            "text": transcript_text,
            "language": api_result.get("language_code", api_result.get("language", "unknown")),
            "segments": formatted_segments
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        import re
        # Simple sentence splitting on periods, exclamation, question marks
        # Split on sentence endings followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned.append(sent)
        
        # If no sentences found, return the whole text
        if not cleaned:
            cleaned = [text]
        
        return cleaned
    
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
    model: str = "saarika:v2.5",
    language: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    Convenience function to transcribe audio.
    
    Args:
        audio_path: Path to audio file
        model: Sarvam model version
        language: Language code or None for auto-detect
        api_key: Optional API key
    
    Returns:
        Transcription result dictionary
    """
    transcriber = SarvamTranscriber(api_key=api_key, model=model)
    return transcriber.transcribe(audio_path, language=language)
