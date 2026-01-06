"""
Audio prosody analysis module.
Extracts pitch, tempo, energy, and other prosodic features for emotion inference.
"""

import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class ProsodyAnalyzer:
    """Analyzes audio prosody features for emotion detection."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize prosody analyzer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def analyze_file(self, audio_path: str) -> Dict:
        """
        Analyze prosody features of entire audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with global prosody features
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return self.analyze_audio(y, sr)
    
    def analyze_audio(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analyze prosody features from audio array.
        
        Args:
            y: Audio time series
            sr: Sample rate
        
        Returns:
            Dictionary with prosody features
        """
        features = {}
        
        # Pitch (F0) - fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features["pitch_mean"] = float(np.mean(f0_clean))
            features["pitch_std"] = float(np.std(f0_clean))
            features["pitch_range"] = float(np.max(f0_clean) - np.min(f0_clean))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_range"] = 0.0
        
        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo)
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
        features["energy_max"] = float(np.max(rms))
        
        # Zero crossing rate (speech activity indicator)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        
        # Spectral rolloff (high frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        
        # Duration
        features["duration"] = len(y) / sr
        
        return features
    
    def analyze_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        Analyze prosody features for a specific time segment.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Dictionary with prosody features for segment
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_segment = y[start_sample:end_sample]
        
        if len(y_segment) == 0:
            return self._empty_features()
        
        return self.analyze_audio(y_segment, sr)
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary."""
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0,
            "tempo": 0.0,
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "energy_max": 0.0,
            "zcr_mean": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "duration": 0.0
        }
    
    def infer_emotion_from_prosody(self, features: Dict) -> str:
        """
        Infer emotion from prosody features (simple rule-based).
        This is a basic heuristic - LLM will refine this later.
        
        Args:
            features: Prosody features dictionary
        
        Returns:
            Inferred emotion label
        """
        pitch_mean = features.get("pitch_mean", 0)
        energy_mean = features.get("energy_mean", 0)
        tempo = features.get("tempo", 0)
        pitch_std = features.get("pitch_std", 0)
        
        # Simple heuristics
        if pitch_mean > 200 and energy_mean > 0.1 and tempo > 120:
            return "Excitement"
        elif pitch_mean > 180 and energy_mean > 0.08:
            return "Joy"
        elif pitch_mean < 120 and energy_mean < 0.05:
            return "Sadness"
        elif pitch_std > 50:
            return "Surprise"
        elif energy_mean < 0.03:
            return "Neutral"
        else:
            return "Neutral"


def analyze_prosody(audio_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict:
    """
    Convenience function to analyze prosody.
    
    Args:
        audio_path: Path to audio file
        start_time: Optional start time for segment analysis
        end_time: Optional end time for segment analysis
    
    Returns:
        Prosody features dictionary
    """
    analyzer = ProsodyAnalyzer()
    
    if start_time is not None and end_time is not None:
        return analyzer.analyze_segment(audio_path, start_time, end_time)
    else:
        return analyzer.analyze_file(audio_path)
