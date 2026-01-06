"""
Speaker diarization module.
Uses clustering-based approach with librosa and scikit-learn.
Free, open-source, and doesn't require external API keys.
"""

import os
import numpy as np
from typing import List, Dict, Optional
import librosa
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SpeakerDiarizer:
    """Simple clustering-based speaker diarization."""
    
    def __init__(self, num_speakers: Optional[int] = None, method: str = "kmeans"):
        """
        Initialize diarization.
        
        Args:
            num_speakers: Number of speakers (auto-detect if None)
            method: Clustering method ('kmeans' or 'agglomerative')
        """
        self.num_speakers = num_speakers
        self.method = method
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of dictionaries with speaker, start, end times
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            # Extract features for diarization
            # Use MFCC features which capture speaker characteristics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Also use spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Combine features
            # Transpose to get time frames as rows
            mfccs_t = mfccs.T
            
            # Resample other features to match MFCC frame count
            frame_count = mfccs_t.shape[0]
            spectral_centroids_resampled = np.interp(
                np.linspace(0, len(spectral_centroids) - 1, frame_count),
                np.arange(len(spectral_centroids)),
                spectral_centroids
            )
            zcr_resampled = np.interp(
                np.linspace(0, len(zero_crossing_rate) - 1, frame_count),
                np.arange(len(zero_crossing_rate)),
                zero_crossing_rate
            )
            
            # Combine all features
            features = np.column_stack([
                mfccs_t,
                spectral_centroids_resampled.reshape(-1, 1),
                zcr_resampled.reshape(-1, 1)
            ])
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine number of speakers
            if self.num_speakers is None:
                # Auto-detect: try 2-4 speakers, choose best based on silhouette
                best_n = 2
                best_score = -1
                
                for n in range(2, min(5, frame_count // 10 + 1)):  # At least 10 frames per speaker
                    try:
                        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(features_scaled)
                        
                        # Simple score: within-cluster variance
                        score = -kmeans.inertia_
                        
                        if score > best_score:
                            best_score = score
                            best_n = n
                    except:
                        continue
                
                n_speakers = best_n
            else:
                n_speakers = self.num_speakers
            
            # Perform clustering
            if self.method == "kmeans":
                clusterer = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            else:
                clusterer = AgglomerativeClustering(n_clusters=n_speakers)
            
            labels = clusterer.fit_predict(features_scaled)
            
            # Convert frame labels to time segments
            hop_length = 512
            frame_duration = hop_length / sr
            
            segments = []
            current_speaker = labels[0]
            segment_start = 0.0
            
            for i in range(1, len(labels)):
                frame_time = i * frame_duration
                
                # If speaker changes, create a segment
                if labels[i] != current_speaker:
                    if frame_time - segment_start > 0.1:  # Minimum segment duration
                        segments.append({
                            "speaker": f"SPEAKER_{current_speaker:02d}",
                            "start": segment_start,
                            "end": frame_time,
                            "duration": frame_time - segment_start
                        })
                    segment_start = frame_time
                    current_speaker = labels[i]
            
            # Add final segment
            if duration - segment_start > 0.1:
                segments.append({
                    "speaker": f"SPEAKER_{current_speaker:02d}",
                    "start": segment_start,
                    "end": duration,
                    "duration": duration - segment_start
                })
            
            # Merge very short segments with adjacent ones
            segments = self._merge_short_segments(segments, min_duration=0.5)
            
            return segments
        
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")
    
    def _merge_short_segments(self, segments: List[Dict], min_duration: float = 0.5) -> List[Dict]:
        """
        Merge segments shorter than min_duration with adjacent segments.
        
        Args:
            segments: List of segments
            min_duration: Minimum segment duration in seconds
        
        Returns:
            Merged segments
        """
        if not segments:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i].copy()
            
            # If segment is too short, try to merge with next
            if current["duration"] < min_duration and i < len(segments) - 1:
                next_seg = segments[i + 1]
                # Merge with next segment
                current["end"] = next_seg["end"]
                current["duration"] = current["end"] - current["start"]
                current["speaker"] = next_seg["speaker"]  # Use next speaker's label
                i += 1  # Skip next segment
            
            merged.append(current)
            i += 1
        
        return merged
    
    def assign_speakers_to_segments(
        self,
        transcription_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Assign speaker labels to transcription segments.
        
        Args:
            transcription_segments: Segments from ASR (with start/end times)
            diarization_segments: Segments from diarization (with speaker labels)
        
        Returns:
            Transcription segments with added speaker labels
        """
        # Assign speakers to transcription segments
        enriched_segments = []
        
        for trans_seg in transcription_segments:
            seg_start = trans_seg["start"]
            seg_end = trans_seg["end"]
            seg_midpoint = (seg_start + seg_end) / 2
            
            # Find speaker with maximum overlap
            assigned_speaker = None
            max_overlap = 0.0
            
            for diar_seg in diarization_segments:
                # Check if transcription segment overlaps with diarization segment
                if not (seg_end < diar_seg["start"] or seg_start > diar_seg["end"]):
                    # Calculate overlap
                    overlap_start = max(seg_start, diar_seg["start"])
                    overlap_end = min(seg_end, diar_seg["end"])
                    overlap = overlap_end - overlap_start
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_speaker = diar_seg["speaker"]
            
            # Fallback: find closest speaker segment
            if assigned_speaker is None:
                min_distance = float('inf')
                for diar_seg in diarization_segments:
                    diar_midpoint = (diar_seg["start"] + diar_seg["end"]) / 2
                    distance = abs(seg_midpoint - diar_midpoint)
                    if distance < min_distance:
                        min_distance = distance
                        assigned_speaker = diar_seg["speaker"]
            
            # Create enriched segment
            enriched_seg = trans_seg.copy()
            enriched_seg["speaker"] = assigned_speaker or "UNKNOWN"
            enriched_segments.append(enriched_seg)
        
        return enriched_segments


def diarize_audio(audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to diarize audio.
    
    Args:
        audio_path: Path to audio file
        num_speakers: Optional number of speakers (auto-detect if None)
    
    Returns:
        List of diarization segments
    """
    diarizer = SpeakerDiarizer(num_speakers=num_speakers)
    return diarizer.diarize(audio_path)
