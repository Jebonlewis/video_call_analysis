"""
Multimodal fusion module.
Aligns and merges transcription, speaker, audio emotion, and visual emotion data.
"""

from typing import List, Dict, Optional


class MultimodalFusion:
    """Fuses multimodal data by timestamp alignment."""
    
    def __init__(self, time_tolerance: float = 0.5):
        """
        Initialize fusion module.
        
        Args:
            time_tolerance: Maximum time difference for alignment (seconds)
        """
        self.time_tolerance = time_tolerance
    
    def align_data(
        self,
        transcription_segments: List[Dict],
        audio_emotions: List[Dict],
        visual_emotions: List[Dict]
    ) -> List[Dict]:
        """
        Align transcription, audio emotion, and visual emotion by timestamps.
        
        Args:
            transcription_segments: Transcription segments with start/end times
            audio_emotions: Audio emotion features per segment
            visual_emotions: Visual emotion detections with timestamps
        
        Returns:
            List of aligned segments with all metadata
        """
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            seg_start = trans_seg["start"]
            seg_end = trans_seg["end"]
            seg_midpoint = (seg_start + seg_end) / 2
            
            # Find matching audio emotion (should be same segment)
            audio_emotion = None
            for audio_em in audio_emotions:
                if (audio_em.get("start", 0) <= seg_midpoint <= audio_em.get("end", float('inf'))):
                    audio_emotion = audio_em
                    break
            
            # Find closest visual emotion
            visual_emotion = self._find_closest_visual_emotion(
                visual_emotions,
                seg_midpoint
            )
            
            # Create aligned segment
            aligned_seg = {
                "text": trans_seg.get("text", ""),
                "speaker": trans_seg.get("speaker", "UNKNOWN"),
                "start": seg_start,
                "end": seg_end,
                "timestamp": self._format_timestamp(seg_start),
                "audio_emotion": audio_emotion,
                "visual_emotion": visual_emotion,
                "words": trans_seg.get("words", [])
            }
            
            aligned_segments.append(aligned_seg)
        
        return aligned_segments
    
    def _find_closest_visual_emotion(
        self,
        visual_emotions: List[Dict],
        target_timestamp: float
    ) -> Optional[Dict]:
        """
        Find visual emotion closest to target timestamp.
        
        Args:
            visual_emotions: List of visual emotion detections
            target_timestamp: Target timestamp
        
        Returns:
            Closest visual emotion or None
        """
        if not visual_emotions:
            return None
        
        closest = None
        min_diff = float('inf')
        
        for vis_em in visual_emotions:
            timestamp = vis_em.get("timestamp", 0)
            diff = abs(timestamp - target_timestamp)
            
            if diff < min_diff and diff <= self.time_tolerance:
                min_diff = diff
                closest = vis_em
        
        return closest
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def merge_emotions(
        self,
        audio_emotion: Optional[Dict],
        visual_emotion: Optional[Dict]
    ) -> Dict:
        """
        Merge audio and visual emotion predictions.
        
        Args:
            audio_emotion: Audio emotion data
            visual_emotion: Visual emotion data
        
        Returns:
            Merged emotion dictionary
        """
        merged = {
            "audio_emotion": audio_emotion.get("emotion") if audio_emotion else None,
            "audio_confidence": audio_emotion.get("confidence", 0.0) if audio_emotion else 0.0,
            "visual_emotion": visual_emotion.get("emotion") if visual_emotion else None,
            "visual_confidence": visual_emotion.get("confidence", 0.0) if visual_emotion else 0.0,
            "all_visual_emotions": visual_emotion.get("all_emotions", {}) if visual_emotion else {}
        }
        
        # Simple fusion: prefer visual if high confidence, else audio
        if merged["visual_confidence"] > 0.7:
            merged["primary_emotion"] = merged["visual_emotion"]
            merged["primary_source"] = "visual"
        elif merged["audio_confidence"] > 0.5:
            merged["primary_emotion"] = merged["audio_emotion"]
            merged["primary_source"] = "audio"
        else:
            merged["primary_emotion"] = merged["visual_emotion"] or merged["audio_emotion"] or "Neutral"
            merged["primary_source"] = "combined"
        
        return merged


def align_multimodal_data(
    transcription_segments: List[Dict],
    audio_emotions: List[Dict],
    visual_emotions: List[Dict]
) -> List[Dict]:
    """
    Convenience function to align multimodal data.
    
    Args:
        transcription_segments: Transcription segments
        audio_emotions: Audio emotion features
        visual_emotions: Visual emotion detections
    
    Returns:
        Aligned segments
    """
    fusion = MultimodalFusion()
    return fusion.align_data(transcription_segments, audio_emotions, visual_emotions)
