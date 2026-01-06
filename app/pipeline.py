"""
Main processing pipeline.
Orchestrates all components to process video and generate enriched transcript.
"""

import os
import tempfile
from typing import Dict, List, Optional
from pathlib import Path

from ingestion.extract_audio import extract_audio, get_video_duration
from ingestion.extract_frames import extract_frames
from asr.sarvam_asr import SarvamTranscriber
from asr.diarization import SpeakerDiarizer
from audio_emotion.prosody import ProsodyAnalyzer
# from vision.facial_emotion import FacialEmotionDetector
from fusion.align_and_merge import MultimodalFusion
from llm.groq_reasoning import GroqReasoner


class VideoTranscriptionPipeline:
    """Main pipeline for video transcription and enrichment."""
    
    def __init__(
        self,
        sarvam_model: str = "saarika:v2.5",
        use_diarization: bool = True,
        use_llm: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            sarvam_model: Sarvam model version (default: saarika:v2.5)
            use_diarization: Whether to use speaker diarization
            use_llm: Whether to use LLM for refinement
        """
        self.sarvam_model = sarvam_model
        self.use_diarization = use_diarization
        self.use_llm = use_llm
        
        # Initialize components (lazy loading)
        self.transcriber = None
        self.diarizer = None
        self.prosody_analyzer = None
        self.emotion_detector = None
        self.fusion = MultimodalFusion()
        self.reasoner = None
    
    def process(self, video_path: str) -> Dict:
        """
        Process video file and generate enriched transcript.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with enriched transcript and metadata
        """
        print(f"Processing video: {video_path}")
        
        # Step 1: Extract audio and frames
        print("Step 1: Extracting audio and frames...")
        audio_path = extract_audio(video_path)
        frames = extract_frames(video_path, fps=2.0)
        duration = get_video_duration(video_path)
        
        print(f"  - Audio extracted: {audio_path}")
        print(f"  - Frames extracted: {len(frames)}")
        print(f"  - Duration: {duration:.2f}s")
        
        # Step 2: Transcribe audio
        print("Step 2: Transcribing audio with Sarvam API...")
        if self.transcriber is None:
            self.transcriber = SarvamTranscriber(model=self.sarvam_model)
        
        transcription_result = self.transcriber.transcribe(audio_path)
        transcription_segments = self.transcriber.get_segments(transcription_result)
        
        print(f"  - Segments transcribed: {len(transcription_segments)}")
        
        # Step 3: Speaker diarization
        if self.use_diarization:
            print("Step 3: Performing speaker diarization...")
            try:
                if self.diarizer is None:
                    self.diarizer = SpeakerDiarizer()
                
                diarization_segments = self.diarizer.diarize(audio_path)
                transcription_segments = self.diarizer.assign_speakers_to_segments(
                    transcription_segments,
                    diarization_segments
                )
                
                print(f"  - Speakers identified: {len(set(s.get('speaker') for s in transcription_segments))}")
            except Exception as e:
                print(f"  - Diarization failed: {e}. Continuing without speaker labels.")
                for seg in transcription_segments:
                    seg["speaker"] = "UNKNOWN"
        else:
            for seg in transcription_segments:
                seg["speaker"] = "UNKNOWN"
        
        # Step 4: Audio emotion analysis
        print("Step 4: Analyzing audio prosody...")
        if self.prosody_analyzer is None:
            self.prosody_analyzer = ProsodyAnalyzer()
        
        audio_emotions = []
        for seg in transcription_segments:
            try:
                prosody_features = self.prosody_analyzer.analyze_segment(
                    audio_path,
                    seg["start"],
                    seg["end"]
                )
                emotion = self.prosody_analyzer.infer_emotion_from_prosody(prosody_features)
                audio_emotions.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "emotion": emotion,
                    "confidence": 0.7,  # Basic confidence
                    "features": prosody_features
                })
            except Exception as e:
                print(f"  - Error analyzing segment {seg['start']:.2f}s: {e}")
                audio_emotions.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "emotion": "Neutral",
                    "confidence": 0.0,
                    "features": {}
                })
        
        print(f"  - Audio emotions analyzed: {len(audio_emotions)}")
        
        # # Step 5: Visual emotion detection
        # print("Step 5: Detecting facial emotions...")
        # if self.emotion_detector is None:
        #     self.emotion_detector = FacialEmotionDetector()
        
        # visual_emotions = self.emotion_detector.detect_emotions_batch(frames)
        
        # print(f"  - Visual emotions detected: {len([e for e in visual_emotions if e.get('emotion')])}")
        
        # Step 6: Multimodal fusion
        print("Step 6: Fusing multimodal data...")
        aligned_segments = self.fusion.align_data(
            transcription_segments,
            audio_emotions,
            None
        )
        
        # Merge emotions in each segment
        for seg in aligned_segments:
            merged_emotion = self.fusion.merge_emotions(
                seg.get("audio_emotion"),
                seg.get("visual_emotion")
            )
            seg["merged_emotion"] = merged_emotion
        
        print(f"  - Segments aligned: {len(aligned_segments)}")
        
        # Step 7: LLM refinement
        if self.use_llm:
            print("Step 7: Refining with LLM...")
            try:
                if self.reasoner is None:
                    self.reasoner = GroqReasoner()
                
                aligned_segments = self.reasoner.refine_emotions_and_reactions(aligned_segments)
                summary = self.reasoner.generate_summary(aligned_segments)
                
                print("  - LLM refinement complete")
            except Exception as e:
                print(f"  - LLM refinement failed: {e}. Using basic metadata.")
                summary = {
                    "total_speakers": len(set(s.get("speaker") for s in aligned_segments)),
                    "total_segments": len(aligned_segments),
                    "dominant_emotions": ["Neutral"],
                    "summary": "Summary unavailable"
                }
        else:
            # Basic summary without LLM
            summary = {
                "total_speakers": len(set(s.get("speaker") for s in aligned_segments)),
                "total_segments": len(aligned_segments),
                "dominant_emotions": list(set(
                    s.get("merged_emotion", {}).get("primary_emotion", "Neutral")
                    for s in aligned_segments
                ))[:5],
                "summary": "Basic summary"
            }
        
        # Cleanup temp audio file
        try:
            if os.path.exists(audio_path) and tempfile.gettempdir() in audio_path:
                os.remove(audio_path)
        except:
            pass
        
        # Format final output
        formatted_segments = []
        for seg in aligned_segments:
            formatted_segments.append({
                "timestamp": seg.get("timestamp", ""),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg.get("text", ""),
                "emotion": seg.get("emotion", "Neutral"),
                "tone": seg.get("tone", "Neutral"),
                "reaction": seg.get("reaction", "No clear reaction"),
                "audio_confidence": seg.get("merged_emotion", {}).get("audio_confidence", 0.0),
                "visual_confidence": seg.get("merged_emotion", {}).get("visual_confidence", 0.0)
            })
        
        return {
            "transcript": formatted_segments,
            "summary": summary,
            "metadata": {
                "duration": duration,
                "total_segments": len(formatted_segments),
                "video_path": video_path
            }
        }
    
    def format_transcript_text(self, result: Dict) -> str:
        """
        Format transcript as readable text.
        
        Args:
            result: Pipeline result dictionary
        
        Returns:
            Formatted text string
        """
        lines = []
        
        for seg in result["transcript"]:
            line = f"[{seg['timestamp']}] {seg['speaker']}: \"{seg['text']}\""
            line += f"\n[Emotion: {seg['emotion']}, Tone: {seg['tone']}, Reaction: {seg['reaction']}]"
            lines.append(line)
        
        return "\n\n".join(lines)


def process_video(video_path: str, **kwargs) -> Dict:
    """
    Convenience function to process video.
    
    Args:
        video_path: Path to video file
        **kwargs: Additional pipeline arguments
    
    Returns:
        Processing result dictionary
    """
    pipeline = VideoTranscriptionPipeline(**kwargs)
    return pipeline.process(video_path)
