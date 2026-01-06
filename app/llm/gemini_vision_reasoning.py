"""
Gemini Vision reasoning module.
Uses Google Gemini Vision API for joint visual-language reasoning.
Replaces text-only LLM with multimodal analysis.
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from collections import deque
import base64
import io
from PIL import Image
import numpy as np
import dotenv

dotenv.load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiVisionReasoner:
    """Uses Google Gemini Vision for multimodal emotion refinement."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini Vision client.
        
        Args:
            api_key: Google API key (or from GEMINI_API_KEY env var)
            model: Model name (default: env GEMINI_MODEL or gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it in .env file or environment. "
                "Get key from: https://makersuite.google.com/app/apikey"
            )
        
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(self.model_name)
        
        # Rate limiting: max 5 requests per minute
        self.rate_limit = 5
        self.rate_window = 60  # seconds
        self.request_times = deque()
        
        # Cache: keyed by (transcript_text + frame_hash)
        self.cache = {}
        
        # Fallback reasoner (text-only)
        self.fallback_reasoner = None
    
    def refine_emotions_and_reactions(
        self,
        segments: List[Dict],
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[Dict]:
        """
        Use Gemini Vision to refine emotions and classify reactions.
        
        Args:
            segments: List of aligned segments with raw emotion data
            frames: List of (timestamp, frame_array) tuples
        
        Returns:
            Segments with refined emotions and reactions
        """
        # Drop any null/invalid segments early to avoid attribute errors
        clean_segments = [s for s in segments if isinstance(s, dict)]

        # Select frames for each segment and check for faces
        segments_with_frames = self._prepare_segments_with_frames(clean_segments, frames)
        
        # Batch segments for efficient processing
        batched_segments = self._batch_segments(segments_with_frames)
        
        refined_segments = []
        
        for batch in batched_segments:
            try:
                print("batch", batch)
                refined_batch = self._process_batch_with_gemini(batch)
                refined_segments.extend(refined_batch)
            except Exception as e:
                print(f"Gemini processing failed: {e}. Using fallback.")
                # Fallback to text-only logic
                refined_batch = self._fallback_refinements([s["segment"] for s in batch])
                refined_segments.extend(refined_batch)
        
        return refined_segments
    
    def _prepare_segments_with_frames(
        self,
        segments: List[Dict],
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[Dict]:
        """Prepare segments with selected frames and face detection."""
        segments_with_frames = []
        
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", seg_start + 1.0)
            seg_midpoint = (seg_start + seg_end) / 2
            
            # Find closest frame to midpoint
            selected_frame = None
            min_time_diff = float('inf')
            
            for frame_time, frame_array in frames:
                time_diff = abs(frame_time - seg_midpoint)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    selected_frame = (frame_time, frame_array)
            
            # Check if face is detected (simple heuristic: frame exists and is close enough)
            # Note: Actual face detection would require additional processing
            # For now, we assume frames contain faces if they're close to segment
            has_face = selected_frame is not None and min_time_diff < 2.0
            
            segments_with_frames.append({
                "segment": seg,
                "frame": selected_frame,
                "has_face": has_face
            })
        
        return segments_with_frames
    
    def _batch_segments(self, segments_with_frames: List[Dict], max_batch_size: int = 5) -> List[List[Dict]]:
        """Batch segments for efficient Gemini API calls."""
        batches = []
        current_batch = []
        
        for seg_data in segments_with_frames:
            # Skip if no face detected
            if not seg_data["has_face"]:
                # Add to batch anyway, but will use text-only
                current_batch.append(seg_data)
            else:
                current_batch.append(seg_data)
            
            # Create batch when we have enough segments or hit rate limit
            if len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _process_batch_with_gemini(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of segments with Gemini Vision."""
        # Check rate limit
        self._check_rate_limit()
        
        # Prepare content for Gemini
        contents = []
        
        for seg_data in batch:
            seg = seg_data["segment"]
            if not isinstance(seg, dict):
                # Skip malformed segment
                continue
            frame_data = seg_data.get("frame")
            has_face = seg_data.get("has_face", False)
            
            text = seg.get("text", "")
            speaker = seg.get("speaker", "UNKNOWN")
            audio_emotion = seg.get("audio_emotion", {}).get("emotion") if seg.get("audio_emotion") else None
            
            # Create cache key
            cache_key = self._create_cache_key(text, frame_data)
            
            # Check cache (will be handled in processing loop)
            
            # Prepare prompt
            prompt = self._create_gemini_prompt(text, speaker, audio_emotion)
            
            # Prepare content parts
            parts = [prompt]
            
            # Add frame if available and face detected
            if has_face and frame_data is not None:
                _, frame_array = frame_data
                # Convert frame to PIL Image
                if isinstance(frame_array, np.ndarray):
                    # Ensure RGB format
                    if len(frame_array.shape) == 3:
                        if frame_array.shape[2] == 4:  # RGBA
                            frame_array = frame_array[:, :, :3]
                        frame_image = Image.fromarray(frame_array.astype('uint8'))
                        parts.append(frame_image)
            
            contents.append({
                "parts": parts,
                "cache_key": cache_key,
                "segment": seg
            })
        
        # Process with Gemini (batch API call)
        refined_segments = []
        
        for content_data in contents:
            # Cache check
            if content_data["cache_key"] in self.cache:
                cached = self.cache[content_data["cache_key"]]
                base_seg = content_data.get("segment") or {}
                if not isinstance(base_seg, dict):
                    base_seg = {}
                refined_seg = base_seg.copy()
                refined_seg.update(cached)
                refined_segments.append(refined_seg)
                continue

            try:
                # Check rate limit before each request
                self._check_rate_limit()
                
                # Call Gemini with JSON-only response
                response = self.model.generate_content(
                    content_data["parts"],
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 300,
                        "response_mime_type": "application/json",
                    }
                )
                
                # Extract text safely (handle no parts / finish_reason)
                result_text = self._extract_response_text(response)
                
                # If still empty, raise to trigger fallback
                if not result_text.strip():
                    raise ValueError("Empty Gemini response")
                
                # Parse response
                result = self._parse_gemini_response(result_text)
                
                # Cache result
                self.cache[content_data["cache_key"]] = result
                
                # Merge with segment
                base_seg = content_data.get("segment") or {}
                if not isinstance(base_seg, dict):
                    
                    base_seg = {}
                refined_seg = base_seg.copy()
                refined_seg.update(result)
                refined_segments.append(refined_seg)
                
                # Record request time
                self.request_times.append(time.time())
                
            except Exception as e:
                # Fallback for this segment
                base_seg = content_data.get("segment") or {}
                if not isinstance(base_seg, dict):
                    base_seg = {}
                refined_seg = base_seg.copy()
                fallback = self._fallback_refinements([refined_seg])[0]
                refined_seg.update(fallback)
                refined_segments.append(refined_seg)
        
        return refined_segments
    
    def _create_cache_key(self, text: str, frame_data: Optional[Tuple[float, np.ndarray]]) -> str:
        """Create cache key from text and frame."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        
        if frame_data is not None:
            _, frame_array = frame_data
            # Create hash from frame (sample pixels for efficiency)
            frame_sample = frame_array[::50, ::50] if isinstance(frame_array, np.ndarray) else None
            if frame_sample is not None:
                frame_hash = hashlib.md5(frame_sample.tobytes()).hexdigest()[:8]
            else:
                frame_hash = "no_frame"
        else:
            frame_hash = "no_frame"
        
        return f"{text_hash}_{frame_hash}"
    
    def _create_gemini_prompt(self, text: str, speaker: str, audio_emotion: Optional[str]) -> str:
        """Create prompt for Gemini Vision."""
        prompt = f"""Analyze this conversation segment and provide emotion, tone, and reaction classification.

Segment:
- Speaker: {speaker}
- Text: "{text}"
- Audio Emotion (raw): {audio_emotion or "Unknown"}

Analyze the text and visual cues (if frame provided) to determine:
1. emotion: One of [Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral, Excitement, Contemplation]
2. tone: Brief description (e.g., "Enthusiastic", "Hesitant", "Confident", "Uncertain")
3. reaction: Classification (e.g., "Positive engagement", "Neutral response", "Question", "Agreement", "Disagreement", "No clear reaction")
4. visual_cues: List of visual observations (e.g., ["smiling", "gesturing", "looking away"])

Return ONLY valid JSON, no explanations, no markdown:
{{
  "emotion": "string",
  "tone": "string",
  "reaction": "string",
  "visual_cues": ["string"]
}}"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        try:
            if not response_text or not response_text.strip():
                raise ValueError("Empty response")
    
            text = response_text.strip()
    
            if "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
    
            parsed = json.loads(text)
    
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
    
            if not isinstance(parsed, dict):
                raise ValueError("Response is not dict")
    
            return {
                "emotion": parsed.get("emotion", "Neutral"),
                "tone": parsed.get("tone", "Neutral"),
                "reaction": parsed.get("reaction", "No clear reaction"),
                "visual_cues": parsed.get("visual_cues", []) or []
            }
    
        except Exception as e:
            print(f"Failed to parse Gemini response: {e}")
            return {
                "emotion": "Neutral",
                "tone": "Neutral",
                "reaction": "No clear reaction",
                "visual_cues": []
            }


    def _extract_response_text(self, response) -> str:
        if not response:
            return ""

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if hasattr(response, "candidates"):
            for cand in response.candidates or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", [])
                text = "".join(
                    p.text for p in parts if hasattr(p, "text") and p.text
                )
                if text.strip():
                    return text.strip()

        return ""

    
    def _check_rate_limit(self):
        """Check and enforce rate limit (5 requests per minute)."""
        current_time = time.time()
        
        # Remove requests older than rate window
        while self.request_times and self.request_times[0] < current_time - self.rate_window:
            self.request_times.popleft()
        
        # If at limit, wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time = self.rate_window - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
    
    def _fallback_refinements(self, segments: List[Dict]) -> List[Dict]:
        """Fallback when Gemini fails - use basic logic."""
        fallback = []
        
        for seg in segments:
            if not isinstance(seg, dict):
                # Skip completely invalid entries
                continue

            fallback_seg = seg.copy()
            
            # Use primary emotion from fusion, guarding against None values
            audio_emotion = seg.get("audio_emotion") or {}
            visual_emotion = seg.get("visual_emotion") or {}
            text = seg.get("text", "") or ""
            
            # Heuristic text sentiment/tone if nothing else is available
            heuristic = self._heuristic_from_text(text)

            emotion = (
                visual_emotion.get("emotion")
                or audio_emotion.get("emotion")
                or (heuristic.get("emotion") if heuristic else None)
                or "Neutral"
            )
            
            fallback_seg["emotion"] = emotion
            fallback_seg["tone"] = heuristic.get("tone") if heuristic else "Neutral"
            fallback_seg["reaction"] = heuristic.get("reaction") if heuristic else "No clear reaction"
            fallback_seg["visual_cues"] = []
            
            fallback.append(fallback_seg)
        
        return fallback

    def _heuristic_from_text(self, text: str) -> Dict:
        """
        Simple keyword-based heuristic for emotion/tone/reaction when LLM is unavailable.
        """
        if not text:
            return {}

        t = text.lower()

        # Joy / positivity keywords
        joy_keywords = ["happy", "glad", "excited", "grateful", "gratitude", "thank", "love", "proud", "thrilled"]
        sad_keywords = ["sad", "down", "unhappy", "depressed", "upset"]
        anger_keywords = ["angry", "mad", "furious", "frustrated"]
        fear_keywords = ["afraid", "scared", "worried", "anxious"]
        surprise_keywords = ["surprised", "astonished", "amazed"]
        disgust_keywords = ["disgusted", "gross", "nasty"]

        emotion = None
        tone = None
        reaction = None

        if any(k in t for k in joy_keywords):
            emotion = "Joy"
            tone = "Positive / Enthusiastic"
            reaction = "Positive engagement"
        elif any(k in t for k in sad_keywords):
            emotion = "Sadness"
            tone = "Somber"
            reaction = "No clear reaction"
        elif any(k in t for k in anger_keywords):
            emotion = "Anger"
            tone = "Frustrated"
            reaction = "Disagreement"
        elif any(k in t for k in fear_keywords):
            emotion = "Fear"
            tone = "Apprehensive"
            reaction = "No clear reaction"
        elif any(k in t for k in surprise_keywords):
            emotion = "Surprise"
            tone = "Surprised"
            reaction = "No clear reaction"
        elif any(k in t for k in disgust_keywords):
            emotion = "Disgust"
            tone = "Displeased"
            reaction = "No clear reaction"

        # If we inferred anything, return it
        if emotion or tone or reaction:
            return {
                "emotion": emotion or "Neutral",
                "tone": tone or "Neutral",
                "reaction": reaction or "No clear reaction",
            }

        return {}
    
    def generate_summary(self, segments: List[Dict]) -> Dict:
        """
        Generate conversation summary.
        
        Args:
            segments: All processed segments
        
        Returns:
            Summary dictionary
        """
        # Count speakers and emotions
        speakers = set()
        emotions = []
        
        for seg in segments:
            speakers.add(seg.get("speaker", "UNKNOWN"))
            emotions.append(seg.get("emotion", "Neutral"))
        
        return {
            "total_speakers": len(speakers),
            "total_segments": len(segments),
            "dominant_emotions": list(set(emotions))[:5],
            "summary": f"Conversation with {len(speakers)} speakers. Dominant emotions: {', '.join(set(emotions[:5]))}."
        }


def refine_with_gemini_vision(
    segments: List[Dict],
    frames: List[Tuple[float, np.ndarray]],
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function to refine segments with Gemini Vision.
    
    Args:
        segments: Aligned segments
        frames: Video frames with timestamps
        api_key: Optional Gemini API key
    
    Returns:
        Refined segments
    """
    reasoner = GeminiVisionReasoner(api_key=api_key)
    return reasoner.refine_emotions_and_reactions(segments, frames)
