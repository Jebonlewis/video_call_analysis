"""
LLM reasoning module using Groq API.
Refines emotions, classifies reactions, and generates final metadata.
"""

import os
from typing import List, Dict, Optional
from groq import Groq
import json


class GroqReasoner:
    """Uses Groq LLM for emotion refinement and metadata synthesis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (or from GROQ_API_KEY env var)
            model: Model name (default: llama3-70b-8192)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Please set it in .env file or environment. "
                "Get key from: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
    
    def refine_emotions_and_reactions(
        self,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Use LLM to refine emotions and classify reactions for all segments.
        
        Args:
            segments: List of aligned segments with raw emotion data
        
        Returns:
            Segments with refined emotions and reactions
        """
        # Process in batches to avoid token limits
        batch_size = 10
        refined_segments = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            refined_batch = self._process_batch(batch)
            refined_segments.extend(refined_batch)
        
        return refined_segments
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of segments."""
        # Prepare context for LLM
        context = self._prepare_context(batch)
        
        # Create prompt
        prompt = self._create_reasoning_prompt(context)
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing conversation transcripts and inferring emotions, tone, and reactions. You receive structured data about speech segments and must provide refined emotion labels, tone descriptions, and reaction classifications."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            refined_data = self._parse_llm_response(result_text, batch)
            
            # Merge with original segments
            return self._merge_refinements(batch, refined_data)
        
        except Exception as e:
            # Fallback: return segments with basic metadata
            print(f"LLM processing failed: {e}. Using fallback metadata.")
            return self._fallback_refinements(batch)
    
    def _prepare_context(self, segments: List[Dict]) -> str:
        """Prepare context string for LLM."""
        context_parts = []
        
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            speaker = seg.get("speaker", "UNKNOWN")
            audio_emotion = seg.get("audio_emotion", {}).get("emotion") if seg.get("audio_emotion") else None
            visual_emotion = seg.get("visual_emotion", {}).get("emotion") if seg.get("visual_emotion") else None
            
            context_parts.append(
                f"Segment {i+1}:\n"
                f"  Speaker: {speaker}\n"
                f"  Text: \"{text}\"\n"
                f"  Audio Emotion (raw): {audio_emotion}\n"
                f"  Visual Emotion (raw): {visual_emotion}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_reasoning_prompt(self, context: str) -> str:
        """Create prompt for LLM reasoning."""
        return f"""Analyze the following conversation segments and provide refined emotion labels, tone descriptions, and reaction classifications.

For each segment, provide:
1. Refined Emotion: One of [Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral, Excitement, Contemplation]
2. Tone: Brief description (e.g., "Enthusiastic", "Hesitant", "Confident", "Uncertain")
3. Reaction: Classification if this is a response to a question or shows engagement (e.g., "Positive engagement", "Neutral response", "Question", "Agreement", "Disagreement", "No clear reaction")

Return your analysis as a JSON array, one object per segment with keys: "emotion", "tone", "reaction".

Segments:
{context}

JSON Response:"""
    
    def _parse_llm_response(self, response_text: str, original_segments: List[Dict]) -> List[Dict]:
        """Parse LLM JSON response."""
        try:
            # Extract JSON from response (might have markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Ensure it's a list
            if isinstance(parsed, dict):
                parsed = [parsed]
            
            return parsed
        
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return []
    
    def _merge_refinements(
        self,
        original_segments: List[Dict],
        refined_data: List[Dict]
    ) -> List[Dict]:
        """Merge LLM refinements with original segments."""
        merged = []
        
        for i, seg in enumerate(original_segments):
            refined = refined_data[i] if i < len(refined_data) else {}
            
            merged_seg = seg.copy()
            merged_seg["emotion"] = refined.get("emotion", "Neutral")
            merged_seg["tone"] = refined.get("tone", "Neutral")
            merged_seg["reaction"] = refined.get("reaction", "No clear reaction")
            
            merged.append(merged_seg)
        
        return merged
    
    def _fallback_refinements(self, segments: List[Dict]) -> List[Dict]:
        """Fallback when LLM fails."""
        fallback = []
        
        for seg in segments:
            fallback_seg = seg.copy()
            
            # Use primary emotion from fusion
            audio_emotion = seg.get("audio_emotion", {})
            visual_emotion = seg.get("visual_emotion", {})
            
            emotion = (
                visual_emotion.get("emotion") or
                audio_emotion.get("emotion") or
                "Neutral"
            )
            
            fallback_seg["emotion"] = emotion
            fallback_seg["tone"] = "Neutral"
            fallback_seg["reaction"] = "No clear reaction"
            
            fallback.append(fallback_seg)
        
        return fallback
    
    def generate_summary(self, segments: List[Dict]) -> Dict:
        """
        Generate conversation summary using LLM.
        
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
        
        # Create summary text
        summary_text = f"Conversation with {len(speakers)} speakers. "
        summary_text += f"Dominant emotions: {', '.join(set(emotions[:5]))}."
        
        # Try LLM summary (optional, can skip if needed)
        try:
            prompt = f"Summarize this conversation in 2-3 sentences:\n\n{summary_text}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a conversation summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            summary_text = response.choices[0].message.content
        except:
            pass  # Use basic summary
        
        return {
            "total_speakers": len(speakers),
            "total_segments": len(segments),
            "dominant_emotions": list(set(emotions))[:5],
            "summary": summary_text
        }


def refine_with_llm(segments: List[Dict], api_key: Optional[str] = None) -> List[Dict]:
    """
    Convenience function to refine segments with LLM.
    
    Args:
        segments: Aligned segments
        api_key: Optional Groq API key
    
    Returns:
        Refined segments
    """
    reasoner = GroqReasoner(api_key=api_key)
    return reasoner.refine_emotions_and_reactions(segments)
