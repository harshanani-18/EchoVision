"""
Transcription Analyzer Module with Gemini 2.5 Flash (Paid Tier)

Uses Google's latest Gemini 2.5 Flash AI model for intelligent classification
of transcribed text into three categories:
1. FILLER - Filler words, hesitations, and discourse markers
2. ADMINISTRATION - Meta-statements, logistics, timestamps, speaker intro
3. VISUAL_CONCEPT - Core educational content meant for visualization
"""

import re
import os
import json
import asyncio
import time
import base64
import io
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed.")

try:
    from google import genai as genai_new
    GENAI_IMAGE_AVAILABLE = True
except ImportError:
    GENAI_IMAGE_AVAILABLE = False
    print("Warning: google-genai not installed. Image generation unavailable.")


class ContentType(Enum):
    """Classification categories for transcribed content"""
    FILLER = "filler"
    ADMINISTRATION = "administration"
    VISUAL_CONCEPT = "visual_concept"


@dataclass
class SegmentedContent:
    """Container for segregated transcription content"""
    filler: List[str]
    administration: List[str]
    visual_concept: List[str]
    
    def to_dict(self):
        return {
            "filler": self.filler,
            "administration": self.administration,
            "visual_concept": self.visual_concept
        }


class TranscriptionAnalyzer:
    """
    Analyzes and segregates transcribed text using Google's latest Gemini 2.5 Flash AI model (Paid Tier)
    
    This analyzer uses advanced language understanding to classify content intelligently,
    providing better accuracy than pattern matching alone.
    """
    
    def __init__(self, use_gemini: bool = True, buffer_size: int = 5, buffer_timeout: float = 3.0):
        """
        Initialize the analyzer
        
        Args:
            use_gemini: Whether to use Gemini (requires API key)
            buffer_size: Number of segments to accumulate before classification
            buffer_timeout: Seconds to wait before classifying partial buffer
        """
        load_dotenv()
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        
        # Buffering configuration
        self.buffer_size = buffer_size  # Accumulate 5 segments for context
        self.buffer_timeout = buffer_timeout  # Classify after 3 seconds if buffer not full
        self.segment_buffer = []  # Current buffer of accumulated segments
        
        # Rate limiting (Gemini paid tier: 2000 RPM, no strict daily limit)
        self.max_rpm = 1000  # Paid tier allows up to 2000 RPM; using half for safety
        self.max_rpd = 100000  # Paid tier has no strict daily limit
        self._request_timestamps = []  # Track per-minute requests
        self._daily_request_count = 0
        self._daily_reset_time = time.time()  # Resets every 24h
        
        if self.use_gemini:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Warning: GOOGLE_API_KEY not set in .env. Falling back to pattern matching.")
                self.use_gemini = False
            else:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Image generation client (new SDK)
        self.image_client = None
        if GENAI_IMAGE_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.image_client = genai_new.Client(api_key=api_key)
        
        # Visual concept accumulation for image generation
        self.visual_concepts = []  # Accumulated visual concept texts
        self.min_concepts_for_image = 3  # Min concepts before generating image
        
        # Fallback patterns for when Gemini is unavailable
        self.FILLER_WORDS = {
            'um', 'uh', 'uhh', 'umm', 'errs', 'err',
            'like', 'you know', 'i mean', 'kind of', 'sort of',
            'actually', 'basically', 'essentially', 'literally', 'obviously',
            'right', 'okay', 'so', 'well', 'anyway',
            'uh-huh', 'hmm', 'ah', 'yeah'
        }
        
        self.ADMIN_PATTERNS = [
            r'\bone\b.*\btwo\b.*\bthree\b',
            r'check',
            r'hello+',
            r'testing|test\b',
            r'can you hear|audible|mic',
            r'ladies and gentlemen',
            r'welcome|thank you for|introduction',
            r'\btimestamp\b|\btime\b.*\bmark',
            r'(?:speaker|presenter).*(?:says|here)',
            r'raise your hand|everyone|all',
        ]
        
        self.compiled_admin_patterns = [re.compile(p, re.IGNORECASE) for p in self.ADMIN_PATTERNS]
    
    async def _wait_for_rate_limit(self):
        """
        Enforce rate limits before making a Gemini API call.
        Waits if per-minute limit is close, raises error if daily limit exhausted.
        """
        now = time.time()
        
        # Reset daily counter every 24 hours
        if now - self._daily_reset_time >= 86400:
            self._daily_request_count = 0
            self._daily_reset_time = now
        
        # Check daily limit
        if self._daily_request_count >= self.max_rpd:
            raise RuntimeError(
                f"Daily Gemini API limit reached ({self.max_rpd} requests). "
                f"Resets in {int(86400 - (now - self._daily_reset_time))}s. "
                f"Consider upgrading your API plan."
            )
        
        # Clean up timestamps older than 60 seconds
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60]
        
        # If at per-minute limit, wait until oldest request expires
        if len(self._request_timestamps) >= self.max_rpm:
            wait_time = 60 - (now - self._request_timestamps[0]) + 0.5
            if wait_time > 0:
                print(f"Rate limit: waiting {wait_time:.1f}s before next Gemini call")
                await asyncio.sleep(wait_time)
                self._request_timestamps = [t for t in self._request_timestamps if time.time() - t < 60]
        
        # Record this request
        self._request_timestamps.append(time.time())
        self._daily_request_count += 1
    
    def get_rate_limit_status(self) -> dict:
        """Get current rate limit usage"""
        now = time.time()
        recent = len([t for t in self._request_timestamps if now - t < 60])
        return {
            "requests_this_minute": recent,
            "max_rpm": self.max_rpm,
            "requests_today": self._daily_request_count,
            "max_rpd": self.max_rpd,
            "daily_remaining": self.max_rpd - self._daily_request_count
        }
    
    async def classify_batch_async(self, segments: List[str], max_retries: int = 3) -> List[tuple]:
        """
        Classify multiple segments together for better context.
        Retries with exponential backoff on rate limit errors.
        
        Args:
            segments: List of text segments to classify together
            max_retries: Number of retry attempts on failure
            
        Returns:
            List of tuples: (segment_text, ContentType)
        """
        if not self.use_gemini:
            raise RuntimeError("Gemini API is required but not available. Set GOOGLE_API_KEY in .env")
        
        if not segments:
            return []
        
        # Create batch prompt with all segments
        segments_text = "\n".join([f"{i+1}. {seg}" for i, seg in enumerate(segments)])
        
        prompt = f"""Classify each of the following speech segments into ONE category per segment.
Categories:
- FILLER: Hesitations (um, uh), discourse markers (like, you know, basically)
- ADMINISTRATION: Microphone tests, greetings, meta-statements (e.g., "one two three check")
- VISUAL_CONCEPT: Educational content meant for visualization

Speech segments:
{segments_text}

Respond with ONLY a numbered list with categories, one per line, like:
1. VISUAL_CONCEPT
2. FILLER
3. ADMINISTRATION
(No explanations)"""

        last_error = None
        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                response = await self.model.generate_content_async(prompt)
                lines = response.text.strip().split('\n')
                
                results = []
                valid_types = {'FILLER', 'ADMINISTRATION', 'VISUAL_CONCEPT'}
                
                for i, line in enumerate(lines):
                    if i >= len(segments):
                        break
                    
                    # Extract category (handle "1. CATEGORY" format)
                    category = line.split('.')[-1].strip().upper()
                    
                    if category not in valid_types:
                        category = 'VISUAL_CONCEPT'  # Default
                    
                    results.append((segments[i], ContentType[category]))
                
                # Pad with defaults if needed
                while len(results) < len(segments):
                    idx = len(results)
                    results.append((segments[idx], ContentType.VISUAL_CONCEPT))
                
                return results
            
            except Exception as e:
                last_error = e
                wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"Gemini attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        
        raise RuntimeError(f"Gemini classification failed after {max_retries} attempts: {last_error}")
    
    async def add_to_buffer(self, segment: str) -> Optional[List[tuple]]:
        """
        Add a segment to the accumulation buffer.
        Returns classifications if buffer is full, None otherwise.
        
        Args:
            segment: Text segment to accumulate
            
        Returns:
            List of (segment, classification) tuples if buffer full, None otherwise
        """
        if not segment.strip():
            return None
        
        self.segment_buffer.append(segment)
        
        # Return classifications when buffer reaches size
        if len(self.segment_buffer) >= self.buffer_size:
            return await self.get_buffer_classifications()
        
        return None
    
    async def get_buffer_classifications(self) -> Optional[List[tuple]]:
        """
        Get classifications for current buffer contents using Gemini.
        Clears the buffer after successful classification.
        On failure, segments are put back into the buffer for retry.
        
        Returns:
            List of (segment, classification) tuples, None if buffer empty or classification failed
        """
        if not self.segment_buffer:
            return None
        
        segments_to_classify = self.segment_buffer.copy()
        self.segment_buffer.clear()
        
        try:
            classifications = await self.classify_batch_async(segments_to_classify)
            return classifications
        except Exception as e:
            print(f"Gemini classification failed, keeping segments for retry: {e}")
            # Put segments back so they aren't lost
            self.segment_buffer = segments_to_classify + self.segment_buffer
            return None
    
    def get_buffer_status(self) -> dict:
        """
        Get current buffer status
        
        Returns:
            Dict with buffer info
        """
        return {
            "buffered_segments": len(self.segment_buffer),
            "buffer_size": self.buffer_size,
            "is_full": len(self.segment_buffer) >= self.buffer_size,
            "segments": self.segment_buffer.copy()
        }
    
    async def segment_sentence_async(self, text: str) -> ContentType:
        """
        Classify a single sentence using Gemini 2.5 Flash
        
        Args:
            text: Text to classify
            
        Returns:
            ContentType classification
            
        Raises:
            RuntimeError: If Gemini API is not available
        """
        if not self.use_gemini:
            raise RuntimeError("Gemini API is required but not available. Set GOOGLE_API_KEY in .env")
        
        prompt = f"""Classify the following speech segment into ONE category:
- FILLER: Hesitations (um, uh), discourse markers (like, you know, basically)
- ADMINISTRATION: Microphone tests, greetings, meta-statements (e.g., "one two three check", "hello everyone")
- VISUAL_CONCEPT: Educational content meant for visualization

Speech segment: "{text}"

Respond with ONLY the category name (FILLER, ADMINISTRATION, or VISUAL_CONCEPT). No explanation."""

        await self._wait_for_rate_limit()
        response = await self.model.generate_content_async(prompt)
        classification = response.text.strip().upper()
        
        # Validate response
        valid_types = {'FILLER', 'ADMINISTRATION', 'VISUAL_CONCEPT'}
        if classification not in valid_types:
            classification = 'VISUAL_CONCEPT'  # Default to visual concept
        
        return ContentType[classification]
    
    def segment_sentence(self, text: str) -> ContentType:
        """
        Classify a single sentence/phrase using pattern matching (fallback method)
        
        Args:
            text: Text to classify
            
        Returns:
            ContentType classification
        """
        sentence = text.strip()
        if not sentence:
            return None
        
        text_lower = sentence.lower()
        
        # Check for filler
        for word in self.FILLER_WORDS:
            if word in text_lower:
                return ContentType.FILLER
        
        if re.match(r'^[a-z]{1,3}$', text_lower) and text_lower in ['um', 'uh', 'ah', 'eh', 'hm', 'mm']:
            return ContentType.FILLER
        
        # Check for administration
        if len(text_lower) >= 3:
            for pattern in self.compiled_admin_patterns:
                if pattern.search(text_lower):
                    return ContentType.ADMINISTRATION
        
        # Default to visual concept if meaningful length
        return ContentType.VISUAL_CONCEPT if len(text_lower) > 10 else ContentType.FILLER
    
    async def segment_text(self, text: str, split_by: str = "sentence") -> SegmentedContent:
        """
        Segregate full transcription text into three categories using Gemini.
        
        Args:
            text: Full transcribed text
            split_by: "sentence" (uses .) or "phrase" (uses comma)
        
        Returns:
            SegmentedContent with three lists
            
        Raises:
            RuntimeError: If Gemini API is not available or fails
        """
        if not self.use_gemini:
            raise RuntimeError("Gemini API is required but GOOGLE_API_KEY is not set in .env")
        
        # Split text into segments
        if split_by == "sentence":
            segments = [s.strip() for s in text.split('.') if s.strip()]
        elif split_by == "phrase":
            segments = re.split(r'[.,;!?]+', text)
            segments = [s.strip() for s in segments if s.strip()]
        else:
            segments = [text]
        
        filler_list = []
        admin_list = []
        concept_list = []
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            content_type = await self.segment_sentence_async(segment)
            
            if content_type == ContentType.FILLER:
                filler_list.append(segment)
            elif content_type == ContentType.ADMINISTRATION:
                admin_list.append(segment)
            elif content_type == ContentType.VISUAL_CONCEPT:
                concept_list.append(segment)
        
        return SegmentedContent(
            filler=filler_list,
            administration=admin_list,
            visual_concept=concept_list
        )
    
    async def analyze_file(self, filepath: str) -> SegmentedContent:
        """Read transcription file and analyze"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return await self.segment_text(text)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return SegmentedContent([], [], [])
    
    def print_analysis(self, segmented: SegmentedContent):
        """Pretty print the analysis results"""
        print("\n" + "="*80)
        print("TRANSCRIPTION ANALYSIS REPORT (Powered by Gemini 2.5 Flash)" if self.use_gemini else "TRANSCRIPTION ANALYSIS REPORT (Pattern Matching Fallback)")
        print("="*80)
        
        print(f"\n📝 FILLER WORDS & HESITATIONS ({len(segmented.filler)})")
        print("-" * 80)
        for i, item in enumerate(segmented.filler[:10], 1):
            print(f"  {i}. {item}")
        if len(segmented.filler) > 10:
            print(f"  ... and {len(segmented.filler) - 10} more")
        
        print(f"\n⚙️  ADMINISTRATIVE CONTENT ({len(segmented.administration)})")
        print("-" * 80)
        for i, item in enumerate(segmented.administration[:10], 1):
            print(f"  {i}. {item}")
        if len(segmented.administration) > 10:
            print(f"  ... and {len(segmented.administration) - 10} more")
        
        print(f"\n💡 VISUAL CONCEPTS ({len(segmented.visual_concept)})")
        print("-" * 80)
        for i, item in enumerate(segmented.visual_concept[:15], 1):
            print(f"  {i}. {item}")
        if len(segmented.visual_concept) > 15:
            print(f"  ... and {len(segmented.visual_concept) - 15} more")
        
        print("\n" + "="*80)
        print(f"SUMMARY: {len(segmented.filler)} filler | {len(segmented.administration)} admin | {len(segmented.visual_concept)} concepts")
        print("="*80 + "\n")

    def add_visual_concept(self, concept_text: str):
        """Add a visual concept to the accumulation buffer"""
        self.visual_concepts.append(concept_text)
    
    def get_visual_concepts_status(self) -> dict:
        """Get current visual concepts buffer status"""
        return {
            "count": len(self.visual_concepts),
            "min_required": self.min_concepts_for_image,
            "ready": len(self.visual_concepts) >= self.min_concepts_for_image,
            "concepts": list(self.visual_concepts)
        }
    
    def should_generate_image(self) -> bool:
        """Check if enough visual concepts have accumulated for image generation"""
        return len(self.visual_concepts) >= self.min_concepts_for_image
    
    async def generate_image_from_concepts(self) -> Optional[dict]:
        """
        Generate an educational image from accumulated visual concepts using Gemini.
        
        Returns:
            Dict with 'image_base64' (str), 'prompt' (str), 'concepts_used' (list)
            or None if generation fails
        """
        if not self.image_client:
            raise RuntimeError("Image generation client not available. Install google-genai and set GOOGLE_API_KEY.")
        
        if not self.visual_concepts:
            return None
        
        # Take current concepts and clear buffer
        concepts = self.visual_concepts.copy()
        self.visual_concepts.clear()
        
        # Build a prompt from accumulated concepts
        concepts_text = "\n".join(f"- {c}" for c in concepts)
        prompt = f"""Create a clear, educational diagram or illustration for a classroom setting.
The image should visually explain the following concepts that are being taught:

{concepts_text}

Make it a clean, labeled scientific diagram with clear annotations.
Use a white background, bright colors, and clear labels.
Style: educational textbook illustration."""
        
        await self._wait_for_rate_limit()
        
        try:
            # Run the sync image generation call in a thread to avoid blocking the event loop
            def _sync_generate():
                return self.image_client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt],
                )
            
            response = await asyncio.to_thread(_sync_generate)
            
            # Extract image from response
            for part in response.parts:
                if part.inline_data is not None:
                    # Convert to base64
                    image_bytes = part.inline_data.data
                    if isinstance(image_bytes, str):
                        image_base64 = image_bytes
                    else:
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    mime_type = part.inline_data.mime_type or 'image/png'
                    
                    return {
                        "image_base64": image_base64,
                        "mime_type": mime_type,
                        "prompt": prompt,
                        "concepts_used": concepts
                    }
            
            # No image in response — put concepts back
            print("Gemini response had no image data")
            self.visual_concepts = concepts + self.visual_concepts
            return None
            
        except Exception as e:
            print(f"Image generation failed: {e}")
            # Put concepts back so they aren't lost
            self.visual_concepts = concepts + self.visual_concepts
            raise RuntimeError(f"Image generation failed: {e}")


# Utility function for quick analysis
def analyze_transcription(text: str, use_gemini: bool = True) -> SegmentedContent:
    """Quick function to analyze transcription text"""
    analyzer = TranscriptionAnalyzer(use_gemini=use_gemini)
    return analyzer.segment_text(text)


if __name__ == "__main__":
    # Example usage
    analyzer = TranscriptionAnalyzer(use_gemini=True)
    
    print(f"Using Gemini: {analyzer.use_gemini}")
    
    # Analyze the transcriptions.txt file
    print("Analyzing transcriptions.txt...")
    results = asyncio.run(analyzer.analyze_file("transcriptions.txt"))
    analyzer.print_analysis(results)
    
    # Save results to JSON
    output = {
        "model": "Gemini 2.5 Flash" if analyzer.use_gemini else "Pattern Matching",
        "summary": {
            "total_filler": len(results.filler),
            "total_administration": len(results.administration),
            "total_concepts": len(results.visual_concept)
        },
        "categories": results.to_dict()
    }
    
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to analysis_results.json")

