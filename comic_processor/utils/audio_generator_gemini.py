"""Audio generation utilities using Gemini Native Audio.

Gemini 2.5 Flash includes native audio generation capability that can generate
natural-sounding narration directly from text. This implementation processes entire
panel narrations at once for a more cohesive narrative flow.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import wave
from google import genai
from google.genai import types


class GeminiTTSAudioGenerator:
    """
    Handles text-to-speech audio generation with Gemini Native Audio.
    
    Features:
    - Processes entire panel narrations at once for cohesive flow
    - Natural-sounding AI voice narration using gemini-2.5-flash-native-audio-dialog
    - Simpler setup than IndexTTS (no local models needed)
    - No separate Google Cloud credentials required
    - Cloud-based processing
    """
    
    def __init__(self, voice_name: str = "Enceladus", speaking_rate: float = 1.0):
        """
        Initialize Gemini Native Audio engine.
        
        Args:
            voice_name: Voice to use for narration. Available voices:
                - "Enceladus" (default): High-quality, natural narration
                - "Puck": Neutral, clear narration voice
                - "Charon": Deep, authoritative voice
                - "Kore": Warm, friendly voice
                - "Fenrir": Dynamic, energetic voice
                - "Aoede": Smooth, professional voice
            speaking_rate: Speed of speech (0.25 to 4.0, default 1.0)
        """
        print("🎤 Initializing Gemini Native Audio (gemini-2.5-flash-preview-tts)...")
        
        # Load API keys (support both single and multiple keys)
        self.api_keys = []
        
        # Try comma-separated list first
        keys_list = os.getenv('GEMINI_API_KEYS', '')
        if keys_list:
            self.api_keys = [k.strip() for k in keys_list.split(',') if k.strip()]
        
        # Fallback to single key
        if not self.api_keys:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                self.api_keys = [api_key]
        
        if not self.api_keys:
            raise ValueError(
                "No GEMINI_API_KEY found! "
                "Gemini Native Audio requires an API key. Add to your .env file."
            )
        
        # Initialize with first key
        self.current_key_index = 0
        self.client = genai.Client(api_key=self.api_keys[0])
        
        self.voice_name = voice_name
        self.speaking_rate = speaking_rate
        
        print(f"   🔑 Loaded {len(self.api_keys)} API key(s)")
        print(f"   🎙️  Voice: {voice_name}")
        print(f"   ⚡ Speaking rate: {speaking_rate}x")
        print(f"   ✅ Gemini Native Audio initialized")
    
    def generate_audio(
        self,
        narrative_segments: List[Dict],
        audio_dir: Path,
        **kwargs
    ) -> List[Dict]:
        """
        Generate TTS audio for narrative segments using Gemini Native Audio.
        
        Unlike IndexTTS which processes sentence by sentence, Gemini Native Audio
        processes the entire narration at once for better flow.
        
        Args:
            narrative_segments: List of dicts with 'segment_id', 'text', 'show_panels'
            audio_dir: Directory to save audio files
            **kwargs: Additional parameters (for compatibility with IndexTTS interface)
            
        Returns:
            List of audio segment dicts with paths and durations
        """
        print(f"🎙️  Generating narrative audio with Gemini Native Audio...")
        
        # Combine all narrative segments into one coherent text
        full_narrative = self._combine_narrative_segments(narrative_segments)
        
        if not full_narrative.strip():
            print("   ⚠️  No narrative text to generate")
            return []
        
        print(f"   📝 Combined narration: {len(full_narrative)} characters")
        print(f"   🎬 Generating single audio file...")
        
        # Generate single audio file for entire narration
        audio_path = audio_dir / "full_narration.wav"
        
        try:
            success = self._generate_tts(full_narrative, audio_path)
            
            if not success:
                print("   ❌ Failed to generate audio")
                return []
            
            # Get duration
            duration = self._get_audio_duration(audio_path)
            print(f"   ✅ Generated audio: {duration:.1f}s")
            
            # Split into multiple segments to cycle through panels
            # Even though we have one audio file, create multiple segments with time slices
            segments_out = []
            num_segments = len(narrative_segments)
            
            if num_segments == 0:
                num_segments = 1
            
            segment_duration = duration / num_segments
            
            for i, orig_segment in enumerate(narrative_segments):
                segments_out.append({
                    'chunk_id': 1,
                    'segment_id': i + 1,
                    'sentence_index': i,
                    'text': orig_segment.get('text', ''),
                    'audio_path': audio_path,  # Same audio file for all
                    'duration': segment_duration,
                    'show_panels': orig_segment.get('show_panels', []),
                    'is_first': i == 0,
                    'is_last': i == num_segments - 1
                })
            
            print(f"   📋 Split into {num_segments} segments for panel cycling")
            return segments_out
            
        except Exception as e:
            print(f"   ❌ Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _combine_narrative_segments(self, narrative_segments: List[Dict]) -> str:
        """
        Combine all narrative segments into one coherent text.
        
        Args:
            narrative_segments: List of segment dicts
            
        Returns:
            Combined narrative text
        """
        narrative_parts = []
        
        for segment in narrative_segments:
            text = segment.get('text', '').strip()
            if text:
                narrative_parts.append(text)
        
        # Join with proper spacing and punctuation
        full_text = ' '.join(narrative_parts)
        
        # Clean up any double spaces or punctuation issues
        full_text = full_text.replace('  ', ' ')
        full_text = full_text.replace(' .', '.')
        full_text = full_text.replace(' ,', ',')
        full_text = full_text.replace(' !', '!')
        full_text = full_text.replace(' ?', '?')
        
        return full_text
    
    def _get_all_panels(self, narrative_segments: List[Dict]) -> List[int]:
        """
        Get all unique panel numbers from segments.
        
        Args:
            narrative_segments: List of segment dicts
            
        Returns:
            Sorted list of unique panel numbers
        """
        all_panels = set()
        
        for segment in narrative_segments:
            panels = segment.get('show_panels', [])
            all_panels.update(panels)
        
        return sorted(all_panels)
    
    def _rotate_api_key(self) -> bool:
        """
        Rotate to the next API key.
        
        Returns:
            True if there's another key to try, False if all keys exhausted
        """
        if len(self.api_keys) <= 1:
            return False
        
        # Try next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        
        # Don't rotate back to the first key (we've tried them all)
        if self.current_key_index == 0:
            return False
        
        # Initialize client with new key
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        print(f"   🔄 Rotating to API key #{self.current_key_index + 1}")
        return True
    
    def _generate_tts(self, text: str, output_path: Path) -> bool:
        """
        Generate TTS audio using Gemini Native Audio API.
        
        Uses gemini-2.5-flash-preview-tts model to generate
        natural-sounding speech directly from text.
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        # Try with current key, rotate on quota error
        max_retries = len(self.api_keys)
        
        for attempt in range(max_retries):
            try:
                # Generate audio using Gemini TTS
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=self.voice_name,
                                )
                            )
                        )
                    )
                )
                
                # Extract audio data from response
                if response.candidates and len(response.candidates) > 0:
                    audio_data = response.candidates[0].content.parts[0].inline_data.data
                    
                    # Save as WAV file
                    with wave.open(str(output_path), "wb") as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(24000)  # 24kHz
                        wf.writeframes(audio_data)
                    
                    print(f"   ✅ Audio generated successfully")
                    return True
                
                print(f"   ❌ No audio data in response")
                return False
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a quota error
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                    print(f"   ⚠️  API key #{self.current_key_index + 1} quota exhausted")
                    
                    # Try to rotate to next key
                    if not self._rotate_api_key():
                        print(f"   ❌ All API keys exhausted")
                        print(f"   ❌ TTS generation failed: {e}")
                        return False
                    
                    # Continue to next attempt with new key
                    continue
                else:
                    # Not a quota error, fail immediately
                    print(f"   ❌ TTS generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
        
        print(f"   ❌ Failed after trying all {max_retries} API keys")
        return False
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration in seconds using ffprobe.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            # Try ffprobe first (most reliable)
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            # Fallback to afinfo on macOS
            try:
                cmd = ["afinfo", str(audio_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                for line in result.stdout.split('\n'):
                    if 'estimated duration:' in line.lower():
                        duration_str = line.split(':')[-1].strip().split()[0]
                        return float(duration_str)
            except:
                pass
        
        # Default fallback
        return 5.0  # Conservative estimate
