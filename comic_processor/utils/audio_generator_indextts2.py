"""Audio generation utilities using IndexTTS2.

IndexTTS2 is a breakthrough autoregressive zero-shot TTS system offering:
- Precise speech duration control
- Emotionally expressive generation
- Disentanglement of emotional expression and speaker identity
- Zero-shot voice cloning from reference audio
- Multi-language support (Chinese, English, and more)
- Text-based emotion control
- Superior naturalness and speaker similarity
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch


class IndexTTS2AudioGenerator:
    """
    Handles text-to-speech audio generation with IndexTTS2.
    
    Features:
    - Zero-shot voice cloning from reference audio
    - Emotional expression control
    - Precise duration control
    - Multi-language support
    - Parallel audio generation for speed
    - Automatic sentence splitting for optimal quality
    """
    
    # Configuration constants
    MAX_SENTENCE_LENGTH = 200  # Characters per segment for optimal quality
    USE_FP16 = True  # Use half-precision for faster inference (lower VRAM)
    USE_DEEPSPEED = False  # DeepSpeed may help on some systems
    USE_CUDA_KERNEL = False  # Compiled CUDA kernels (may be faster)
    USE_TORCH_COMPILE = False  # PyTorch 2.0 compile (experimental)
    MAX_TEXT_TOKENS = 120  # Max tokens per segment
    
    # Emotion control parameters (configured for consistent narration)
    DEFAULT_EMO_ALPHA = 0.3  # Low emotion for consistent tone (0.0-1.0)
    DEFAULT_EMOTION_MODE = "neutral"  # Use neutral mode for consistent narration
    
    def __init__(self, speaker_wav: str = None, model_dir: str = "index-tts/checkpoints"):
        """ p
        Initialize IndexTTS2 engine.
        
        Args:
            speaker_wav: Path to reference voice WAV file for voice cloning.
                        This is the voice that will be cloned for narration.
            model_dir: Directory containing IndexTTS2 model files.
                      Default: "checkpoints" (will be downloaded automatically)
        """
        print("🎤 Initializing IndexTTS2 (State-of-the-Art Zero-Shot TTS)...")
        print("   📦 Loading models from:", model_dir)
        
        # Set cache directory for Hugging Face models
        os.environ['HF_HUB_CACHE'] = f'{model_dir}/hf_cache'
        
        # Check for CUDA/MPS availability
        self._check_device()
        
        # Import IndexTTS2
        try:
            from indextts.infer_v2 import IndexTTS2
        except ImportError:
            raise ImportError(
                "IndexTTS2 not installed! Please install from: "
                "https://github.com/index-tts/index-tts\n"
                "See INDEXTTS_SETUP.md for detailed instructions."
            )
        
        # Initialize IndexTTS2 model
        config_path = os.path.join(model_dir, "config.yaml")
        
        print(f"   ⚙️  Configuration:")
        print(f"      - FP16: {self.USE_FP16} (faster, lower VRAM)")
        print(f"      - DeepSpeed: {self.USE_DEEPSPEED}")
        print(f"      - CUDA Kernel: {self.USE_CUDA_KERNEL}")
        print(f"      - Torch Compile: {self.USE_TORCH_COMPILE}")
        
        # Patch IndexTTS2 to handle missing QwenEmotion model gracefully
        # This allows the system to work without the QwenEmotion model for neutral narration
        import indextts.infer_v2 as infer_module
        
        # Store original QwenEmotion class
        original_qwen = infer_module.QwenEmotion
        
        # Create a mock QwenEmotion that works without the model files
        class MockQwenEmotion:
            def __init__(self, model_dir):
                print("   ℹ️  QwenEmotion model not loaded (not needed for neutral narration)")
                
            def inference(self, text):
                # Return neutral emotion vector (calm voice)
                return {
                    "高兴": 0.0, "愤怒": 0.0, "悲伤": 0.0, "恐惧": 0.0,
                    "反感": 0.0, "低落": 0.0, "惊讶": 0.0, "自然": 1.0
                }
        
        # Replace QwenEmotion with mock version
        infer_module.QwenEmotion = MockQwenEmotion
        
        try:
            self.tts = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_dir,
                use_fp16=self.USE_FP16,
                use_deepspeed=self.USE_DEEPSPEED,
                use_cuda_kernel=self.USE_CUDA_KERNEL,
                use_torch_compile=self.USE_TORCH_COMPILE
            )
        finally:
            # Restore original QwenEmotion class for other uses
            infer_module.QwenEmotion = original_qwen
        
        self._tts_lock = threading.Lock()  # Thread-safe TTS access
        
        # Store reference voice for cloning
        self.reference_voice = speaker_wav
        
        if speaker_wav:
            if not os.path.exists(speaker_wav):
                print(f"   ⚠️  Warning: Reference voice not found: {speaker_wav}")
                print(f"      Will use default IndexTTS2 voice")
                self.reference_voice = None
            else:
                print(f"   🎙️  Voice cloning enabled: {speaker_wav}")
        else:
            print("   🎙️  Using default IndexTTS2 voice (no cloning)")
        
        print(f"   ✅ IndexTTS2 initialized successfully")
        print(f"   📏 Max sentence length: {self.MAX_SENTENCE_LENGTH} chars")
        print(f"   🎭 Emotion mode: {self.DEFAULT_EMOTION_MODE} (consistent tone)")
        print(f"   🎚️  Emotion strength: {self.DEFAULT_EMO_ALPHA:.1f} (low for stability)")
    
    def _check_device(self):
        """Check and report available compute devices."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"   🚀 CUDA GPU detected: {device_name}")
            print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   🍎 Apple Metal (MPS) detected")
        else:
            print(f"   💻 Running on CPU (slower)")
            print(f"      Consider using GPU for faster inference")
    
    def generate_audio(
        self,
        narrative_segments: List[Dict],
        audio_dir: Path,
        emotion_mode: str = None,
        emotion_intensity: Optional[float] = None
    ) -> List[Dict]:
        """
        Generate TTS audio for narrative segments using IndexTTS2.
        
        Args:
            narrative_segments: List of dicts with 'segment_id', 'text', 'show_panels'
            audio_dir: Directory to save audio files
            emotion_mode: Emotion control mode:
                - "neutral": Calm, consistent narration (default - recommended)
                - "auto": Automatic emotion from text (may vary tone)
                - "reference": Use emotion from reference voice
                If None, uses DEFAULT_EMOTION_MODE (neutral)
            emotion_intensity: Override emotion strength (0.0-1.0)
                              If None, uses DEFAULT_EMO_ALPHA (0.3 for consistency)
            
        Returns:
            List of audio segment dicts with paths and durations
        """
        print(f"🎙️  Generating narrative audio with IndexTTS2...")
        
        # Use default emotion mode if not specified
        if emotion_mode is None:
            emotion_mode = self.DEFAULT_EMOTION_MODE
        
        print(f"   🎭 Emotion mode: {emotion_mode}")
        
        # Process segments: split long text into sentences
        processed_segments = self._prepare_segments_with_splitting(narrative_segments)
        print(f"   📝 Split into {len(processed_segments)} audio chunks (from {len(narrative_segments)} segments)")
        
        # Set emotion parameters for consistent tone
        emo_alpha = emotion_intensity if emotion_intensity is not None else self.DEFAULT_EMO_ALPHA
        use_emo_text = (emotion_mode == "auto")  # False for neutral/reference modes
        
        # Parallel audio generation
        max_workers = min(2, len(processed_segments))  # Limit workers to conserve VRAM
        audio_segments = []
        
        print(f"   🔄 Generating with {max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_segment = {
                executor.submit(
                    self._generate_single_audio,
                    segment,
                    audio_dir,
                    use_emo_text,
                    emo_alpha
                ): segment
                for segment in processed_segments
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    result = future.result()
                    if result:
                        audio_segments.append(result)
                        print(f"     ✅ Chunk {result['chunk_id']}: {result['duration']:.1f}s")
                except Exception as e:
                    print(f"     ❌ Failed chunk {segment.get('chunk_id', '?')}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Sort by chunk_id to maintain order
        audio_segments.sort(key=lambda x: x['chunk_id'])
        
        # Validate audio quality
        audio_segments = self._validate_audio_quality(audio_segments)
        
        total_duration = sum(seg['duration'] for seg in audio_segments)
        print(f"   ✅ Generated {len(audio_segments)} audio chunks ({total_duration:.1f}s total)")
        
        return audio_segments
    
    def _prepare_segments_with_splitting(self, narrative_segments: List[Dict]) -> List[Dict]:
        """
        Split long narrative segments into shorter sentences for optimal TTS quality.
        
        IndexTTS2 performs better with shorter, natural sentences.
        
        Args:
            narrative_segments: Original segments from AI analysis
            
        Returns:
            List of processed segments with split sentences
        """
        processed = []
        chunk_counter = 1
        
        for segment in narrative_segments:
            text = segment.get('text', '').strip()
            if not text:
                continue
            
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            # Check if any sentence is too long
            final_sentences = []
            for sentence in sentences:
                if len(sentence) > self.MAX_SENTENCE_LENGTH:
                    # Further split by commas or semicolons
                    sub_parts = self._split_long_sentence(sentence)
                    final_sentences.extend(sub_parts)
                else:
                    final_sentences.append(sentence)
            
            # Create chunk for each sentence, preserving metadata
            for i, sentence in enumerate(final_sentences):
                if not sentence.strip():
                    continue
                
                processed.append({
                    'chunk_id': chunk_counter,
                    'segment_id': segment['segment_id'],
                    'sentence_index': i,
                    'total_sentences': len(final_sentences),
                    'text': sentence.strip(),
                    'show_panels': segment.get('show_panels', []),
                    'is_first': i == 0,
                    'is_last': i == len(final_sentences) - 1
                })
                chunk_counter += 1
        
        return processed
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using punctuation-aware rules.
        
        Handles:
        - Periods, question marks, exclamation points
        - Common abbreviations
        - Ellipsis
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Common abbreviations that shouldn't split sentences
        abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr', 'etc', 'vs', 'e.g', 'i.e']
        
        # Protect abbreviations temporarily
        protected = text
        replacements = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            protected = protected.replace(f"{abbr}.", placeholder)
            replacements[placeholder] = f"{abbr}."
        
        # Split on sentence-ending punctuation
        sentence_endings = r'([.!?]+)\s+'
        parts = re.split(sentence_endings, protected)
        
        # Reconstruct sentences
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text part
                current += part
            else:  # Punctuation part
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
        
        # Restore abbreviations
        restored = []
        for sentence in sentences:
            for placeholder, original in replacements.items():
                sentence = sentence.replace(placeholder, original)
            restored.append(sentence)
        
        return restored
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split overly long sentences by commas or semicolons.
        
        Args:
            sentence: Long sentence text
            
        Returns:
            List of shorter parts
        """
        # Try splitting by semicolons first
        if ';' in sentence:
            parts = [p.strip() + '.' for p in sentence.split(';') if p.strip()]
            return parts
        
        # Then try commas
        if ',' in sentence:
            parts = sentence.split(',')
            # Recombine into chunks under MAX_SENTENCE_LENGTH
            chunks = []
            current_chunk = ""
            
            for part in parts:
                test_chunk = (current_chunk + ', ' + part).strip(', ')
                if len(test_chunk) <= self.MAX_SENTENCE_LENGTH:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk + '.')
                    current_chunk = part.strip()
            
            if current_chunk:
                chunks.append(current_chunk + '.')
            
            return chunks if chunks else [sentence]
        
        # If no good split points, just split at word boundaries
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= self.MAX_SENTENCE_LENGTH:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk) + '.')
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk) + '.')
        
        return chunks if chunks else [sentence]
    
    def _generate_single_audio(
        self,
        segment: Dict,
        audio_dir: Path,
        use_emo_text: bool,
        emo_alpha: float
    ) -> Optional[Dict]:
        """
        Generate audio for a single segment using IndexTTS2 (thread-safe).
        
        Args:
            segment: Segment dict with 'chunk_id', 'text', 'show_panels', etc.
            audio_dir: Directory to save audio files
            use_emo_text: Whether to use automatic emotion from text
            emo_alpha: Emotion strength (0.0-1.0)
            
        Returns:
            Audio segment dict or None if failed
        """
        text = self._clean_text_for_tts(segment['text'])
        if not text:
            return None
        
        chunk_id = segment['chunk_id']
        audio_filename = f"chunk_{chunk_id:03d}.wav"
        audio_path = audio_dir / audio_filename
        
        try:
            # Thread-safe TTS generation
            with self._tts_lock:
                # IndexTTS2 inference with zero-shot voice cloning
                # spk_audio_prompt: reference voice for speaker identity
                # use_emo_text: automatic emotion extraction from text
                # emo_alpha: emotion intensity (0.0-1.0)
                # verbose: detailed logging
                
                output_path = self.tts.infer(
                    spk_audio_prompt=self.reference_voice,
                    text=text,
                    output_path=str(audio_path),
                    use_emo_text=use_emo_text,
                    emo_alpha=emo_alpha,
                    max_text_tokens_per_segment=self.MAX_TEXT_TOKENS,
                    verbose=False  # Reduce logging spam
                )
            
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            
            return {
                'chunk_id': chunk_id,
                'segment_id': segment['segment_id'],
                'sentence_index': segment.get('sentence_index', 0),
                'text': text,
                'audio_path': audio_path,
                'duration': duration,
                'show_panels': segment.get('show_panels', []),
                'is_first': segment.get('is_first', True),
                'is_last': segment.get('is_last', True)
            }
            
        except Exception as e:
            print(f"     ❌ Error generating chunk {chunk_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean and normalize text for better TTS pronunciation.
        
        IndexTTS2 supports:
        - Chinese and English mixed text
        - Punctuation for prosody control
        - Pinyin for Chinese pronunciation control
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # Normalize ellipsis
        text = text.replace("...", "…")
        text = text.replace("…", ", ")
        
        # Normalize dashes
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        
        # Remove multiple exclamation/question marks (keep just one)
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        
        # Remove special markdown characters
        text = text.replace("*", "")
        text = text.replace("_", "")
        text = text.replace("#", "")
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure sentence ends with punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()
    
    def _validate_audio_quality(self, audio_segments: List[Dict]) -> List[Dict]:
        """
        Validate generated audio quality and detect potential issues.
        
        Checks:
        - Duration reasonableness
        - Audio file exists and is readable
        - Basic sanity checks
        
        Args:
            audio_segments: List of generated audio segments
            
        Returns:
            Filtered list of valid audio segments
        """
        valid_segments = []
        
        for segment in audio_segments:
            audio_path = segment['audio_path']
            duration = segment['duration']
            text = segment['text']
            
            # Check file exists
            if not audio_path.exists():
                print(f"     ⚠️  Chunk {segment['chunk_id']}: Audio file not found")
                continue
            
            # Check duration reasonableness
            # Rough estimate: ~150 words per minute = 2.5 words/sec
            word_count = len(text.split())
            min_duration = word_count * 0.2  # Conservative minimum
            max_duration = word_count * 3.0  # Conservative maximum
            
            if duration < min_duration:
                print(f"     ⚠️  Chunk {segment['chunk_id']}: Audio too short ({duration:.1f}s for {word_count} words)")
                # Still include but warn
            elif duration > max_duration:
                print(f"     ⚠️  Chunk {segment['chunk_id']}: Audio too long ({duration:.1f}s for {word_count} words)")
                # Still include but warn
            
            valid_segments.append(segment)
        
        return valid_segments
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration in seconds.
        
        Uses ffprobe (most reliable cross-platform method).
        
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
        
        # Default fallback (rough estimate based on file size)
        # WAV files are ~176KB per second at 44.1kHz stereo
        try:
            file_size = audio_path.stat().st_size
            estimated_duration = file_size / 176000  # Rough estimate
            return max(1.0, estimated_duration)  # Minimum 1 second
        except:
            return 3.0  # Last resort default


# Alias for backward compatibility
AudioGenerator = IndexTTS2AudioGenerator
