#!/usr/bin/env python3
"""
Enhanced Comic Narrative Pipeline - Modular Version
Combines computer vision panel extraction with Gemini AI narrative generation
to create professional narrative video recaps of comic/manga pages.

Features:
- CV-based panel extraction
- AI-powered narrative generation with context history
- Editable narration JSON files
- Regeneration mode (skip analysis, use edited JSON)
- Modular architecture with separate audio/video utilities

Author: Comic Processor Project
License: MIT
"""

# Fix protobuf compatibility issue for IndexTTS2
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import json
import time
import tempfile
import shutil
import base64
from pathlib import Path
from typing import List, Dict, Optional
import cv2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from utils.panel_extractor import PanelExtractor
from utils.audio_generator_indextts2 import IndexTTS2AudioGenerator
from utils.audio_generator_gemini import GeminiTTSAudioGenerator
from utils.video_generator import VideoGenerator
from utils.voice_selector import VoiceSelector
from concurrent.futures import ThreadPoolExecutor, as_completed


class EnhancedComicPipeline:
    """
    Main pipeline for creating narrative videos from comic pages.
    
    Process:
    1. Extract panels using computer vision
    2. Analyze panels with AI (Gemini or Ollama Qwen3-VL)
    3. Save narration to editable JSON
    4. Generate narrative audio with TTS
    5. Compose video with correct panels at correct times
    6. Combine all page videos into one complete video
    """
    
    def __init__(self, speaker_wav: str = None, tts_model: str = None):
        """
        Initialize the comic processing pipeline.
        
        Args:
            speaker_wav: Optional path to voice reference WAV file for narration.
                        If None, will use auto-detected or prompt for selection.
            tts_model: TTS model to use ('indextts' or 'gemini').
                      If None, will prompt user to select.
        """
        load_dotenv()
        
        # Setup directories
        self.input_dir = Path("comic_pages")
        self.output_dir = Path("results")
        self.cache_dir = Path("cache")
        
        for directory in [self.output_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Detect and setup AI provider
        self.ai_provider = os.getenv('AI_PROVIDER', 'gemini').lower()
        if self.ai_provider == 'ollama':
            self._setup_ollama()
        else:
            self._setup_gemini()
        
        # Select TTS model
        if tts_model is None:
            tts_model = self._ask_tts_model()
        
        self.tts_model = tts_model.lower()
        
        # Setup TTS audio generator based on selection
        if self.tts_model == 'gemini':
            self.audio_generator = GeminiTTSAudioGenerator()
        else:  # indextts
            # Setup voice for narration with interactive selection
            if speaker_wav is None:
                voice_result = VoiceSelector.quick_select()
                # Handle tuple return (voice_path, preset) from quick_select
                speaker_wav = voice_result[0] if isinstance(voice_result, tuple) else voice_result
            
            self.audio_generator = IndexTTS2AudioGenerator(speaker_wav=speaker_wav)
        
        self.video_generator = VideoGenerator()
        self.panel_extractor = PanelExtractor()
        
        self.content_type = None
        self.narrative_history = []  # Store previous narrations for context
        self.max_history = 3  # Maximum previous narrations to keep
        print("✅ Pipeline initialized")
    
    def _setup_gemini(self):
        """Setup Gemini AI with API key rotation."""
        api_keys = []
        
        # Try comma-separated list first
        keys_list = os.getenv('GEMINI_API_KEYS', '')
        if keys_list:
            api_keys = [k.strip() for k in keys_list.split(',') if k.strip()]
        
        # Fallback to single key
        if not api_keys:
            single_key = os.getenv('GEMINI_API_KEY', '')
            if single_key:
                api_keys = [single_key]
        
        if not api_keys:
            raise ValueError("No GEMINI_API_KEY found! Create a .env file with your API key.")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        
        genai.configure(api_key=self.api_keys[0])
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        print(f"🤖 AI Provider: Gemini")
        print(f"🔑 Loaded {len(self.api_keys)} API key(s)")
    
    def _setup_ollama(self):
        """Setup Ollama with Qwen3-VL."""
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package not installed!\n"
                "Install with: pip install ollama"
            )
        
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2-vl:latest')
        
        try:
            # Test connection and list models
            client = ollama.Client(host=self.ollama_base_url)
            response = client.list()
            models_list = response.get('models', [])
            
            # Extract model names (handle both 'name' and 'model' keys)
            available_models = [m.get('name') or m.get('model', '') for m in models_list]
            
            print(f"🤖 AI Provider: Ollama (Local)")
            print(f"🌐 Base URL: {self.ollama_base_url}")
            
            # Check if specified model exists
            if self.ollama_model not in available_models:
                # Try to find a similar model
                qwen_models = [m for m in available_models if 'qwen' in m.lower() and 'vl' in m.lower()]
                
                if qwen_models:
                    self.ollama_model = qwen_models[0]
                    print(f"⚠️  Specified model not found, using: {self.ollama_model}")
                else:
                    print(f"❌ Model '{self.ollama_model}' not found!")
                    print(f"   Available models: {', '.join(available_models) if available_models else 'None'}")
                    print(f"\n💡 To install the model, run:")
                    print(f"   ollama pull {self.ollama_model}")
                    raise Exception(f"Model {self.ollama_model} not available")
            
            print(f"📦 Model: {self.ollama_model}")
            print(f"✅ Ollama ready")
            
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.ollama_base_url}\n"
                    f"Make sure Ollama is running:\n"
                    f"  • Check if running: ps aux | grep ollama\n"
                    f"  • Start Ollama: ollama serve\n"
                    f"  • Or it may already be running in background\n"
                    f"Error: {e}"
                )
            raise
    
    def _rotate_api_key(self):
        """Rotate to next API key on rate limit."""
        if len(self.api_keys) <= 1:
            return
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        print(f"🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _ask_tts_model(self) -> str:
        """Ask user for TTS model selection."""
        print("\n🎙️  TTS Model Selection")
        print("=" * 50)
        print("1. 🎵 IndexTTS2 (Local, Zero-shot voice cloning, High quality)")
        print("2. ☁️  Gemini TTS (Cloud-based, Simpler, Whole narration at once)")
        
        while True:
            choice = input("\nEnter choice (1/2): ").strip()
            if choice == "1":
                print("✅ Selected: IndexTTS2")
                return "indextts"
            elif choice == "2":
                print("✅ Selected: Gemini TTS")
                return "gemini"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def _ask_content_type(self):
        """Ask user for content type."""
        print("\n📚 Content Type Selection")
        print("=" * 50)
        print("1. 📖 Manga (right-to-left reading)")
        print("2. 📘 Comic (left-to-right reading)")
        
        while True:
            choice = input("\nEnter choice (1/2): ").strip()
            if choice == "1":
                self.content_type = "manga"
                print("✅ Selected: Manga")
                break
            elif choice == "2":
                self.content_type = "comic"
                print("✅ Selected: Comic")
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def _ask_processing_mode(self) -> str:
        """Ask user for processing mode."""
        print("\n🔄 Processing Mode")
        print("=" * 50)
        print("1. 🆕 Full Analysis (extract panels + AI analysis)")
        print("2. ♻️  Regenerate (use existing JSON, skip AI analysis)")
        
        while True:
            choice = input("\nEnter choice (1/2): ").strip()
            if choice == "1":
                return "full"
            elif choice == "2":
                return "regenerate"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def _get_image_files(self) -> List[Path]:
        """Get all comic page images from input directory."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
        files = []
        
        for ext in extensions:
            files.extend(self.input_dir.glob(ext))
            files.extend(self.input_dir.glob(ext.upper()))
        
        files = [f for f in files if not f.name.startswith('.')]
        return sorted(files)
    
    def _load_narrative_prompt(self) -> str:
        """Load the narrative analysis prompt from file."""
        # Use Ollama-specific prompt if using Ollama
        if self.ai_provider == 'ollama':
            prompt_file = Path("prompt_narrative_ollama.txt")
        else:
            prompt_file = Path("prompt_narrative.txt")
        
        if prompt_file.exists():
            return prompt_file.read_text()
        else:
            # Fallback inline prompt if file doesn't exist
            return """You are analyzing extracted comic panels for narrative video creation.
Score each panel (1-10), generate engaging narrative, and map narrative to specific panels.
Return JSON with panel_scores, narrative_segments, and top_panels."""
    
    def analyze_panels(self, panels: List[Dict], page_name: str) -> Optional[Dict]:
        """
        Analyze extracted panels with AI (Gemini or Ollama).
        
        Returns analysis with:
        - Panel importance scores
        - Narrative text
        - Panel-to-narrative mapping
        """
        provider_name = "Ollama" if self.ai_provider == 'ollama' else "Gemini"
        print(f"🤖 Analyzing {len(panels)} panels with {provider_name}...")
        
        if self.ai_provider == 'ollama':
            return self._analyze_with_ollama(panels, page_name)
        else:
            return self._analyze_with_gemini(panels, page_name)
    
    def _analyze_with_gemini(self, panels: List[Dict], page_name: str) -> Optional[Dict]:
        """Analyze panels using Gemini AI."""
        # Prepare panel images
        panel_images = []
        for panel in panels:
            img_rgb = cv2.cvtColor(panel['image'], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            panel_images.append(pil_img)
        
        reading_order = "RIGHT-TO-LEFT" if self.content_type == "manga" else "LEFT-TO-RIGHT"
        
        # Load narrative prompt from file
        narrative_prompt = self._load_narrative_prompt()
        
        # Build context from previous narrations
        context_section = ""
        if self.narrative_history:
            context_section = "\n\n=== PREVIOUS NARRATIONS FOR CONTEXT ===\n"
            context_section += "Use these to maintain story continuity and consistent tone:\n\n"
            for i, prev_narration in enumerate(self.narrative_history, 1):
                context_section += f"Page {prev_narration['page_number']}:\n"
                context_section += f"{prev_narration['narrative']}\n\n"
            context_section += "=== END OF PREVIOUS CONTEXT ===\n"
        
        prompt = f"""CONTENT TYPE: {self.content_type.upper()}
READING ORDER: {reading_order}
NUMBER OF PANELS: {len(panels)}
{context_section}
{narrative_prompt}"""
        
        try:
            content_parts = [prompt] + panel_images
            response = self.model.generate_content(content_parts)
            analysis = self._parse_json_response(response.text)
            
            # Store this narration in history for next pages
            self._update_narrative_history(analysis, page_name)
            
            print(f"✅ Analysis complete")
            return analysis
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Analysis failed: {e}")
            
            # Check if it's a quota/rate limit error
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                print("🔄 Quota exceeded, rotating to next API key...")
                self._rotate_api_key()
                # Don't return None - let the caller retry
                raise Exception(f"RETRY_WITH_NEXT_KEY: {error_msg}")
            
            # For other errors, rotate key but still return None
            self._rotate_api_key()
            return None
    
    def _analyze_with_ollama(self, panels: List[Dict], page_name: str) -> Optional[Dict]:
        """Analyze panels using Ollama with Qwen3-VL."""
        import io
        
        reading_order = "RIGHT-TO-LEFT" if self.content_type == "manga" else "LEFT-TO-RIGHT"
        
        # Load narrative prompt from file
        narrative_prompt = self._load_narrative_prompt()
        
        # Build context from previous narrations
        context_section = ""
        if self.narrative_history:
            context_section = "\n\n=== PREVIOUS NARRATIONS FOR CONTEXT ===\n"
            context_section += "Use these to maintain story continuity and consistent tone:\n\n"
            for i, prev_narration in enumerate(self.narrative_history, 1):
                context_section += f"Page {prev_narration['page_number']}:\n"
                context_section += f"{prev_narration['narrative']}\n\n"
            context_section += "=== END OF PREVIOUS CONTEXT ===\n"
        
        # Process panels one at a time to avoid context overflow
        panel_analyses = []
        
        try:
            client = ollama.Client(host=self.ollama_base_url)
            
            print(f"   🔄 Analyzing {len(panels)} panels individually with Ollama...")
            
            for idx, panel in enumerate(panels, 1):
                print(f"   📸 Processing panel {idx}/{len(panels)}...")
                
                # Prepare single panel image as base64
                img_rgb = cv2.cvtColor(panel['image'], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Simplified prompt for single panel
                single_panel_prompt = f"""Analyze this comic panel (Panel {idx} of {len(panels)}).

CONTENT TYPE: {self.content_type.upper()}
READING ORDER: {reading_order}

Describe what you see:
- Characters and their actions/expressions
- Dialogue or text (transcribe exactly)
- Key visual elements
- Story significance (score 1-10)

Keep it brief and factual."""
                
                # Build message with single image
                messages = [{
                    'role': 'user',
                    'content': single_panel_prompt,
                    'images': [img_base64]
                }]
                
                response = client.chat(
                    model=self.ollama_model,
                    messages=messages,
                    options={
                        'temperature': 0.7,
                        'num_predict': 1000,  # Moderate limit for single panel
                        'num_ctx': 4096
                    }
                )
                
                panel_text = response['message']['content']
                
                if panel_text and len(panel_text.strip()) > 0:
                    panel_analyses.append({
                        'panel_number': idx,
                        'description': panel_text.strip()
                    })
                    print(f"      ✓ Panel {idx} analyzed ({len(panel_text)} chars)")
                else:
                    print(f"      ⚠️  Panel {idx} returned empty")
            
            # Now create the final narrative from all panel descriptions
            print(f"   📝 Generating final narrative from {len(panel_analyses)} panel descriptions...")
            
            # Combine panel descriptions
            combined_descriptions = "\n\n".join([
                f"Panel {p['panel_number']}: {p['description']}" 
                for p in panel_analyses
            ])
            
            # Final narrative generation prompt
            final_prompt = f"""Based on these panel descriptions, create a dramatic narrative summary.

PANELS ANALYZED:
{combined_descriptions}

{narrative_prompt}

Remember: Output ONLY raw JSON. No markdown, no extra text."""
            
            messages = [{
                'role': 'user',
                'content': final_prompt
            }]
            
            response = client.chat(
                model=self.ollama_model,
                messages=messages,
                options={
                    'temperature': 0.7,
                    'num_predict': 6000,  # Higher limit for final narrative
                    'num_ctx': 8192
                }
            )
            
            response_text = response['message']['content']
            
            # Check if response is empty
            if not response_text or len(response_text.strip()) == 0:
                print(f"   ❌ Ollama returned empty response")
                return None
            
            # Debug info
            print(f"   📝 Final response length: {len(response_text)} chars")
            print(f"   📝 Response preview: {response_text[:300]}...")
            
            # Check if response looks incomplete
            if not response_text.rstrip().endswith('}'):
                print(f"   ⚠️  WARNING: Response may be incomplete (doesn't end with }})")
                print(f"   Last 100 chars: ...{response_text[-100:]}")
            
            # Try to parse JSON
            try:
                analysis = self._parse_json_response(response_text)
            except ValueError as ve:
                print(f"   ❌ JSON parsing failed: {ve}")
                
                # Save raw response for debugging
                debug_file = Path("cache") / f"{page_name}_ollama_response.txt"
                debug_file.write_text(response_text)
                print(f"   💾 Saved raw response to: {debug_file}")
                raise ve
            
            # Store this narration in history for next pages
            self._update_narrative_history(analysis, page_name)
            
            print(f"✅ Analysis complete")
            return analysis
            
        except Exception as e:
            print(f"❌ Ollama analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _update_narrative_history(self, analysis: Dict, page_name: str):
        """Update narrative history with latest analysis."""
        narrative_text = ""
        if 'narrative_segments' in analysis:
            narrative_text = " ".join([seg.get('text', '') for seg in analysis['narrative_segments']])
        elif 'page_narrative' in analysis:
            narrative_text = analysis['page_narrative']
        
        # Add to history
        page_number = len(self.narrative_history) + 1
        self.narrative_history.append({
            'page_number': page_number,
            'page_name': page_name,
            'narrative': narrative_text
        })
        
        # Keep only last N narrations
        if len(self.narrative_history) > self.max_history:
            self.narrative_history = self.narrative_history[-self.max_history:]
    
    def _parse_json_response(self, text: str) -> Dict:
        """Extract JSON from Gemini response."""
        text = text.strip()
        
        # Remove markdown code blocks
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                if part.strip().startswith('json'):
                    text = part[4:].strip()
                elif '{' in part:
                    text = part.strip()
        
        # Find JSON bounds
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        
        return json.loads(text[start:end])
    
    def save_narration_json(self, analysis: Dict, page_name: str, text_dir: Path):
        """Save narration to editable JSON file."""
        narration_file = text_dir / f"{page_name}_narration.json"
        
        with open(narration_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Narration saved: {narration_file.name}")
        return narration_file
    
    def load_narration_json(self, page_name: str, text_dir: Path) -> Optional[Dict]:
        """Load narration from JSON file."""
        narration_file = text_dir / f"{page_name}_narration.json"
        
        if not narration_file.exists():
            print(f"⚠️  Narration file not found: {narration_file.name}")
            return None
        
        with open(narration_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        print(f"📖 Loaded narration: {narration_file.name}")
        return analysis
    
    def process_pages(self):
        """Main processing pipeline."""
        print("\n🚀 Enhanced Comic Narrative Pipeline")
        print("=" * 60)
        
        # Get content type
        self._ask_content_type()
        
        # Get processing mode
        mode = self._ask_processing_mode()
        
        # Get project name
        project_name = input("\n📁 Enter project name: ").strip() or "default"
        safe_name = ''.join(c for c in project_name if c.isalnum() or c in '_-')
        
        # Setup output directories
        project_dir = self.output_dir / safe_name
        videos_dir = project_dir / "videos"
        audio_dir = project_dir / "audio"
        text_dir = project_dir / "text"
        
        # Create temporary directory for intermediate videos
        temp_videos_dir = Path(tempfile.mkdtemp())
        
        for directory in [project_dir, videos_dir, audio_dir, text_dir]:
            directory.mkdir(exist_ok=True)
        
        # Get pages
        pages = self._get_image_files()
        if not pages:
            print("❌ No images found in comic_pages/")
            return
        
        # Use optimized parallel processing for full Gemini TTS flow
        if self.tts_model == 'gemini' and mode == "full":
            print(f"\n📚 Processing {len(pages)} pages with pipelined Gemini optimization")
            print(f"⚡ Pipeline: Analyze page N → TTS page N-1 → Video page N-1 (overlapped)")
            self._process_pages_parallel_gemini(pages, project_dir, temp_videos_dir, text_dir, audio_dir)
        else:
            print(f"\n📚 Processing {len(pages)} pages in '{mode}' mode")
            print(f"⚡ Internal optimizations: Audio generation & video frames")
            self._process_pages_sequential(pages, mode, temp_videos_dir, text_dir, audio_dir)
        
        # Combine all videos into one
        print(f"\n{'='*60}")
        print("🎬 Combining all videos into single file...")
        print(f"{'='*60}")
        
        try:
            video_files = sorted(temp_videos_dir.glob("page_*.mp4"))
            if video_files:
                output_path = videos_dir / f"{project_name}_complete.mp4"
                self.video_generator.combine_videos(video_files, output_path)
            
            print(f"\n{'='*60}")
            print(f"🎉 Processing complete!")
            print(f"📝 Narration JSON files: {text_dir}")
            print(f"   (Edit these and rerun in 'Regenerate' mode)")
            print(f"🎬 Combined video: {videos_dir / f'{project_name}_complete.mp4'}")
            print(f"{'='*60}")
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_videos_dir, ignore_errors=True)
    
    def _process_pages_parallel_gemini(self, pages: List[Path], project_dir: Path, temp_videos_dir: Path, text_dir: Path, audio_dir: Path):
        """
        Optimized pipelined processing for Gemini TTS:
        - Analyze page N while generating TTS for page N-1
        - Generate video for page N-1 while analyzing page N
        """
        print(f"\n{'='*60}")
        print("Pipelined processing: Analysis → TTS → Video (overlapped)")
        print(f"{'='*60}")
        
        # Thread pool for parallel TTS and video generation
        executor = ThreadPoolExecutor(max_workers=4)
        pending_futures = []
        
        for i, page_path in enumerate(pages, 1):
            print(f"\n{'='*60}")
            print(f"Page {i}/{len(pages)}: {page_path.name}")
            print(f"{'='*60}")
            
            try:
                # Extract panels
                print("✂️  Extracting panels...")
                panels = self.panel_extractor.extract_panels(page_path, self.content_type)
                if not panels:
                    print("⚠️  No panels found, skipping")
                    continue
                
                print(f"  Found {len(panels)} panels")
                
                # Analyze with AI (sequential for context)
                print("🤖 Analyzing panels...")
                max_retries = len(self.api_keys) if self.ai_provider == 'gemini' else 1
                analysis = None
                
                for retry in range(max_retries):
                    try:
                        analysis = self.analyze_panels(panels, page_path.stem)
                        if analysis:
                            break
                    except Exception as e:
                        error_msg = str(e)
                        if "RETRY_WITH_NEXT_KEY" in error_msg and self.ai_provider == 'gemini':
                            if retry < max_retries - 1:
                                print(f"   🔄 Retrying with next API key ({retry + 2}/{max_retries})...")
                                time.sleep(2)
                                continue
                
                if not analysis:
                    print("⚠️  Analysis failed, skipping")
                    continue
                
                # Save narration JSON
                self.save_narration_json(analysis, page_path.stem, text_dir)
                
                # Prepare narrative segments
                narrative_segments = analysis.get('narrative_segments', [])
                if not narrative_segments:
                    narrative_segments = [{
                        'segment_id': 1,
                        'text': analysis.get('page_narrative', ''),
                        'show_panels': analysis.get('top_panels', [])[:2]
                    }]
                
                # Submit TTS + video generation to background thread
                # This happens in parallel while we analyze the next page
                def process_audio_video(page_idx, page_name, page_panels, page_analysis, page_narrative):
                    try:
                        print(f"\n[Page {page_idx}] 🎙️  Generating TTS audio...")
                        
                        # Create page-specific audio directory
                        page_audio_dir = audio_dir / f"page_{page_idx:03d}"
                        page_audio_dir.mkdir(exist_ok=True)
                        
                        # Generate audio
                        audio_segs = self.audio_generator.generate_audio(page_narrative, page_audio_dir)
                        
                        if not audio_segs:
                            print(f"[Page {page_idx}] ⚠️  No audio generated")
                            return None
                        
                        print(f"[Page {page_idx}] 🎬 Creating video...")
                        
                        # Create video
                        output_vid = temp_videos_dir / f"page_{page_idx:03d}.mp4"
                        self.video_generator.create_video(page_panels, audio_segs, page_analysis, output_vid)
                        
                        print(f"[Page {page_idx}] ✅ Complete: {page_name}")
                        return output_vid
                        
                    except Exception as e:
                        print(f"[Page {page_idx}] ❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        return None
                
                # Submit to executor (runs in background)
                future = executor.submit(
                    process_audio_video, 
                    i, 
                    page_path.name, 
                    panels, 
                    analysis, 
                    narrative_segments
                )
                pending_futures.append(future)
                
                # Rate limiting for AI analysis
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Wait for all background tasks to complete
        print(f"\n{'='*60}")
        print("⏳ Waiting for remaining TTS & video tasks to complete...")
        print(f"{'='*60}")
        
        for future in as_completed(pending_futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"   ❌ Background task error: {e}")
        
        executor.shutdown(wait=True)
        print("✅ All pages processed!")
    
    def _process_pages_sequential(self, pages: List[Path], mode: str, temp_videos_dir: Path, text_dir: Path, audio_dir: Path):
        for i, page_path in enumerate(pages, 1):
            print(f"\n{'='*60}")
            print(f"Page {i}/{len(pages)}: {page_path.name}")
            print(f"{'='*60}")
            
            try:
                # Extract panels (always needed for images)
                print("✂️  Extracting panels...")
                panels = self.panel_extractor.extract_panels(page_path, self.content_type)
                if not panels:
                    print("⚠️  No panels found, skipping")
                    continue
                
                print(f"  Found {len(panels)} panels")
                
                # Get or generate analysis
                if mode == "regenerate":
                    # Load existing narration JSON
                    analysis = self.load_narration_json(page_path.stem, text_dir)
                    if not analysis:
                        print("⚠️  No existing narration, skipping")
                        continue
                else:
                    # Analyze with AI (with retry on quota errors for Gemini)
                    if self.ai_provider == 'ollama':
                        # Ollama doesn't need retries
                        max_retries = 1
                    else:
                        # Gemini may need retries with different API keys
                        max_retries = len(self.api_keys)
                    
                    analysis = None
                    
                    for retry in range(max_retries):
                        try:
                            analysis = self.analyze_panels(panels, page_path.stem)
                            if analysis:
                                break
                        except Exception as e:
                            error_msg = str(e)
                            if "RETRY_WITH_NEXT_KEY" in error_msg and self.ai_provider == 'gemini':
                                if retry < max_retries - 1:
                                    print(f"   Retrying with next API key ({retry + 2}/{max_retries})...")
                                    time.sleep(2)
                                    continue
                                else:
                                    print(f"   ❌ All {max_retries} API keys exhausted")
                                    analysis = None
                                    break
                            else:
                                analysis = None
                                break
                    
                    if not analysis:
                        print("⚠️  Analysis failed after all retries, skipping")
                        continue
                    
                    # Save narration to JSON for editing
                    self.save_narration_json(analysis, page_path.stem, text_dir)
                
                # Generate audio (optimized internally)
                narrative_segments = analysis.get('narrative_segments', [])
                if not narrative_segments:
                    narrative_segments = [{
                        'segment_id': 1,
                        'text': analysis.get('page_narrative', ''),
                        'show_panels': analysis.get('top_panels', [])[:2]
                    }]
                
                audio_segments = self.audio_generator.generate_audio(narrative_segments, audio_dir)
                if not audio_segments:
                    print("⚠️  No audio generated, skipping")
                    continue
                
                # Create video in temp directory (optimized internally)
                output_video = temp_videos_dir / f"page_{i:03d}.mp4"
                self.video_generator.create_video(panels, audio_segments, analysis, output_video)
                
                print(f"✅ Completed: {page_path.name}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Rate limiting (only for full analysis mode)
            if mode == "full":
                time.sleep(1)


# ==================== MAIN ====================

def main():
    """Main entry point."""
    try:
        pipeline = EnhancedComicPipeline()
        pipeline.process_pages()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
