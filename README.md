# Comic Narrative Pipeline

Professional narrative video generator for comics and manga. Extracts panels using computer vision and creates engaging narrative videos using AI-powered story analysis.

## 🎥 Demo

<video src="https://github.com/user-attachments/assets/your-username-repo/demo.mp4" controls></video>

> **To make the video work:** After pushing to GitHub, drag and drop `static/demo.mp4` into any GitHub issue/comment, copy the generated URL, and replace the src above.

*Example output: AI-generated narrative video from comic panels with synchronized TTS narration*

## 🎙️ TTS Options

Choose between two powerful text-to-speech engines:

### **IndexTTS2** (Local, Zero-shot Voice Cloning)
- 🎙️ Zero-shot voice cloning from any reference audio
- 🎭 Automatic emotional expression control
- 🌍 Multi-language support (Chinese, English, mixed)
- 🚀 Superior quality and naturalness
- 💾 Runs locally, no API costs

### **Gemini TTS** (Cloud, Fast & Easy)
- ☁️ Cloud-based, no model downloads needed
- ⚡ Fast generation with pipelined processing
- 🎵 High-quality narration voices (Enceladus, Puck, Charon, etc.)
- 🔄 API key rotation for higher limits
- 📦 Processes entire page narration at once

You'll be prompted to choose your TTS engine when running the processor.

## 📁 Modular Architecture

### **Comic Processor** - `comic_processor/`
- ✏️ **Editable Narration** - Save to JSON, edit, and regenerate
- ♻️ **Regeneration Mode** - Skip AI analysis, use your edited JSON
- 🧩 **Modular Design** - Separate panel extraction, audio generation, and video composition
- 📚 **Context History** - AI remembers previous 3 pages for better continuity
- 🤖 **Dual AI Support** - Use Gemini (cloud) or Ollama (local) for analysis
- [See comic_processor/README.md for detailed documentation](comic_processor/README.md)

## ✨ Features

- 🎯 **Intelligent Panel Extraction** - Computer vision detects actual panel boundaries
- 🤖 **AI-Powered Analysis** - Gemini or Ollama AI scores panels and generates engaging narratives
- 🎙️ **IndexTTS2 Voice Synthesis** - Zero-shot voice cloning with emotional expression
- 🎬 **Professional Videos** - Dynamic panel layouts with perfect timing
- 📖 **Cultural Awareness** - Supports both manga (right-to-left) and comics (left-to-right)
- ✏️ **Editable Narrations** - Save to JSON, edit, and regenerate without re-analyzing

## Installation

### 1. Clone or download this repository

### 2. Install base dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup IndexTTS2 (Optional - for local TTS)

**Only needed if you want to use IndexTTS instead of Gemini TTS**

**Automated Setup:**
```bash
./setup_indextts.sh
```

**Manual Setup:**
```bash
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
pip install -e .
cd ..
```

### 4. Setup AI Provider

**Option A: Gemini (Cloud AI - Recommended for beginners)**

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free Gemini API key at: https://makersuite.google.com/app/apikey

**For Gemini TTS** (optional - enables TTS generation):
The same Gemini API key also supports TTS. For higher rate limits, add multiple keys:
```bash
GEMINI_API_KEYS=key1,key2,key3
```

**Option B: Ollama (Local AI - No API key needed)**

1. Install Ollama: https://ollama.ai
2. Download vision model:
   ```bash
   ollama pull qwen2-vl:latest
   ```
3. Set in `.env`:
   ```bash
   AI_PROVIDER=ollama
   OLLAMA_MODEL=qwen2-vl:latest
   ```

## 🚀 Quick Start

### 1. Add Your Comics

Place comic images in the `comic_pages/` folder:
```
comic_pages/
  ├── page_001.png
  ├── page_002.png
  └── page_003.png
```

### 2. Run the Processor

```bash
cd comic_processor
python3 main.py
```

### 3. Select TTS Engine

- **IndexTTS**: Local voice cloning, no API costs (requires model download)
- **Gemini TTS**: Cloud-based, fast, high-quality (requires API key)

### 4. Choose Processing Mode

- **Full Analysis**: First time processing (extracts panels, AI analysis, generates narration)
- **Regenerate**: After editing JSON files (skips AI, regenerates audio/video only)

### 5. Select Content Type

- **Manga**: Right-to-left reading order (Japanese comics)
- **Comic**: Left-to-right reading order (Western comics)

### 6. Enter Project Name

Give your project a name. Output will be saved to `results/your_project_name/`

## 📤 Output Structure

```
results/your_project_name/
├── text/
│   ├── page_001_narration.json    # ✏️ Edit these to customize narration
│   ├── page_002_narration.json
│   └── ...
├── audio/
│   ├── segment_01.wav             # Generated TTS audio
│   ├── segment_02.wav
│   └── ...
└── videos/
    └── your_project_complete.mp4  # 🎬 Final combined video
```

## ✏️ Edit & Regenerate Workflow

1. **First Run**: Process your comics with Full Analysis mode
2. **Edit**: Modify `results/{project}/text/*.json` files
   - Change narration text
   - Adjust which panels to show
   - Modify panel scores
3. **Regenerate**: Run in Regenerate mode to create new videos with your edits
4. **Iterate**: Repeat until perfect!

## 🔮 Next Steps & Future Features

The following features are planned for future releases:

- **📝 Individual Section Editing**: Replace only specific narrative segments with regenerated or human-written content (instead of editing the entire JSON)
- **🎬 Full Video Mode**: Generate complete multi-page videos in one command with optimized batching
- **🎨 Custom Panel Layouts**: More control over how panels appear in the final video
- **🗣️ Multi-Voice Support**: Assign different voices to different characters or narrative types

## 🔧 How It Works

### Processing Pipeline

```
1. Panel Extraction (Computer Vision)
   ├─ Detect panel boundaries using gutter detection
   ├─ Extract individual panels as images
   ├─ Sort in correct reading order (manga/comic aware)
   └─ Cache results for speed

2. AI Analysis (Gemini or Ollama)
   ├─ Analyze each panel's visual content
   ├─ Score panel importance (1-10)
   ├─ Generate engaging narrative text
   ├─ Map narrative segments to panels
   └─ Save as editable JSON

3. Audio Generation (TTS)
   ├─ IndexTTS: Clone your voice from reference sample
   │   ├─ Convert narrative to speech
   │   ├─ Apply emotional expression automatically
   │   └─ Generate high-quality audio segments
   │
   └─ Gemini TTS: Cloud-based narration
       ├─ Process entire page narration at once
       ├─ Use professional narration voices
       ├─ Automatic API key rotation for rate limits
       └─ Fast parallel generation

4. Video Composition
   ├─ Sync panels with audio timing
   ├─ Create dynamic layouts (single/dual/grid)
   ├─ Combine all pages into final video
   └─ Export as MP4
```

## ⚙️ Configuration

### AI Narrative Prompts

**Gemini**: Edit `prompt_narrative.txt` to customize:
- Panel scoring criteria
- Narrative style (dramatic, factual, etc.)
- Output format
- Story analysis approach

**Ollama**: Edit `prompt_narrative_ollama.txt` for local AI customization

### Voice Customization (IndexTTS only)

Add your voice samples to `voice_samples/`:
```bash
voice_samples/
  ├── narrator.wav     # Your voice for cloning
  ├── character1.wav   # Optional: different voices
  └── character2.wav
```

**Voice Requirements:**
- 3-10 seconds of clear speech
- WAV format, 16kHz+ sample rate
- Minimal background noise
- Natural speaking pace

**Note**: Gemini TTS uses pre-built voices (Enceladus, Puck, Charon, etc.) and doesn't require voice samples.

### Panel Extraction Tuning

Edit `comic_processor/utils/panel_extractor.py`:
```python
PanelExtractor(
    min_panel_pct=2.0,   # Minimum panel size (% of page)
    max_panel_pct=90.0   # Maximum panel size (% of page)
)
```

### Video Settings

Edit `comic_processor/utils/video_generator.py`:
```python
frame_rate = 30           # FPS
frame_size = (1920, 1080) # Resolution
```

## 🔍 Troubleshooting

### Panel Extraction Issues
- **No panels detected**: Check `cache/extracted_panels/` to see detection results
- **Wrong panels**: Adjust `min_panel_pct` and `max_panel_pct` in panel extractor
- **Partial panels**: Ensure comic has clear gutters (white space) between panels
- **Best results**: High-quality scans with distinct panel boundaries

### AI Analysis Errors
- **Gemini API error**: 
  - Verify `.env` has valid `GEMINI_API_KEY`
  - Check quota at https://makersuite.google.com
  - Add multiple keys with `GEMINI_API_KEYS=key1,key2,key3`
- **Ollama not responding**:
  - Ensure Ollama is running: `ollama list`
  - Pull model: `ollama pull qwen2-vl:latest`
  - Check `OLLAMA_BASE_URL` in `.env` (default: `http://localhost:11434`)

### Audio/TTS Problems
- **IndexTTS2 not found**: Run `./setup_indextts.sh` or manually install (only needed if using IndexTTS)
- **Model loading fails**: Check `index-tts/checkpoints/` has all model files (~8GB)
- **Voice cloning poor quality**: Use 5-10 second clear voice sample in `voice_samples/`
- **Gemini TTS quota exceeded**: Add multiple API keys with `GEMINI_API_KEYS=key1,key2,key3`
- **Gemini TTS variations**: Voice may vary slightly between generations (inherent to AI TTS)
- **Audio generation slow**: First run downloads models; subsequent runs are faster

### Video Issues
- **FFmpeg not found**: Install with `brew install ffmpeg` (macOS) or system package manager
- **Video won't play**: Check codec support; try VLC media player
- **Sync issues**: Check audio files in `results/{project}/audio/` are valid

## 📋 Requirements

### Software
- **Python**: 3.10 or higher
- **Git**: With Git-LFS enabled (for model downloads)
- **FFmpeg**: For video encoding
- **Ollama** (optional): For local AI processing

### API Keys
- **Gemini API** (free tier): https://makersuite.google.com/app/apikey
  - Used for AI analysis (required)
  - Also supports Gemini TTS (optional, alternative to IndexTTS)
  - Free tier: 15 requests/day per API key (use multiple keys for higher limits)
- **OR Ollama** (no API key needed): https://ollama.ai

### Storage
- **~8GB**: IndexTTS2 models (optional, only if using IndexTTS)
- **~500MB**: Base dependencies
- **Variable**: Cache and results (depends on usage)

## 📁 Project Structure

```
comic-processor/
├── comic_processor/              # Main application (modular design)
│   ├── main.py                  # Pipeline orchestrator
│   ├── utils/
│   │   ├── panel_extractor.py   # Computer vision panel detection
│   │   ├── audio_generator_indextts2.py  # IndexTTS2 integration
│   │   ├── audio_generator_gemini.py     # Gemini TTS integration
│   │   ├── video_generator.py   # Video composition
│   │   └── voice_selector.py    # Interactive voice selection
│   └── README.md                # Detailed module documentation
│
├── prompt_narrative.txt         # Gemini AI prompt template
├── prompt_narrative_ollama.txt  # Ollama AI prompt template
├── setup_indextts.sh           # Automated IndexTTS2 setup
├── requirements.txt            # Python dependencies
├── .env                        # API keys (you create this)
├── .gitignore                  # Git ignore rules
│
├── comic_pages/                # 📥 INPUT: Your comic images
├── voice_samples/              # 🎤 INPUT: Voice cloning references
│
├── results/                    # 📤 OUTPUT: Generated content
│   └── {project}/
│       ├── text/              # Editable narration JSON
│       ├── audio/             # Generated audio segments
│       └── videos/            # Final MP4 videos
│
├── cache/                      # Auto-generated cache
│   └── extracted_panels/      # Cached panel extractions
│
└── index-tts/                  # IndexTTS2 installation (auto-installed)
    └── checkpoints/            # TTS models (~8GB, auto-downloaded)
```

## 💡 Tips & Best Practices

### Comic Preparation
- **Sequential naming**: `page_001.png`, `page_002.png`, etc.
- **High quality**: 1200px+ width recommended for best panel detection
- **Clear panels**: Comics with distinct gutters work best
- **Formats**: PNG, JPG, JPEG supported

### Voice Cloning
- **Quality sample**: 5-10 seconds of clear speech
- **Natural delivery**: Read naturally, not overly dramatic
- **Clean audio**: Minimal background noise
- **Good mic**: Better input = better cloned voice

### Workflow Optimization
- **Start small**: Process 1-2 pages first to test settings
- **Use cache**: Panel extraction is cached; AI re-runs are fast
- **Edit JSON**: Tweak narration without re-analyzing with AI
- **Multiple voices**: Create different voice samples for variety

### Performance
- **First run**: Slower (downloads models, generates cache)
- **Subsequent runs**: Much faster (uses cache)
- **Regenerate mode**: Fastest (skips AI analysis entirely)
- **API keys**: Use multiple Gemini keys to avoid rate limits

## 📜 License

MIT License - Free to use and modify

## 🙏 Credits

- **IndexTTS2**: https://github.com/index-tts/index-tts
- **Gemini AI**: Google's multimodal AI
- **OpenCV**: Computer vision panel extraction
- Inspired by [manga-reader](https://github.com/pashpashpash/manga-reader)

## 🆘 Support

Having issues? Try these steps:
1. ✅ Check the [Troubleshooting](#-troubleshooting) section
2. 📁 Review `cache/` folder for analysis results
3. 🧪 Test with a single page first
4. 📋 Verify all dependencies are installed
5. 🔍 Check `results/{project}/text/` JSON for issues
