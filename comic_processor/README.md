# Comic Processor - Modular Edition

Enhanced narrative video generation from comic/manga pages with **editable narration** and **modular architecture**.

## 🆕 New Features

### 1. **Editable Narration JSON**
- All narrations are saved as JSON files in `results/{project}/text/`
- Edit the text, panel mappings, or segments as you like
- Rerun in "Regenerate" mode to skip AI analysis and use your edited JSON

### 2. **Regeneration Mode**
- **Full Analysis Mode**: Extract panels → AI analysis → Save JSON → Generate audio/video
- **Regenerate Mode**: Load existing JSON → Generate audio/video (no AI calls)

### 3. **Modular Architecture**
```
comic_processor/
├── main.py                    # Main pipeline orchestrator
├── utils/
│   ├── panel_extractor.py     # Computer vision panel detection
│   ├── audio_generator.py     # TTS audio generation
│   └── video_generator.py     # Video composition & combining
```

### 4. **Context-Aware Narration**
- AI remembers the last 3 pages of narration
- Better story continuity and consistent tone
- Smarter character references and transitions

## 🚀 Quick Start

### Run the Pipeline
```bash
cd comic_processor
python3 main.py
```

### Workflow Options

#### Option 1: Full Analysis (First Time)
1. Select "Full Analysis" mode
2. Pipeline extracts panels, analyzes with AI, saves narration JSON
3. Generates audio and video
4. Creates `results/{project}/text/{page}_narration.json` files

#### Option 2: Edit & Regenerate
1. Edit the JSON files in `results/{project}/text/`
2. Modify `text`, `show_panels`, or any field
3. Select "Regenerate" mode
4. Pipeline loads your edited JSON and regenerates audio/video

## 📝 Narration JSON Format

Example `page_001_narration.json`:
```json
{
  "panel_scores": {
    "1": 8,
    "2": 9,
    "3": 6
  },
  "narrative_segments": [
    {
      "segment_id": 1,
      "text": "Your custom narration here...",
      "show_panels": [1, 2]
    },
    {
      "segment_id": 2,
      "text": "Another segment of narration...",
      "show_panels": [3]
    }
  ],
  "top_panels": [2, 1, 3]
}
```

**Editable Fields:**
- `text`: Change the narration
- `show_panels`: Which panels to display (array of panel numbers)
- `panel_scores`: Adjust importance ratings
- Add/remove segments as needed

## 📁 Output Structure

```
results/{project_name}/
├── text/
│   ├── page_001_narration.json    # ✏️ EDIT THESE!
│   ├── page_002_narration.json
│   └── ...
├── audio/
│   ├── segment_01.wav
│   ├── segment_02.wav
│   └── ...
└── videos/
    └── {project_name}_complete.mp4  # 🎬 Final output
```

## 🔧 Customization

### Adjust Context History
In `main.py`, change:
```python
self.max_history = 3  # Number of previous pages to remember
```

### Adjust Panel Detection Sensitivity
In `utils/panel_extractor.py`:
```python
PanelExtractor(
    min_panel_pct=2.0,   # Minimum panel size (% of page)
    max_panel_pct=90.0   # Maximum panel size (% of page)
)
```

### Change Video Settings
In `main.py`:
```python
self.video_generator = VideoGenerator(
    frame_rate=30,           # FPS
    frame_size=(1920, 1080)  # Resolution
)
```

## 🛠️ Module Overview

### `main.py`
- Main pipeline orchestrator
- Handles Gemini AI analysis
- Manages context history
- Coordinates all utilities

### `utils/panel_extractor.py`
- Computer vision-based panel detection
- Reading order sorting (manga vs comic)
- Contour detection and filtering

### `utils/audio_generator.py`
- Text-to-speech generation
- Audio duration calculation
- Text cleaning for better pronunciation

### `utils/video_generator.py`
- Frame composition (single/dual/grid layouts)
- Audio-video synchronization
- Video combining with FFmpeg

## 💡 Tips

### Editing Narration
1. Run full analysis first to get AI-generated base
2. Review JSON files and adjust text
3. Change `show_panels` to display different panels
4. Rerun in regenerate mode

### Panel Numbers
- Panels are numbered in reading order (1, 2, 3, ...)
- Manga: right-to-left, top-to-bottom
- Comic: left-to-right, top-to-bottom

### Segment Timing
- Audio duration determines video segment length
- Longer text = longer audio = longer on-screen time
- Edit text length to control pacing

## 🔄 Typical Workflow

```
1. Place comics in comic_pages/
2. Run: python3 main.py (select "Full Analysis")
3. Review: results/{project}/text/*.json
4. Edit: Modify narration text and panel mappings
5. Rerun: python3 main.py (select "Regenerate")
6. Iterate: Keep editing and regenerating until perfect
7. Watch: results/{project}/videos/{project}_complete.mp4
```

## 📦 Dependencies

Same as parent project - all dependencies installed via `requirements.txt` in root.
