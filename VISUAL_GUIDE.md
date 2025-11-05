# 📖 IndexTTS2 Visual Guide

## 🗺️ Project Structure (After Setup)

```
comic-processor/
│
├── 📁 index-tts/                     ← IndexTTS repository (cloned)
│   ├── 📁 checkpoints/               ← Models (~8GB)
│   │   ├── config.yaml
│   │   ├── *.pth files
│   │   └── hf_cache/
│   ├── 📁 indextts/                  ← Python package
│   ├── 📁 examples/
│   └── webui.py
│
├── 📁 voice_samples/                 ← Your reference voices
│   ├── narrator_voice.wav
│   ├── character1.wav
│   └── ...
│
├── 📁 comic_processor/               ← Main processor
│   ├── main.py                       ← Entry point (uses IndexTTS2)
│   └── 📁 utils/
│       ├── audio_generator_indextts2.py  ← NEW: IndexTTS2
│       ├── audio_generator.py            ← OLD: StyleTTS2 (backup)
│       ├── panel_extractor.py
│       ├── video_generator.py
│       └── voice_selector.py
│
├── 📁 comic_pages/                   ← Input: your comics
├── 📁 results/                       ← Output: videos
│
├── 📄 requirements.txt               ← Updated for IndexTTS2
├── 📄 .env                           ← Gemini API key
│
├── 📚 QUICK_START.md                 ← Start here!
├── 📚 INDEXTTS_SETUP.md              ← Detailed guide
├── 📚 MIGRATION_COMPLETE.md          ← What changed
├── 📚 COMPARISON.md                  ← IndexTTS2 vs StyleTTS2
├── 📚 SUMMARY.md                     ← This migration summary
└── 🚀 setup_indextts.sh              ← Automated setup script
```

---

## 🔄 Data Flow

```
┌─────────────────┐
│  Comic Pages    │
│  (Input Images) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Panel Extractor │  ← Computer Vision
│   (CV-based)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini AI      │  ← Analyze & Score
│   (Analysis)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IndexTTS2      │  ← Generate Audio
│  (Voice Clone)  │     with Emotion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Video Generator │  ← Compose Video
│  (FFmpeg)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Video   │
│   (results/)    │
└─────────────────┘
```

---

## 🎙️ IndexTTS2 Voice Cloning Flow

```
┌──────────────────────┐
│ Reference Voice      │  3-10 seconds, clear speech
│ (voice_samples/)     │  Any language, any speaker
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ IndexTTS2 Model      │  Zero-shot voice cloning
│ (Encoder)            │  Extracts speaker features
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Text Input           │  Your comic narration
│ + Emotion Control    │  Auto/Neutral/Reference
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ IndexTTS2 Generator  │  Synthesize with cloned voice
│ (Decoder)            │  + Emotional expression
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Output Audio (WAV)   │  Natural speech, cloned voice
│ (audio/)             │  with emotions!
└──────────────────────┘
```

---

## 🎭 Emotion Control System

```
                    ┌─────────────┐
                    │  Text Input │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         [Auto Mode]              [Manual Mode]
              │                         │
              ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  Qwen Emotion    │      │  User Specified  │
    │  Analyzer        │      │  Emotion Vector  │
    └────────┬─────────┘      └────────┬─────────┘
             │                          │
             └──────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ 8D Emotion Vec  │
              │ [h,a,s,f,d,m,s,c]│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  IndexTTS2      │
              │  Voice Gen      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Emotional      │
              │  Speech Output  │
              └─────────────────┘

Legend:
h = happy      f = afraid
a = angry      d = disgusted  
s = sad        m = melancholic
               s = surprised
               c = calm
```

---

## 🚀 Installation Process (Visual)

```
Step 1: Prerequisites
┌────────────────────┐
│ ✓ Python 3.10+     │
│ ✓ Git + Git-LFS    │
│ ✓ 10GB disk space  │
└────────────────────┘
         │
         ▼
Step 2: Clone Repo
┌────────────────────┐
│ git clone          │
│ index-tts/         │
│ index-tts          │
└────────────────────┘
         │
         ▼
Step 3: Download Models
┌────────────────────┐
│ hf download        │
│ IndexTeam/         │
│ IndexTTS-2         │
│ (~8GB, one-time)   │
└────────────────────┘
         │
         ▼
Step 4: Install Deps
┌────────────────────┐
│ pip install -r     │
│ requirements.txt   │
└────────────────────┘
         │
         ▼
Step 5: Install Package
┌────────────────────┐
│ cd index-tts       │
│ pip install -e .   │
└────────────────────┘
         │
         ▼
Step 6: Verify
┌────────────────────┐
│ python -c          │
│ "from indextts..."│
│ ✅ Success!        │
└────────────────────┘
```

---

## 📊 Performance Comparison (Visual)

### Speed
```
StyleTTS2:  ████████████████      (~3 sec/sentence)
IndexTTS2:  ██████████            (~2 sec/sentence)
                    ↑ 50% faster!
```

### VRAM Usage
```
StyleTTS2:  ████████████          (~6GB)
IndexTTS2:  ████████              (~4GB with FP16)
                    ↑ 33% less!
```

### Quality
```
StyleTTS2:  ⭐⭐⭐⭐
IndexTTS2:  ⭐⭐⭐⭐⭐
```

### Features
```
                    StyleTTS2  IndexTTS2
Voice Cloning         ✓          ✓✓✓
Emotion Control       ✗          ✓✓✓
Multi-language        ~          ✓✓✓
Duration Control      ✗          ✓✓✓
Setup Simplicity     ✓✓✓         ✓
```

---

## 🎯 Usage Flow (Visual)

### Before (StyleTTS2)
```
1. Place comics → 2. Run script → 3. Basic TTS → 4. Get video
                                      ↓
                                 Generic voice
                                 No emotion control
```

### After (IndexTTS2)
```
1. Place comics → 2. Select voice → 3. Run script → 4. Advanced TTS → 5. Get video
                         ↓                               ↓
                  Any reference                   Voice cloning
                  (3-10 sec)                      + Auto emotion
                                                  + Better quality
```

---

## 🎨 Emotion Mode Comparison

### Auto Mode (Drama)
```
Input Text: "The hero charged forward with a mighty roar!"
                    ↓
            [Qwen Analysis]
                    ↓
Emotion: [0.2 happy, 0.6 angry, 0 sad, 0 fear, ...]
                    ↓
        [IndexTTS2 Generation]
                    ↓
Output: Energetic, angry-tinged voice 🗣️💥
```

### Neutral Mode (Documentary)
```
Input Text: "The battle took place in the city center."
                    ↓
       [Neutral Emotion Vector]
                    ↓
Emotion: [0, 0, 0, 0, 0, 0, 0, 1.0 calm]
                    ↓
        [IndexTTS2 Generation]
                    ↓
Output: Calm, balanced narration 🗣️📰
```

### Reference Mode (Matching)
```
Reference Voice (happy & excited)
                    ↓
       [Extract Emotion Features]
                    ↓
Emotion: Matches reference voice emotion
                    ↓
        [IndexTTS2 Generation]
                    ↓
Output: Voice clone with matched emotion 🗣️🎭
```

---

## 🗂️ File Relationships

```
main.py
  │
  ├──imports──► audio_generator_indextts2.py
  │                     │
  │                     ├──uses──► IndexTTS2 (index-tts/indextts/)
  │                     │                │
  │                     │                └──loads──► checkpoints/
  │                     │                              (models)
  │                     │
  │                     └──clones──► voice_samples/
  │                                    (reference voices)
  │
  ├──imports──► panel_extractor.py
  │
  ├──imports──► video_generator.py
  │
  └──imports──► voice_selector.py
```

---

## 💾 Disk Space Breakdown

```
Total: ~10GB

index-tts/
├── checkpoints/        ~8GB   ████████
│   ├── Models          ~7GB   ███████
│   └── Cache           ~1GB   █
├── Source code         ~1GB   █
└── Dependencies        ~1GB   █
```

---

## ⚙️ Configuration Options (Visual)

```
┌─────────────────────────────────────────┐
│  IndexTTS2AudioGenerator Configuration │
├─────────────────────────────────────────┤
│                                         │
│  Performance ⚡                         │
│  ├─ USE_FP16 = True    (2x faster)     │
│  ├─ USE_DEEPSPEED = False              │
│  └─ max_workers = 2    (parallel)      │
│                                         │
│  Quality 🎨                             │
│  ├─ MAX_SENTENCE_LENGTH = 200          │
│  ├─ DEFAULT_EMO_ALPHA = 0.7            │
│  └─ MAX_TEXT_TOKENS = 120              │
│                                         │
│  Emotion 🎭                             │
│  ├─ emotion_mode = "auto"              │
│  │   • "auto" - AI detects from text   │
│  │   • "neutral" - Calm narration      │
│  │   • "reference" - Match ref voice   │
│  └─ emotion_intensity = 0.7            │
│                                         │
└─────────────────────────────────────────┘
```

---

## 🎬 Example Workflow Timeline

```
Time    Activity                          Status
────────────────────────────────────────────────
00:00   Load comic pages                  ▓▓▓▓▓
00:05   Extract panels (CV)               ▓▓▓▓▓
00:10   Analyze with Gemini               ▓▓▓▓▓
00:20   Generate audio (IndexTTS2)        ▓▓▓▓▓▓▓▓▓▓ ← Main time
00:40   Compose video (FFmpeg)            ▓▓▓▓▓
00:50   Final encoding                    ▓▓▓
00:55   Complete! ✅

Total: ~55 seconds for 1 page (GPU)
       ~3-5 minutes for 1 page (CPU)
```

---

## 🎓 Learning Curve

```
Complexity
    ▲
    │                 ┌─ Advanced Features
    │                 │   (Emotion tuning)
    │            ┌────┤
    │            │    │
    │       ┌────┤    └─ Custom Config
    │       │    │       (Performance)
    │  ┌────┤    └─ Voice Selection
    │  │    │       (Choose/Record)
    │  │    └─ Basic Usage
    │  │       (Default settings)
    │  └─ Installation
    │     (One-time setup)
    └────────────────────────────► Time
      30m   1h   1d   1w   1m
```

---

## 🔍 Troubleshooting Decision Tree

```
                   [Problem?]
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    [Import Error] [No Models] [Out of Memory]
         │             │             │
         ▼             ▼             ▼
    pip install    Download      Enable FP16
    in index-tts/  from HF       Reduce workers
         │             │             │
         └─────────────┴─────────────┘
                       │
                       ▼
                 [Test Again]
                       │
              ┌────────┴────────┐
              │                 │
           [Works!]        [Still Broken]
              │                 │
              ▼                 ▼
         [Use It!]      [Check Docs/Issues]
```

---

## 📚 Documentation Hierarchy

```
Start Here
    │
    ├─ QUICK_START.md        (Fast setup, 1 page)
    │       │
    │       └─ For quick installation
    │
    ├─ INDEXTTS_SETUP.md     (Complete guide, detailed)
    │       │
    │       ├─ Prerequisites
    │       ├─ Installation
    │       ├─ Configuration
    │       └─ Troubleshooting
    │
    ├─ MIGRATION_COMPLETE.md (What changed, why)
    │       │
    │       └─ For understanding the upgrade
    │
    ├─ COMPARISON.md         (Deep dive, technical)
    │       │
    │       ├─ Feature comparison
    │       ├─ Emotion system
    │       └─ Use cases
    │
    └─ SUMMARY.md           (This migration, complete)
            │
            └─ Overview of everything
```

---

## 🎯 Quick Decision Guide

```
❓ "What do I need to do first?"
   → Read QUICK_START.md → Run setup_indextts.sh

❓ "How do I install everything?"
   → Follow INDEXTTS_SETUP.md step-by-step

❓ "What's different from before?"
   → Read MIGRATION_COMPLETE.md

❓ "Why is IndexTTS2 better?"
   → Read COMPARISON.md

❓ "Something broke, help!"
   → Check Troubleshooting in INDEXTTS_SETUP.md
   → Review error messages (they're helpful!)
   → Check GitHub issues

❓ "I want to understand everything"
   → Read SUMMARY.md (this doc!)
```

---

## 🎊 Success Indicators

```
✅ All Good                    ❌ Needs Attention
─────────────────────────────────────────────────
✓ import indextts works        ✗ Import fails
✓ Models in checkpoints/       ✗ checkpoints/ empty
✓ Voice sample prepared        ✗ No voice sample
✓ Test generation works        ✗ Generation fails
✓ Video created successfully   ✗ Video errors
✓ Audio has emotion            ✗ Robotic voice
```

---

**Visual guide complete! For text instructions, see other .md files.** 📚✨
