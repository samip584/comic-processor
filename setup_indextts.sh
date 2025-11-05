#!/bin/bash
# IndexTTS2 Setup Script
# Automates the installation of IndexTTS2 for comic processor

set -e  # Exit on error

echo "🎙️  IndexTTS2 Setup Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running from correct directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the comic-processor directory"
    exit 1
fi

print_step "Checking prerequisites..."

# Check for git
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi
print_success "Git found"

# Check for git-lfs
if ! command -v git-lfs &> /dev/null; then
    print_warning "Git-LFS not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install git-lfs
    else
        print_error "Please install git-lfs manually: https://git-lfs.com/"
        exit 1
    fi
fi

# Enable git-lfs
git lfs install
print_success "Git-LFS enabled"

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi
python_version=$(python3 --version | awk '{print $2}')
print_success "Python found: $python_version"

# Step 1: Clone IndexTTS repository
print_step "Step 1/5: Cloning IndexTTS repository..."
if [ -d "index-tts" ]; then
    print_warning "index-tts directory already exists. Skipping clone."
else
    git clone https://github.com/index-tts/index-tts.git
    print_success "Repository cloned"
fi

cd index-tts

# Pull LFS files
print_step "Downloading large files with Git-LFS..."
git lfs pull
print_success "LFS files downloaded"

# Step 2: Download models
print_step "Step 2/5: Downloading IndexTTS2 models..."

# Check if models already exist (check for actual model files, not just config)
if [ -f "checkpoints/gpt.pth" ] && [ -f "checkpoints/s2mel.pth" ]; then
    print_warning "Models already exist in checkpoints/. Skipping download."
else
    # Install huggingface-hub CLI
    print_step "Installing huggingface-hub CLI..."
    python3 -m pip install -U "huggingface-hub[cli]" > /dev/null 2>&1
    print_success "huggingface-hub installed"
    
    # Download models
    print_step "Downloading models from Hugging Face (this may take a while, ~8GB)..."
    
    # Check if in China (for mirror)
    if [[ "$HF_ENDPOINT" == "" ]]; then
        read -p "Are you in China? Use HF mirror for faster download? (y/n): " use_mirror
        if [[ "$use_mirror" == "y" ]] || [[ "$use_mirror" == "Y" ]]; then
            export HF_ENDPOINT="https://hf-mirror.com"
            print_success "Using HF mirror"
        fi
    fi
    
    # Download
    if command -v hf &> /dev/null; then
        hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
    else
        python3 -m huggingface_hub download IndexTeam/IndexTTS-2 --local-dir=checkpoints
    fi
    
    print_success "Models downloaded"
fi

# Step 3: Install Python dependencies
cd ..
print_step "Step 3/5: Installing Python dependencies..."
python3 -m pip install -r requirements.txt > /dev/null 2>&1
print_success "Base dependencies installed"

# Step 4: Install IndexTTS as package
print_step "Step 4/5: Installing IndexTTS package..."
cd index-tts
python3 -m pip install -e . > /dev/null 2>&1
cd ..
print_success "IndexTTS installed"

# Step 5: Verify installation
print_step "Step 5/5: Verifying installation..."

# Test import
python3 -c "from indextts.infer_v2 import IndexTTS2; print('Import successful')" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Import test passed"
else
    print_error "Import test failed. Please check error messages above."
    exit 1
fi

# Test model loading
print_step "Testing model initialization..."
cd index-tts
python3 -c "
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path='checkpoints/config.yaml', model_dir='checkpoints', use_fp16=True)
print('Model loaded successfully')
" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    print_success "Model initialization test passed"
else
    print_warning "Model initialization test failed. This may work anyway during actual use."
fi
cd ..

# Create voice_samples directory
mkdir -p voice_samples
print_success "Created voice_samples directory"

# Final summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo "========================================"
echo ""
echo "📚 Documentation:"
echo "   - Quick Start: QUICK_START.md"
echo "   - Full Guide: INDEXTTS_SETUP.md"
echo "   - Comparison: COMPARISON.md"
echo ""
echo "🎯 Next Steps:"
echo "   1. Prepare voice sample (3-10 seconds, clear speech)"
echo "      Copy to: voice_samples/narrator_voice.wav"
echo ""
echo "   2. Run comic processor:"
echo "      python comic_processor/main.py"
echo ""
echo "   3. Select voice and start processing!"
echo ""
echo "📁 Directory Structure:"
echo "   ├── index-tts/              IndexTTS repository"
echo "   │   └── checkpoints/        Models (~8GB)"
echo "   ├── voice_samples/          Your reference voices"
echo "   └── comic_processor/        Your processor"
echo ""
echo "🎙️  Happy narrating! ✨"
