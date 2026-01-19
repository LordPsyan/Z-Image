# Z-Image GUI Generator - Installation Instructions

## Overview
This guide will help you install all dependencies and set up the Z-Image GUI Generator for AI image generation.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ recommended (16GB+ optimal)
- **Storage**: 10GB+ free space for models and outputs
- **OS**: Windows 10/11 (64-bit)

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **Git**: For downloading models (optional)

## Installation Steps

### Step 1: Install Python
1. Download Python from https://www.python.org/downloads/
2. Run the installer and check "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version
   ```

### Step 2: Install CUDA (GPU Users)
1. Download CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Install CUDA 11.0 or higher
3. Verify installation:
   ```bash
   nvcc --version
   ```

### Step 3: Clone or Download Repository
**Option A: Clone with Git**
```bash
git clone https://github.com/LordPsyan/Z-Image.git
cd Z-Image
```

**Option B: Download ZIP**
1. Download the repository as ZIP from https://github.com/LordPsyan/Z-Image
2. Extract to a folder (e.g., `C:\Z-Image`)
3. Navigate to the folder

### Step 4: Create Virtual Environment (Recommended)
```bash
python -m venv zimage_env
zimage_env\Scripts\activate
```

### Step 5: Install Dependencies
Install the required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation
Test the installation by running the GUI:

```bash
python z_image_gui.py
```

## Dependencies Explained

### Core Dependencies
- **torch**: PyTorch deep learning framework with CUDA support
- **torchvision**: Computer vision library for PyTorch
- **diffusers**: Hugging Face diffusion models library
- **transformers**: Natural language processing models
- **accelerate**: Distributed training and inference optimization

### GUI Dependencies
- **tkinter**: Python's standard GUI toolkit (included with Python)
- **Pillow**: Image processing and display library
- **numpy**: Numerical computing library

### Additional Libraries
- **opencv-python**: Computer vision and image processing
- **matplotlib**: Plotting and visualization (used internally)

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available
**Problem**: Program runs on CPU instead of GPU
**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Model Download Fails
**Problem**: Error downloading models from Hugging Face
**Solution**:
```bash
# Install huggingface_hub with faster downloads
pip install huggingface_hub[hf_xet]

# Or use VPN if network issues persist
```

#### 3. Import Errors
**Problem**: ModuleNotFoundError for dependencies
**Solution**:
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or install individually
pip install torch torchvision transformers diffusers accelerate Pillow opencv-python numpy matplotlib
```

#### 4. GUI Doesn't Start
**Problem**: tkinter not found or display issues
**Solution**:
```bash
# On Windows, tkinter should be included
# If missing, reinstall Python with tkinter support

# Test tkinter:
python -c "import tkinter; tkinter.Tk().mainloop()"
```

#### 5. Memory Errors
**Problem**: Out of memory during generation
**Solution**:
- Use smaller image dimensions (512x512 instead of 1024x1024)
- Close other applications to free RAM
- Use CPU mode if GPU memory is insufficient

#### 6. Permission Errors
**Problem**: Can't save files or create outputs
**Solution**:
- Run as Administrator
- Check folder permissions
- Ensure outputs folder exists and is writable

### Performance Optimization

#### GPU Optimization
1. **Update GPU Drivers**: Install latest NVIDIA drivers
2. **CUDA Memory**: Monitor GPU memory usage
3. **Batch Size**: Reduce image size for memory constraints

#### CPU Optimization
1. **Image Size**: Use smaller dimensions (512x512)
2. **Steps**: Reduce inference steps (2-4 instead of 8+)
3. **Background Apps**: Close unnecessary applications

## Configuration

### Environment Variables (Optional)
Set these for better performance:

```bash
# Windows Command Prompt
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_VISIBLE_DEVICES=0

# Or add to System Environment Variables
```

### Model Cache
Models are cached automatically at:
- Windows: `%USERPROFILE%\.cache\huggingface`
- Custom: Set `HF_HOME` environment variable

## First Run

### Initial Setup
1. **Launch**: Run `python z_image_gui.py`
2. **Load Model**: Click "Load Model" button
3. **Select Resolution**: Choose from dropdown (adjusts automatically if needed)
4. **Enter Prompt**: Type your image description
5. **Generate**: Click "Generate" button
6. **Save**: Choose location and format when saving

### Tips for Best Results
- **Prompts**: Be descriptive and specific
- **Resolution**: Start with 512x512 for testing
- **Steps**: 4 steps for Z-Image Turbo (recommended)
- **Guidance**: 0.0 for Z-Image Turbo (recommended)
- **Seed**: Use same seed for consistent results

## Support

### Getting Help
- **Issues**: Report bugs on GitHub repository at https://github.com/LordPsyan/Z-Image
- **Models**: Check Hugging Face for model updates
- **Community**: Join Discord/Reddit for user discussions

### Model Information
- **Primary Model**: Tongyi-MAI/Z-Image-Turbo
- **Requirements**: CUDA 11.0+, 8GB+ RAM recommended
- **Performance**: ~4 seconds per image on modern GPU

## Uninstallation

### Clean Removal
```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment folder
rmdir /s zimage_env

# Remove program files (optional)
rmdir /s z-image-gui
```

## Advanced Configuration

### Custom Models
To add custom models:
1. Download model files to `models/` folder
2. Update model list in `z_image_gui.py`
3. Restart application

### Network Settings
For corporate networks:
```bash
# Set proxy if needed
set HTTP_PROXY=http://proxy.company.com:8080
set HTTPS_PROXY=http://proxy.company.com:8080

# Or configure in Python script
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
```

---

## Quick Start Summary

1. Install Python 3.8+
2. Install CUDA 11.0+ (GPU users)
3. Download/extract program files
4. Create virtual environment: `python -m venv zimage_env`
5. Activate: `zimage_env\Scripts\activate`
6. Install: `pip install -r requirements.txt`
7. Run: `python z_image_gui.py`

Enjoy generating AI images! 🎨
