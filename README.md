# Z-Image GUI Generator

A graphical user interface for generating high-quality images using the Z-Image AI model family.

## Features

- **Image Generation**: Create high-quality images using Z-Image-Turbo model
- **Batch Processing**: Load multiple prompts from a file and generate images in batch
- **Customizable Parameters**: Control resolution, steps, guidance scale, and seed
- **GPU Acceleration**: Automatic GPU detection and usage when available
- **Easy Export**: Save images as PNG/JPEG with custom file dialog
- **Theme Support**: Toggle between light and dark interface themes
- **Smart Resolution**: Auto-adjusts dimensions to meet model requirements

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)
- See `requirements.txt` for package dependencies

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python z_image_gui.py
   ```

2. **Load Model**: Click "Load Model" to download and initialize the Z-Image model
3. **Enter Prompt**: Type your text prompt in the prompt area
4. **Adjust Parameters**: Set resolution, steps, guidance scale, and seed as needed
5. **Generate**: Click "Generate" to create your image
6. **Save**: Use the "Save" button to export your creation with custom location

### Batch Processing

1. Create a text file with one prompt per line
2. Click "Load Prompts from File" to load the prompts
3. Click "Batch Generate" to process all prompts
4. Images will be saved in the selected directory with timestamps

## Parameters

- **Resolution**: Output image dimensions (default: matches screen resolution, auto-adjusted for model)
- **Steps**: Number of inference steps (default: 4 for Z-Image-Turbo)
- **Guidance Scale**: Prompt adherence (default: 0.0 for turbo models)
- **Seed**: Random seed for reproducible results

## Models

- **Z-Image-Turbo**: Fast 6B parameter model, optimized for quick generation

## Tips

- Use detailed prompts for better results
- Z-Image-Turbo works best with low guidance scale (0.0-1.0)
- Enable GPU acceleration for much faster generation
- Use the same seed to generate similar images across sessions
- Dimensions are automatically adjusted to be divisible by 16 for model compatibility

## Troubleshooting

- **Out of Memory**: Try reducing image dimensions or using CPU mode
- **Slow Generation**: Ensure CUDA is properly installed and GPU is detected
- **Model Loading Issues**: Check internet connection and Hugging Face access

## License

This project is provided as-is for educational and personal use. Please respect the licenses of the underlying Z-Image models and dependencies.
