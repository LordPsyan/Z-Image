# Z-Image GUI Generator

A graphical user interface for generating images and videos using the Z-Image AI model family.

## Features

- **Image Generation**: Create high-quality images using Z-Image-Turbo and other Z-Image models
- **Video Generation**: Generate animated videos by creating sequences of images
- **Batch Processing**: Load multiple prompts from a file and generate images in batch
- **Customizable Parameters**: Control width, height, steps, guidance scale, and seed
- **GPU Acceleration**: Automatic GPU detection and usage when available
- **Easy Export**: Save images as PNG/JPEG and videos as MP4

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
4. **Adjust Parameters**: Set width, height, steps, guidance scale, and seed as needed
5. **Generate**: Click "Generate" to create your image or video
6. **Save**: Use the "Save" button to export your creation

### Batch Processing

1. Create a text file with one prompt per line
2. Click "Load Prompts from File" to load the prompts
3. Click "Batch Generate" to process all prompts
4. Images will be saved in the `outputs` directory

## Parameters

- **Width/Height**: Output image dimensions (default: 512x512)
- **Steps**: Number of inference steps (default: 4 for Z-Image-Turbo)
- **Guidance Scale**: Prompt adherence (default: 0.0 for turbo models)
- **Seed**: Random seed for reproducible results
- **Frames**: Number of frames for video generation
- **FPS**: Frames per second for video output

## Models

- **Z-Image-Turbo**: Fast 6B parameter model, great for quick generation
- **Z-Image**: Base model with higher quality
- **Z-Image-Edit**: Model for image editing tasks

## Tips

- Use detailed prompts for better results
- Z-Image-Turbo works best with low guidance scale (0.0-1.0)
- For video generation, use prompts that describe motion or change
- Enable GPU acceleration for much faster generation
- Use the same seed to generate similar images across sessions

## Troubleshooting

- **Out of Memory**: Try reducing image dimensions or using CPU mode
- **Slow Generation**: Ensure CUDA is properly installed and GPU is detected
- **Model Loading Issues**: Check internet connection and Hugging Face access

## License

This project is provided as-is for educational and personal use. Please respect the licenses of the underlying Z-Image models and dependencies.
