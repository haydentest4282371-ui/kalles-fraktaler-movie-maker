# Kalles Fraktaler Movie Maker

Kalles Fraktaler Movie Maker is a tool for creating zoom movies from Kalles Fraktaler keyframe files.

I use this tool for my own zoom renders and decided to publish it so others can use it, contribute improvements, and experiment with their own fractal animations.

## Features

* Render zoom sequences from KFB files
* Animated palette flow effects
* Directional lighting based on iteration gradients
* Smooth interpolation between keyframes
* MP4 video output
* Multi-threaded rendering using Numba JIT
* GPU rendering using Numba CUDA

## Requirements

* Python 3.10 or newer
* NumPy
* Numba
* OpenCV (`cv2`)
* FFmpeg available from the command line

For Windows, FFmpeg can be downloaded from https://ffmpeg.org.

For macOS and Linux, FFmpeg can usually be installed using Homebrew or your system package manager.

## Setup

### Step 1: Install Python

Python is required to run this program.

**Windows**

1. Go to https://www.python.org/downloads/
2. Download Python 3.10 or newer.
3. Run the installer.
4. Check **"Add Python to PATH"**.
5. Click **Install**.

**macOS / Linux**

Install Python 3.10 or newer using Homebrew or your system package manager.

### Verify Python Installation

Open Command Prompt (Windows) or Terminal (macOS/Linux) and run:

```bash
python --version
```

You should see a version number of 3.10 or higher.

### Step 2: Download the Program

1. Download this repository as a ZIP file.
2. Extract the ZIP file.
3. Open the extracted folder.

### Step 3: Install Dependencies

Open Command Prompt (Windows) or Terminal (macOS/Linux) inside the project folder and run:

```bash
pip install numpy numba opencv-python mpmath
```

## Usage

```bash
python3 main.py <kfb_folder> <output_video>
```

Example:

```bash
python3 main.py /home/user/fractals/zoom_sequence output.mp4
```

The input folder should contain a sequence of KFB files ordered by zoom progression.

You will also need Kalles Fraktaler to create the KFB files. You can follow this tutorial and stop once you have exported your KFB sequence:

https://www.youtube.com/watch?v=UQz8azo5MWU

## Output

The renderer generates an MP4 video containing the completed zoom animation.

Example:

```bash
python3 main.py zooms/mandelbrot mandelbrot.mp4
```

## Configuration

Rendering options can be modified in `config.py`.

Examples include:

* Palette selection
* Palette flow speed
* Fade-to-black behavior
* Frame interpolation settings

## Notes

* Rendering speed depends heavily on image resolution, sequence length, and hardware performance.
* The renderer is optimized for GPU execution using Numba CUDA.
* A CPU implementation is also included in `cpu.py` for systems without CUDA support.

## License

See the repository license file for details.
