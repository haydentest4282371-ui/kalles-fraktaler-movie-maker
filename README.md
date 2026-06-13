# Kalles Fraktaler Movie Maker

Kalles Fraktaler Movie Maker is a tool that turns fractal zoom keyframe files (KFB files) into smooth zoom videos (MP4).

You create zoom frames using Kalles Fraktaler, then this program turns them into a finished animation.

---

## What this program does

- Turns KFB keyframe folders into zoom videos
- Creates smooth transitions between frames
- Supports color palette animation effects
- Exports MP4 video files
- Can use GPU acceleration (if available)

---

## Requirements

Before starting, install:

- Python 3.10 or newer → https://www.python.org/downloads/
- FFmpeg → https://ffmpeg.org/download.html

You do NOT need to manually install Python libraries — the setup step handles it.

---

## Installation (Windows)

### 1. Install Python

1. Download Python from:
   https://www.python.org/downloads/

2. Run the installer

3. IMPORTANT: Check:
   [x] Add Python to PATH

4. Click Install

---

### 2. Download the project

1. Download or clone this repository
2. Extract the ZIP file
3. Open the extracted folder

---

### 3. Install dependencies

Open Command Prompt inside the project folder:

- Click the folder path bar
- Type `cmd`
- Press Enter

Then run:

py -m pip install numpy numba opencv-python mpmath

---

## How to run

Run this inside the project folder:

py main.py <kfb_folder> <output_video>

---

### Example:

py main.py C:\Users\You\fractals\zoom_sequence output.mp4

---

## Input format

The input folder must contain a sequence of .kfb files.

They should be ordered in zoom progression:
frame 1 → deeper zoom → deeper zoom → etc.

---

## Output

The program generates an MP4 file:

output.mp4

You can open it with any video player.

---

## Settings

Edit configuration in:

config.py

Options include:
- Color palette
- Animation speed
- Frame smoothing
- Render quality

---

## Common problems

### "No module named mpmath"

Run:

py -m pip install mpmath

If it still fails, you are installing into a different Python version.

---

### "python not recognized"

Use:

py

instead of python or python3.

---

### FFmpeg not found

Install FFmpeg and add it to PATH:
https://ffmpeg.org/download.html

---

## Notes

- Higher resolution = slower rendering
- GPU acceleration requires CUDA
- CPU fallback works automatically if GPU is not available
