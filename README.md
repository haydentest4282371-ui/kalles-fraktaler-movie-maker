Kalles Fraktaler Movie Maker is a tool for making zoom movies with Kalles Fraktaler.
I commonly use this tool to make zooms and wanted to publish it to allow community contribution and let other people have fun with it.

## Features

* Render zoom sequences from KFB files
* Animated palette flow effects
* Directional lighting based on iteration gradients
* Smooth interpolation between keyframes
* MP4 video output
* Multi-threaded rendering using Numba JIT
* GPU Rendering via Numba CUDA

## Requirements

* Python 3.10+
* NumPy
* Numba
* OpenCV (cv2)
* FFmpeg in PATH (For windows you can grab it from [FFmpeg](https://ffmpeg.org), and for mac/linux you can use homebrew or your package manager)

Install dependencies:

```bash
pip install numpy numba opencv-python
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

## Output

The renderer generates an MP4 video containing the complete zoom animation.

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

* Rendering speed depends heavily on image resolution, sequence length, and how powerful your PC is.
* The renderer is optimized for GPU execution using Numba.
* It also has a cpu.py which is a version of render.py rewritten to use CPU instead of GPU for compatibility.
## License

See the repository license file for details.
