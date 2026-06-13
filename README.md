# Kalles Fraktaler Movie Maker

This program turns a folder of fractal zoom files (.kfb files) into a video (MP4).

It does NOT create fractals.

You first create the fractal files using another program, then this program turns them into a video.

---

# What you are doing (simple explanation)

You will:

1. Make fractal zoom files using Kalles Fraktaler
2. Put those files in a folder
3. Run this program once
4. Get a video file

---

# What you need

You need:

- Python (runs this program)
- FFmpeg (creates the video)
- A folder of .kfb files (your fractal frames)

---

# Step 1: Install Python

1. Go to:
   https://www.python.org/downloads/

2. Download and install Python

3. IMPORTANT:
   Check this box:
   ☑ Add Python to PATH

---

# Step 2: Install FFmpeg

## Windows

1. Go to:
   https://www.gyan.dev/ffmpeg/builds/

2. Download:
   “ffmpeg-git-full.7z”

3. Extract it

4. Move it to:
   C:\ffmpeg

5. Make sure this file exists:
   C:\ffmpeg\bin\ffmpeg.exe

6. Add to PATH:
   - Search “environment variables”
   - Open “Edit the system environment variables”
   - Click “Environment Variables”
   - Find “Path”
   - Click Edit → New
   - Add:
     C:\ffmpeg\bin

7. Test it:

   ffmpeg -version

If it prints text, it works.

---

## Linux

sudo apt install ffmpeg

---

## macOS

brew install ffmpeg

---

# Step 3: Install Python libraries

Open a terminal in the project folder:

pip install numpy numba opencv-python mpmath

---

# Step 4: Create .kfb files

You must create these using Kalles Fraktaler.

Watch this tutorial:

https://www.youtube.com/watch?v=UQz8azo5MWU

IMPORTANT:

Only follow the tutorial until you have the .kfb files.

STOP before any video rendering steps.

This program replaces that part.

You should end with files like:

frame_001.kfb
frame_002.kfb
frame_003.kfb
...

---

# Step 5: Run the program

Run:

python main.py <kfb_folder> <output_video>

---

# Example

python main.py zooms output.mp4

---

# What the command means

- <kfb_folder> = folder containing .kfb files
- output.mp4 = video file that will be created

---

# When it works

You will get:

output.mp4

---

# If something goes wrong

## Python not found
Try:

python3

---

## Missing module
Run:

pip install numpy numba opencv-python mpmath

---

## FFmpeg not found
FFmpeg is not installed correctly or not in PATH.

Go back to Step 2.

---

# If you still have problems

If you get an error that is not listed above:

Please open an issue on GitHub

Include:
- The full error message
- What command you ran
- Your operating system (Windows / Linux / macOS)

Some issues may be caused by bugs in the program, not user setup.
