# Kalles Fraktaler Movie Maker

This program turns a folder of fractal zoom files (.kfb files) into a video (MP4).

It does NOT create fractals.

You first create the fractal files using another program, then this program turns them into a video.

---

# What you are doing (simple explanation)

You will:

1. Make fractal zoom files using a tool called Kalles Fraktaler
2. Put those files in a folder
3. Run this program once
4. Get a video file

---

# What you need

You need:

- Python (a program that runs this tool)
- FFmpeg (a video tool)
- A folder of .kfb files (your fractal images)

---

# Step 1: Install Python

1. Go to this website:
   https://www.python.org/downloads/

2. Click the big download button

3. Open the file you downloaded

4. VERY IMPORTANT:
   Check this box:
   ☑ Add Python to PATH

5. Click Install

---

# Step 2: Install FFmpeg (Windows only needs attention here)

## Windows

1. Go here:
   https://www.gyan.dev/ffmpeg/builds/

2. Download:
   “ffmpeg-git-full.7z”

3. Open it and extract it (like a ZIP file)

4. Move the folder to:
   C:\ffmpeg

5. Inside it, you should see:
   C:\ffmpeg\bin\ffmpeg.exe

6. Add it to PATH:
   - Press Windows key
   - Type: environment variables
   - Click: Edit the system environment variables
   - Click: Environment Variables
   - Click Path → Edit
   - Click New
   - Add this:
     C:\ffmpeg\bin
   - Click OK

7. Test it:
   Open Command Prompt and type:

   ffmpeg -version

If you see text, it worked.

---

## Linux

Run:

sudo apt install ffmpeg

---

## macOS

Run:

brew install ffmpeg

---

# Step 3: Install required Python tools

Open a terminal inside the project folder and run:

pip install numpy numba opencv-python mpmath

---

# Step 4: Create your fractal files (.kfb)

You must create these using Kalles Fraktaler.

Follow this video:

https://www.youtube.com/watch?v=UQz8azo5MWU

IMPORTANT:

Watch ONLY until you have your .kfb files.

Stop before any video rendering steps in the video.

This program replaces that step.

At the end, you should have a folder like:

frame_001.kfb
frame_002.kfb
frame_003.kfb
...

---

# Step 5: Run the program

Open a terminal in the project folder and run:

python main.py <kfb_folder> <output_video>

---

# Example

python main.py zooms output.mp4

---

# What the command means

- <kfb_folder> = folder that has your .kfb files
- output.mp4 = the video file that will be created

---

# When it is done

You will get a video file called:

output.mp4

You can open it like any normal video.

---

# If something goes wrong

## It says “python not found”

Try:

python3

---

## It says missing module

Run:

pip install numpy numba opencv-python mpmath

---

## It says ffmpeg not found

FFmpeg is not installed correctly or not added to PATH.

Go back to Step 2.

---

# Final note

If you follow the steps exactly, it will work.

If something breaks, it is almost always:
- Python not installed correctly
- FFmpeg not installed correctly
- Wrong folder given to the program
