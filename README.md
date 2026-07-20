This is a program which uses either .kfb or .rfm files generated from Kalles Fraktaler or from RFF

# How to use/setup this program

## Installing Python

To setup ths program you will first need to install Python
You will need at least Python 3.10

For Windows you can find Python on the Microsoft store
For MacOS you can install Python from Homebrew
For Linux you can install Python from your distro's package manager

## Installing libraries

The libraries needed can be installed with a simple command:
```
pip install numpy numba opencv-python pygame mpmath librosa scipy
```
# Installing FFmpeg
For Windows you can grap ffmpeg from [ffmpeg.org](here)
For MacOS you can grab it from homebrew
For Linux you can grab it from your distro's package manager

## Running the program

To run the program simply run:
```
python3 main.py path/to/kfb/files output.mp4
```
# If something goes wrong

## Python not found
Try:

python3

---

## Missing module
Run:

pip install numpy numba opencv-python mpmath pygame sci[y

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
