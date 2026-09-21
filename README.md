This is a program which uses either .kfb or .rfm files generated from Kalles Fraktaler or from RFF.

# How to use/setup this program

## 1. Installing Python

To setup ths program you will first need to install Python.
You will need at least Python 3.10.

For Windows you can find Python on the Microsoft store.
For MacOS you can install Python from Homebrew.
For Linux you can install Python from your distro's package manager.

## 2. Seting up a virtual environment

Run these 2 commands to create a virtual environment

```bash
python3 -m venv <venv_path>
source .venv/bin/activate
```
To exit the virtual environment run `deactivate` and to enter the virtual environment run


## 3. Installing libraries

The libraries needed can be installed with a simple command:

`pip install "numpy<2.5" numba opencv-python mpmath librosa scipy PySide6 numba-cuda[cu13]`

## 4. Installing FFmpeg
For Windows you can grab ffmpeg from [ffmpeg.org](here), make sure it's in the same directory as main.py.
For MacOS you can grab it from homebrew.
For Linux you can grab it from your distro's package manager.

## 5. Running the program

To run the program simply run:
```
python3 main.py path/to/kfb/files output.mp4
```
Make sure you are in the virtual environment you created earlier
If you get an error about no physical device found, either you do not have an NVIDIA GPU or you do not have nvidia drivers setup properly

It should open a GUI to edit config options and a preview

# If something goes wrong

## Python not found
Try:

python3

---

## Missing module
Run:

`pip install "numpy>2.5" numba opencv-python mpmath librosa scipy PySide6 numba-cuda[cu13]`

---

## FFmpeg not found
FFmpeg is not installed correctly or not in PATH.

Go back to Step 3.

---

## If you still have problems

If you get an error that is not listed above:

Please open an issue on GitHub

Include:
- The full error message
- What command you ran
- Your operating system (Windows / Linux / macOS)

Some issues may be caused by bugs in the program, not user setup.
