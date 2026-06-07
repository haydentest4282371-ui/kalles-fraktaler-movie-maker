import json
import pyautogui
import keyboard

palette = []

print("J = sample color")
print("R = remove last color")
print("E = save and exit")


def sample():
    x, y = pyautogui.position()
    r, g, b = pyautogui.pixel(x, y)

    palette.append([r, g, b])
    print(f"Added {[r,g,b]} (total {len(palette)})")


def remove_last():
    if palette:
        removed = palette.pop()
        print(f"Removed {removed} (total {len(palette)})")
    else:
        print("Palette is already empty")


def save_and_exit():
    with open("palette.json", "w") as f:
        json.dump(palette, f, indent=4)

    print(f"Saved {len(palette)} colors to palette.json")
    exit()


keyboard.add_hotkey("j", sample)
keyboard.add_hotkey("r", remove_last)
keyboard.add_hotkey("e", save_and_exit)

keyboard.wait()