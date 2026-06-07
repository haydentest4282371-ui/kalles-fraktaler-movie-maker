import sys
import render

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py /absolute/path/to/kfb_folder [out_dir]")
        sys.exit(1)

    folder = sys.argv[1]
    out = sys.argv[2]

    # safety: enforce absolute path so you don’t accidentally pass junk

    render.render_sequence(folder, out)


if __name__ == "__main__":
    main()