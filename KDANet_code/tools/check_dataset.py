import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Check ImageFolder-style dataset layout.")
    parser.add_argument("root", help="Dataset root directory.")
    return parser.parse_args()


def count_images(path: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for file in path.rglob("*") if file.suffix.lower() in suffixes)


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"{split}: missing")
            continue
        print(f"{split}:")
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            print(f"  {class_dir.name}: {count_images(class_dir)}")


if __name__ == "__main__":
    main()
