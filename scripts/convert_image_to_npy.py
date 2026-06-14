#!/usr/bin/env python3
"""Convert images and masks to .npy files.

Single-file / single-directory mode:
  python scripts/convert_image_to_npy.py --input /path/to/image_or_dir --output /path/to/save

Paired image+mask directory mode for MedSegX:
  python scripts/convert_image_to_npy.py \
      --images /path/to/images \
      --masks /path/to/masks \
      --output_root /path/to/inference

This creates:
  /path/to/inference/npy_imgs/
  /path/to/inference/npy_gts/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_as_numpy(path: Path, is_mask: bool = False) -> np.ndarray:
    """Load an image or mask as a NumPy array."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        img = Image.open(path)
        if is_mask:
            return np.array(img.convert("L"))
        if img.mode in {"1", "L", "I;16", "I", "F"}:
            return np.array(img.convert("RGB"))
        return np.array(img.convert("RGB"))

    if suffix == ".npy":
        return np.load(path)

    raise ValueError(f"Unsupported file type: {path}")


def save_npy(input_path: Path, output_path: Path, is_mask: bool = False) -> None:
    arr = load_as_numpy(input_path, is_mask=is_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)


def iter_input_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return

    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS | {".npy"}:
                yield p
        return

    raise FileNotFoundError(f"Input path not found: {input_path}")


def build_stem_map(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in iter_input_files(folder):
        stem = path.stem
        if stem in files:
            raise ValueError(f"Duplicate stem found in {folder}: {stem}")
        files[stem] = path
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert images and masks to .npy files.")

    parser.add_argument("--input", help="Input image file or directory for single mode.")
    parser.add_argument(
        "--output",
        help="Output .npy file path or output directory for single mode.",
    )

    parser.add_argument("--images", help="Image directory for paired mode.")
    parser.add_argument("--masks", help="Mask directory for paired mode.")
    parser.add_argument(
        "--output_root",
        help="Output root directory for paired mode; npy_imgs/ and npy_gts/ will be created here.",
    )
    return parser.parse_args()


def run_single_mode(input_path: Path, output_path: Path) -> None:
    if input_path.is_file():
        if output_path.suffix.lower() != ".npy":
            output_path = output_path.with_suffix(".npy")
        save_npy(input_path, output_path)
        print(f"saved: {output_path}")
        return

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for file_path in iter_input_files(input_path):
            out_file = output_path / f"{file_path.stem}.npy"
            save_npy(file_path, out_file)
            print(f"saved: {out_file}")
        return

    raise FileNotFoundError(f"Input path not found: {input_path}")


def run_paired_mode(images_dir: Path, masks_dir: Path, output_root: Path) -> None:
    img_map = build_stem_map(images_dir)
    mask_map = build_stem_map(masks_dir)

    img_stems = set(img_map)
    mask_stems = set(mask_map)
    missing_masks = sorted(img_stems - mask_stems)
    missing_images = sorted(mask_stems - img_stems)

    if missing_masks:
        raise ValueError(f"Missing masks for: {', '.join(missing_masks[:10])}")
    if missing_images:
        raise ValueError(f"Missing images for: {', '.join(missing_images[:10])}")

    npy_img_dir = output_root / "npy_imgs"
    npy_gt_dir = output_root / "npy_gts"
    npy_img_dir.mkdir(parents=True, exist_ok=True)
    npy_gt_dir.mkdir(parents=True, exist_ok=True)

    for stem in sorted(img_stems):
        img_path = img_map[stem]
        mask_path = mask_map[stem]
        img_out = npy_img_dir / f"{stem}.npy"
        mask_out = npy_gt_dir / f"{stem}.npy"
        save_npy(img_path, img_out, is_mask=False)
        save_npy(mask_path, mask_out, is_mask=True)
        print(f"saved: {img_out}")
        print(f"saved: {mask_out}")


def main() -> None:
    args = parse_args()

    paired_mode = args.images or args.masks or args.output_root
    if paired_mode:
        if not (args.images and args.masks and args.output_root):
            raise SystemExit("paired mode requires --images, --masks, and --output_root")
        run_paired_mode(Path(args.images), Path(args.masks), Path(args.output_root))
        return

    if not (args.input and args.output):
        raise SystemExit("single mode requires --input and --output")
    run_single_mode(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
