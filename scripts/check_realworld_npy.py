#!/usr/bin/env python3
"""Check converted FinalData/Augtrain NPY pairs.

Examples:
  python scripts/check_realworld_npy.py
  python scripts/check_realworld_npy.py --datasets FinalData Augtrain --max_bad 20
  python scripts/check_realworld_npy.py --root ./playground --datasets FinalData Augtrain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = ["FinalData", "Augtrain"]
DEFAULT_TASK = "US_ThyroidNodule"
DEFAULT_SHIFT = "cross_site"


def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def resolve_inference_dir(root: Path, dataset: str, shift: str, task: str) -> Path:
    return root / dataset / "eval" / "RealWorld" / shift / task / dataset / "inference"


def format_shape(arr: np.ndarray) -> str:
    return "x".join(str(x) for x in arr.shape)


def summarize_image(arr: np.ndarray) -> tuple[str, list[str]]:
    issues: list[str] = []
    if arr.ndim not in (2, 3):
        issues.append(f"image ndim={arr.ndim}")
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
            pass
        elif arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            issues.append("image looks CHW, expected HWC")
        else:
            issues.append("image channels are ambiguous")

    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    summary = f"shape={format_shape(arr)} dtype={arr.dtype} min={arr_min:g} max={arr_max:g}"
    return summary, issues


def summarize_mask(arr: np.ndarray) -> tuple[str, list[str]]:
    issues: list[str] = []
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        issues.append(f"mask ndim={arr.ndim}")

    uniq = np.unique(arr)
    uniq_preview = uniq[:10]
    uniq_text = ", ".join(str(int(x)) if np.issubdtype(uniq.dtype, np.integer) else f"{x:g}" for x in uniq_preview)
    if uniq.size > 10:
        uniq_text += ", ..."

    is_binary = np.all(np.isin(uniq, [0, 1, 255]))
    if not is_binary:
        issues.append("mask has non-binary values")

    fg = arr > 0
    fg_ratio = float(fg.mean()) if fg.size else 0.0
    if fg_ratio == 0:
        issues.append("empty mask")

    summary = f"shape={format_shape(arr)} dtype={arr.dtype} uniq=[{uniq_text}] fg_ratio={fg_ratio:.4f}"
    return summary, issues


def bbox_from_mask(mask: np.ndarray) -> str:
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return "empty"
    y_min, x_min = fg.min(axis=0)
    y_max, x_max = fg.max(axis=0)
    return f"[{x_min},{y_min}] - [{x_max},{y_max}]"


def inspect_dataset(inference_dir: Path, max_bad: int) -> None:
    img_dir = inference_dir / "npy_imgs"
    gt_dir = inference_dir / "npy_gts"

    print(f"\n== {inference_dir} ==")
    if not img_dir.is_dir() or not gt_dir.is_dir():
        print("missing npy_imgs or npy_gts")
        return

    img_files = {p.stem: p for p in sorted(img_dir.glob("*.npy"))}
    gt_files = {p.stem: p for p in sorted(gt_dir.glob("*.npy"))}
    stems = sorted(set(img_files) & set(gt_files))
    missing_imgs = sorted(set(gt_files) - set(img_files))
    missing_gts = sorted(set(img_files) - set(gt_files))

    print(f"pairs={len(stems)} imgs={len(img_files)} gts={len(gt_files)}")
    if missing_imgs:
        print(f"missing images: {len(missing_imgs)}")
    if missing_gts:
        print(f"missing masks: {len(missing_gts)}")

    bad_samples: list[str] = []
    img_shapes = {}
    gt_shapes = {}
    mask_fg_ratios = []

    for stem in stems:
        img = load_npy(img_files[stem])
        gt = load_npy(gt_files[stem])

        img_shapes[img.shape] = img_shapes.get(img.shape, 0) + 1
        gt_shapes[gt.shape] = gt_shapes.get(gt.shape, 0) + 1
        mask_fg_ratios.append(float((gt > 0).mean()) if gt.size else 0.0)

        img_summary, img_issues = summarize_image(img)
        gt_summary, gt_issues = summarize_mask(gt)

        shape_issue = []
        if img.ndim == 2 and gt.ndim == 2 and img.shape != gt.shape:
            shape_issue.append("image/mask shape mismatch")
        elif img.ndim == 3 and gt.ndim == 2:
            if img.shape[:2] != gt.shape and img.shape[-2:] != gt.shape:
                shape_issue.append("image/mask shape mismatch")

        issues = img_issues + gt_issues + shape_issue
        if issues:
            bad_samples.append(
                f"{stem}: {', '.join(sorted(set(issues)))} | img({img_summary}) | gt({gt_summary}) | bbox={bbox_from_mask(gt)}"
            )

    print(f"unique image shapes: {len(img_shapes)}")
    for shape, count in sorted(img_shapes.items(), key=lambda x: (-x[1], x[0]))[:5]:
        print(f"  img {shape}: {count}")
    print(f"unique mask shapes: {len(gt_shapes)}")
    for shape, count in sorted(gt_shapes.items(), key=lambda x: (-x[1], x[0]))[:5]:
        print(f"  gt  {shape}: {count}")

    if mask_fg_ratios:
        ratios = np.asarray(mask_fg_ratios, dtype=float)
        print(
            f"fg_ratio mean={ratios.mean():.4f} min={ratios.min():.4f} max={ratios.max():.4f} "
            f"p10={np.percentile(ratios, 10):.4f} p90={np.percentile(ratios, 90):.4f}"
        )

    if bad_samples:
        print(f"bad samples: {len(bad_samples)}")
        for line in bad_samples[:max_bad]:
            print(f"  - {line}")
    else:
        print("no obvious issues found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FinalData/Augtrain converted npy results")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./playground"),
        help="playground root that contains dataset folders",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--shift", default=DEFAULT_SHIFT)
    parser.add_argument("--max_bad", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        inference_dir = resolve_inference_dir(args.root, dataset, args.shift, args.task)
        inspect_dataset(inference_dir, args.max_bad)


if __name__ == "__main__":
    main()
