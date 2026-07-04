# -*- coding: utf-8 -*-
"""
Pure numpy/scipy implementation of Dice (DSC), HD95 and bootstrap CI95.
No dependency on monai.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coeff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient between two binary masks (same shape)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty → perfect match
    return (2.0 * intersection) / denom


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Extract surface (boundary) points of a binary mask.

    Surface = foreground pixels that have at least one background
    neighbour (4-connectivity).  Returns an (N, 2) array of (y, x) coords.
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.empty((0, 2), dtype=np.float64)

    # erode by 1 pixel (4-connectivity) → boundary = foreground − eroded
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask, iterations=1)
    boundary = mask & ~eroded
    coords = np.argwhere(boundary)
    return coords.astype(np.float64)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff Distance (in pixels).

    Uses the symmetric surface-distance approach:
        d_surface = max(
            percentile_95(d(pred_surf → gt_surf)),
            percentile_95(d(gt_surf  → pred_surf))
        )
    where d(A→B) is, for each point in A, the Euclidean distance to the
    nearest point in B.

    Boundary cases:
        - pred & gt both non-empty: normal calculation
        - pred non-empty, gt empty (false positive): return 0.0
        - pred empty, gt non-empty (false negative): return 0.0
        - pred & gt both empty (true negative):     return 0.0
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        # Any side empty → no surface distance to compute → 0.0
        return 0.0

    # Surface points
    surf_pred = _surface_points(pred)
    surf_gt = _surface_points(gt)

    if surf_pred.shape[0] == 0 or surf_gt.shape[0] == 0:
        # Should not happen after the checks above, but keep as safeguard
        return 0.0

    # For each point in surf_pred, min distance to surf_gt  (d_pred_to_gt)
    # For each point in surf_gt,  min distance to surf_pred (d_gt_to_pred)
    #
    # Use scipy cKDTree for efficiency.
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(surf_gt)
    tree_pred = cKDTree(surf_pred)

    d_pred_to_gt, _ = tree_gt.query(surf_pred)
    d_gt_to_pred, _ = tree_pred.query(surf_gt)

    # Symmetric: take max of the two 95th-percentiles
    # (consistent with the non-directed HD95 definition)
    hd95_pred = np.percentile(d_pred_to_gt, 95)
    hd95_gt = np.percentile(d_gt_to_pred, 95)

    return max(hd95_pred, hd95_gt)


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    """Bootstrap confidence interval for the mean.

    Returns (mean, lower, upper).
    NaNs are removed before bootstrapping.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2
    mean = values.mean()
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return mean, lower, upper
