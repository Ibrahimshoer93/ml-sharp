"""Depth alignment utilities for multi-frame consistency.

Provides functions for detecting static regions, robust affine fitting,
and aligning depth maps across video frames for 4D Gaussian Splat generation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

# RAFT input size must be divisible by 8
_RAFT_SIZE = (520, 960)


@torch.no_grad()
def compute_optical_flow_mask(
    frames: list[np.ndarray],
    device: torch.device | str = "cpu",
    flow_magnitude_threshold: float = 1.0,
    max_pairs: int | None = None,
) -> np.ndarray:
    """Compute a static mask using RAFT optical flow from torchvision.

    Pixels whose median flow magnitude across consecutive pairs stays below
    ``flow_magnitude_threshold`` are considered static.

    Args:
        frames: List of HxWx3 uint8 RGB frames.
        device: Torch device for RAFT inference.
        flow_magnitude_threshold: Maximum median flow magnitude (pixels at RAFT
            resolution) for a pixel to count as static.
        max_pairs: If set, subsample to at most this many consecutive pairs
            (evenly spaced) to save time on long sequences.

    Returns:
        Boolean mask HxW where True = static pixel.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights).eval().to(device)
    transforms = weights.transforms()

    n = len(frames)
    h_orig, w_orig = frames[0].shape[:2]

    # Build list of consecutive pair indices, optionally subsampled
    pair_indices = list(range(n - 1))
    if max_pairs is not None and len(pair_indices) > max_pairs:
        step = max(1, len(pair_indices) // max_pairs)
        pair_indices = pair_indices[::step][:max_pairs]

    flow_magnitudes: list[np.ndarray] = []

    for idx in pair_indices:
        img1 = torch.from_numpy(frames[idx].copy()).permute(2, 0, 1).float()
        img2 = torch.from_numpy(frames[idx + 1].copy()).permute(2, 0, 1).float()

        # Resize to RAFT-friendly resolution
        img1 = F.interpolate(img1[None], size=_RAFT_SIZE, mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2[None], size=_RAFT_SIZE, mode="bilinear", align_corners=False)

        img1_t, img2_t = transforms(img1.to(device), img2.to(device))

        # RAFT returns list of flow predictions (coarse to fine); take the last
        flow_list = raft(img1_t, img2_t)
        flow = flow_list[-1]  # [1, 2, H_raft, W_raft]

        mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)  # [1, H_raft, W_raft]

        # Resize magnitude back to original frame resolution
        mag_orig = F.interpolate(mag[:, None], size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        flow_magnitudes.append(mag_orig[0, 0].cpu().numpy())

    # Median flow magnitude per pixel across all pairs
    mag_stack = np.stack(flow_magnitudes, axis=0)  # [num_pairs, H, W]
    median_mag = np.median(mag_stack, axis=0)  # [H, W]

    static_mask = median_mag < flow_magnitude_threshold

    LOGGER.info(
        "Optical flow static mask: %d / %d pixels (%.1f%%), threshold=%.2f px",
        static_mask.sum(),
        static_mask.size,
        100 * static_mask.sum() / static_mask.size,
        flow_magnitude_threshold,
    )
    return static_mask


def compute_static_mask(
    depths: list[np.ndarray],
    frames: list[np.ndarray] | None = None,
    depth_var_percentile: float = 30.0,
    rgb_diff_threshold: float = 15.0,
    use_optical_flow: bool = False,
    flow_device: torch.device | str = "cpu",
    flow_magnitude_threshold: float = 1.0,
) -> np.ndarray:
    """Identify static pixels using temporal depth variance and optional RGB differencing.

    When ``use_optical_flow=True`` the RGB differencing heuristic is replaced
    by RAFT optical-flow based motion detection, which gives a much cleaner
    static/dynamic separation, especially with subtle camera motion.

    Args:
        depths: List of HxW depth maps.
        frames: Optional list of HxWx3 uint8 RGB frames.
        depth_var_percentile: Percentile threshold for depth variance (lower = stricter).
        rgb_diff_threshold: Max RGB L1 difference to consider a pixel static (ignored when
            ``use_optical_flow`` is True).
        use_optical_flow: If True, use RAFT optical flow instead of RGB differencing.
        flow_device: Device for RAFT inference (only used when ``use_optical_flow`` is True).
        flow_magnitude_threshold: Max median flow magnitude in pixels for a pixel
            to be considered static.

    Returns:
        Boolean mask HxW where True = static pixel.
    """
    depth_stack = np.stack(depths, axis=0)  # [N, H, W]

    # Normalize each depth to [0,1] range before computing variance
    # (accounts for per-frame scale differences in raw monocular depth)
    depth_medians = np.median(depth_stack, axis=(1, 2), keepdims=True)
    depth_normalized = depth_stack / (depth_medians + 1e-6)

    depth_variance = np.var(depth_normalized, axis=0)  # [H, W]
    variance_threshold = np.percentile(depth_variance[depth_variance > 0], depth_var_percentile)
    depth_static_mask = depth_variance < variance_threshold

    LOGGER.info(
        "Depth variance threshold: %.6f, static pixels from depth: %d / %d (%.1f%%)",
        variance_threshold,
        depth_static_mask.sum(),
        depth_static_mask.size,
        100 * depth_static_mask.sum() / depth_static_mask.size,
    )

    if use_optical_flow and frames is not None and len(frames) > 1:
        LOGGER.info("Computing optical-flow based static mask with RAFT...")
        flow_static_mask = compute_optical_flow_mask(
            frames,
            device=flow_device,
            flow_magnitude_threshold=flow_magnitude_threshold,
        )
        combined_mask = depth_static_mask & flow_static_mask
        LOGGER.info(
            "OpticalFlow+Depth combined static pixels: %d / %d (%.1f%%)",
            combined_mask.sum(),
            combined_mask.size,
            100 * combined_mask.sum() / combined_mask.size,
        )
        return combined_mask

    if frames is not None and len(frames) > 1:
        # Use median frame as reference for RGB differencing
        frame_stack = np.stack(frames, axis=0).astype(np.float32)  # [N, H, W, 3]
        median_frame = np.median(frame_stack, axis=0)  # [H, W, 3]
        max_diff = np.max(np.abs(frame_stack - median_frame[None]), axis=(0, 3))  # [H, W]
        rgb_static_mask = max_diff < rgb_diff_threshold

        combined_mask = depth_static_mask & rgb_static_mask
        LOGGER.info(
            "RGB+Depth combined static pixels: %d / %d (%.1f%%)",
            combined_mask.sum(),
            combined_mask.size,
            100 * combined_mask.sum() / combined_mask.size,
        )
        return combined_mask

    return depth_static_mask


def robust_affine_fit(
    source: np.ndarray, target: np.ndarray, num_iterations: int = 100, inlier_fraction: float = 0.7
) -> tuple[float, float]:
    """RANSAC-style robust affine fit: target ~ scale * source + shift.

    Args:
        source: 1D array of source depth values (from frame i).
        target: 1D array of target depth values (from reference frame).
        num_iterations: Number of RANSAC iterations.
        inlier_fraction: Fraction of points to keep as inliers.

    Returns:
        (scale, shift) tuple.
    """
    n = len(source)
    if n < 10:
        LOGGER.warning("Too few points for robust fit (%d), using least squares", n)
        A = np.column_stack([source, np.ones(n)])
        result = np.linalg.lstsq(A, target, rcond=None)
        return float(result[0][0]), float(result[0][1])

    best_scale, best_shift = 1.0, 0.0
    best_inlier_count = 0

    for _ in range(num_iterations):
        # Sample 2 random points
        idx = np.random.choice(n, size=2, replace=False)
        s1, s2 = source[idx]
        t1, t2 = target[idx]

        denom = s1 - s2
        if abs(denom) < 1e-8:
            continue

        scale = (t1 - t2) / denom
        shift = t1 - scale * s1

        if scale <= 0:  # Depth scale should be positive
            continue

        # Count inliers
        residuals = np.abs(target - (scale * source + shift))
        threshold = np.percentile(residuals, inlier_fraction * 100)
        inlier_mask = residuals < threshold
        inlier_count = inlier_mask.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            # Refit on inliers
            A = np.column_stack([source[inlier_mask], np.ones(inlier_count)])
            result = np.linalg.lstsq(A, target[inlier_mask], rcond=None)
            best_scale = float(result[0][0])
            best_shift = float(result[0][1])

    return best_scale, best_shift


def align_depth_maps(
    depths: list[np.ndarray],
    static_mask: np.ndarray,
    reference_idx: int | None = None,
) -> tuple[list[np.ndarray], int]:
    """Align all depth maps to a reference frame using static regions.

    Args:
        depths: List of HxW depth maps.
        static_mask: Boolean HxW mask of static pixels.
        reference_idx: Index of reference frame (auto-selected if None).

    Returns:
        (aligned_depths, reference_idx) tuple.
    """
    n = len(depths)

    # Auto-select reference: frame whose median depth is closest to the overall median
    if reference_idx is None:
        medians = [np.median(d[static_mask]) for d in depths]
        overall_median = np.median(medians)
        reference_idx = int(np.argmin(np.abs(np.array(medians) - overall_median)))
        LOGGER.info("Auto-selected reference frame: %d (median depth: %.2f)", reference_idx, medians[reference_idx])

    ref_depth = depths[reference_idx]
    ref_static = ref_depth[static_mask].flatten()

    aligned_depths = []
    for i in range(n):
        if i == reference_idx:
            aligned_depths.append(ref_depth.copy())
            continue

        src_static = depths[i][static_mask].flatten()
        scale, shift = robust_affine_fit(src_static, ref_static)

        aligned = scale * depths[i] + shift
        aligned = np.clip(aligned, a_min=0.1, a_max=1000.0)  # Safety clamp
        aligned_depths.append(aligned)

        LOGGER.info("Frame %d: scale=%.4f, shift=%.4f", i, scale, shift)

    return aligned_depths, reference_idx


def create_gaussian_static_mask(
    pixel_static_mask: np.ndarray,
    gaussian_grid_size: int = 768,
    num_layers: int = 2,
    stride: int = 2,
) -> np.ndarray:
    """Convert pixel-level static mask to Gaussian-level mask.

    SHARP generates Gaussians on a grid at stride=2 from 1536x1536 -> 768x768,
    with num_layers=2 per pixel -> total 768*768*2 = 1,179,648 Gaussians.

    After flattening in GaussianComposer: shape [B, num_layers*H*W, C]
    Order: [layer0_row0_col0, layer0_row0_col1, ..., layer1_row0_col0, ...]

    Args:
        pixel_static_mask: Boolean HxW mask at original frame resolution.
        gaussian_grid_size: Size of the Gaussian grid (768 for default SHARP).
        num_layers: Number of Gaussian layers per pixel (2 for default SHARP).
        stride: Downsampling stride from internal res to Gaussian grid.

    Returns:
        Boolean 1D mask of length num_layers * gaussian_grid_size^2.
    """
    # Resize static mask to Gaussian grid resolution
    mask_tensor = torch.from_numpy(pixel_static_mask.astype(np.float32))[None, None]
    # First resize to internal resolution (1536x1536), then to grid
    internal_size = gaussian_grid_size * stride  # 1536
    mask_internal = F.interpolate(mask_tensor, size=(internal_size, internal_size), mode="bilinear")
    # Pool to grid size (simulating stride=2 downsampling)
    mask_grid = F.avg_pool2d(mask_internal, kernel_size=stride, stride=stride)
    mask_grid_bool = (mask_grid[0, 0] > 0.5).numpy()  # [768, 768]

    # Expand to all layers: [num_layers, H, W] -> flatten to [num_layers * H * W]
    mask_expanded = np.tile(mask_grid_bool[None], (num_layers, 1, 1))  # [2, 768, 768]
    return mask_expanded.flatten()


def enforce_static_consistency(
    gaussians_list: list,
    reference_idx: int,
    static_mask_1d: np.ndarray,
) -> list:
    """Replace static Gaussians in all frames with reference frame values.

    For static regions, we lock position, scale, and orientation to the reference frame.
    Colors are kept per-frame to allow subtle lighting changes.

    Args:
        gaussians_list: List of per-frame Gaussians3D.
        reference_idx: Index of the reference frame.
        static_mask_1d: Boolean 1D mask over Gaussians (True = static).

    Returns:
        List of Gaussians3D with static consistency enforced.
    """
    from sharp.utils.gaussians import Gaussians3D

    ref = gaussians_list[reference_idx]
    mask = torch.from_numpy(static_mask_1d).bool()

    result = []
    for i, g in enumerate(gaussians_list):
        if i == reference_idx:
            result.append(g)
            continue

        # Clone tensors
        mean_vectors = g.mean_vectors.clone()
        singular_values = g.singular_values.clone()
        quaternions = g.quaternions.clone()
        colors = g.colors.clone()  # Keep per-frame colors
        opacities = g.opacities.clone()

        # Replace static Gaussian geometry with reference
        mean_vectors[0, mask] = ref.mean_vectors[0, mask]
        singular_values[0, mask] = ref.singular_values[0, mask]
        quaternions[0, mask] = ref.quaternions[0, mask]
        opacities[0, mask] = ref.opacities[0, mask]

        result.append(Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        ))

    return result
