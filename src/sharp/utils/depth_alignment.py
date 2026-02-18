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
) -> np.ndarray:
    """Compute a static mask using RAFT optical flow with global motion compensation.

    For each consecutive pair the raw RAFT flow is computed and then an affine
    camera-motion model is fitted (iteratively-reweighted least squares, 2
    rounds) and subtracted.  The residual captures independent object motion
    only — camera translation, rotation and zoom are removed.

    The per-pixel **median** residual magnitude across all pairs is
    thresholded: pixels that stay below ``flow_magnitude_threshold`` in the
    median are marked as static.

    Args:
        frames: List of HxWx3 uint8 RGB frames (N frames).
        device: Torch device for RAFT inference.
        flow_magnitude_threshold: Maximum median residual flow magnitude
            (pixels at original resolution) for a pixel to count as static.

    Returns:
        Boolean mask HxW where True = static pixel.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    raft_model = raft_small(weights=weights).eval().to(device)
    transforms = weights.transforms()

    n = len(frames)
    h_orig, w_orig = frames[0].shape[:2]
    h_raft, w_raft = _RAFT_SIZE
    total_pixels = h_orig * w_orig
    num_pairs = n - 1

    LOGGER.info(
        "Running RAFT optical flow on %d consecutive pairs (%d frames, %dx%d) "
        "with global motion compensation...",
        num_pairs, n, w_orig, h_orig,
    )

    # Pre-compute coordinate grid for affine fit (at RAFT resolution)
    ys_raft, xs_raft = np.mgrid[:h_raft, :w_raft].astype(np.float32)
    n_raft = h_raft * w_raft
    A = np.column_stack([xs_raft.ravel(), ys_raft.ravel(), np.ones(n_raft, dtype=np.float32)])

    # Scale factor for mapping flow magnitudes from RAFT to original resolution
    mag_scale = max(w_orig / w_raft, h_orig / h_raft)

    residual_magnitudes: list[np.ndarray] = []

    for idx in range(num_pairs):
        img1 = torch.from_numpy(frames[idx].copy()).permute(2, 0, 1).float()
        img2 = torch.from_numpy(frames[idx + 1].copy()).permute(2, 0, 1).float()

        img1 = F.interpolate(img1[None], size=_RAFT_SIZE, mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2[None], size=_RAFT_SIZE, mode="bilinear", align_corners=False)

        img1_t, img2_t = transforms(img1.to(device), img2.to(device))

        flow_list = raft_model(img1_t, img2_t)
        flow = flow_list[-1][0].cpu().numpy()  # [2, h_raft, w_raft]

        dx = flow[0].ravel()
        dy = flow[1].ravel()

        # Iteratively-reweighted affine fit to estimate camera motion.
        # dx ≈ a0*x + a1*y + a2   (affine model for horizontal flow)
        # dy ≈ b0*x + b1*y + b2   (affine model for vertical flow)
        # Two iterations: first unweighted, then downweight outliers (dynamic pixels).
        wt = np.ones(n_raft, dtype=np.float32)
        pred_dx = np.zeros(n_raft, dtype=np.float32)
        pred_dy = np.zeros(n_raft, dtype=np.float32)

        for _iter in range(2):
            Aw = A * wt[:, None]  # [N, 3]  weighted design matrix
            AtWA = Aw.T @ A  # [3, 3]
            coeff_x = np.linalg.solve(AtWA, Aw.T @ dx)
            coeff_y = np.linalg.solve(AtWA, Aw.T @ dy)
            pred_dx = A @ coeff_x
            pred_dy = A @ coeff_y
            residual = np.sqrt((dx - pred_dx) ** 2 + (dy - pred_dy) ** 2)
            med_res = np.median(residual)
            wt = (residual < 3.0 * med_res + 1e-6).astype(np.float32)

        res_mag = residual.reshape(h_raft, w_raft)

        # Resize residual magnitude to original frame resolution & scale
        res_t = torch.from_numpy(res_mag)[None, None]
        res_orig = F.interpolate(res_t, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        residual_magnitudes.append(res_orig[0, 0].numpy() * mag_scale)

        if (idx + 1) % 10 == 0 or idx == num_pairs - 1:
            LOGGER.info("  Optical flow: pair %d/%d done", idx + 1, num_pairs)

    # Median residual magnitude per pixel across all consecutive pairs.
    mag_stack = np.stack(residual_magnitudes, axis=0)  # [num_pairs, H, W]
    median_mag = np.median(mag_stack, axis=0)  # [H, W]

    # Diagnostic percentiles so the user can tune the threshold
    pcts = np.percentile(median_mag, [10, 25, 50, 75, 90, 99])
    LOGGER.info(
        "  Residual flow percentiles — p10=%.2f  p25=%.2f  p50=%.2f  "
        "p75=%.2f  p90=%.2f  p99=%.2f px",
        *pcts,
    )

    static_mask = median_mag < flow_magnitude_threshold
    static_count = int(static_mask.sum())
    static_pct = 100.0 * static_count / total_pixels

    LOGGER.info(
        "Optical-flow static mask — static pixels: %d / %d (%.1f%% of %dx%d image), "
        "threshold: %.2f px, pairs evaluated: %d",
        static_count, total_pixels, static_pct, w_orig, h_orig,
        flow_magnitude_threshold, num_pairs,
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
    h, w = depth_stack.shape[1], depth_stack.shape[2]
    total_pixels = h * w

    # Normalize each depth to [0,1] range before computing variance
    # (accounts for per-frame scale differences in raw monocular depth)
    depth_medians = np.median(depth_stack, axis=(1, 2), keepdims=True)
    depth_normalized = depth_stack / (depth_medians + 1e-6)

    depth_variance = np.var(depth_normalized, axis=0)  # [H, W]
    variance_threshold = np.percentile(depth_variance[depth_variance > 0], depth_var_percentile)
    depth_static_mask = depth_variance < variance_threshold

    depth_static_pct = 100.0 * int(depth_static_mask.sum()) / total_pixels
    LOGGER.info(
        "Depth-variance static mask: %d / %d pixels (%.1f%% of %dx%d image)",
        int(depth_static_mask.sum()), total_pixels, depth_static_pct, w, h,
    )

    if use_optical_flow and frames is not None and len(frames) > 1:
        flow_static_mask = compute_optical_flow_mask(
            frames,
            device=flow_device,
            flow_magnitude_threshold=flow_magnitude_threshold,
        )
        combined_mask = depth_static_mask & flow_static_mask
        combined_pct = 100.0 * int(combined_mask.sum()) / total_pixels
        LOGGER.info(
            "Combined (depth + optical-flow) static mask: %d / %d pixels "
            "(%.1f%% of %dx%d image)",
            int(combined_mask.sum()), total_pixels, combined_pct, w, h,
        )
        return combined_mask

    if frames is not None and len(frames) > 1:
        # Use median frame as reference for RGB differencing
        frame_stack = np.stack(frames, axis=0).astype(np.float32)  # [N, H, W, 3]
        median_frame = np.median(frame_stack, axis=0)  # [H, W, 3]
        max_diff = np.max(np.abs(frame_stack - median_frame[None]), axis=(0, 3))  # [H, W]
        rgb_static_mask = max_diff < rgb_diff_threshold

        combined_mask = depth_static_mask & rgb_static_mask
        combined_pct = 100.0 * int(combined_mask.sum()) / total_pixels
        LOGGER.info(
            "Combined (depth + RGB) static mask: %d / %d pixels "
            "(%.1f%% of %dx%d image)",
            int(combined_mask.sum()), total_pixels, combined_pct, w, h,
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
