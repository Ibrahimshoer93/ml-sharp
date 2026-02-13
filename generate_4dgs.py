"""Align depth maps across frames and generate 4D Gaussian Splats.

Usage:
    python generate_4dgs.py -i extracted_output_dir/ -o 4dgs_output/ [-c checkpoint.pt] [--device cuda]

Expects the output directory from extract_depths.py containing:
    extracted_output_dir/
    ├── frames/frame_00000.png, frame_00001.png, ...
    ├── depths/frame_00000.npy, frame_00001.npy, ...
    └── extraction_metadata.json

Pipeline:
1. Load raw depth maps from extract_depths.py output
2. Detect static regions via temporal depth variance + RGB frame differencing
3. Compute per-frame affine alignment to a reference frame (robust fit on static regions)
4. Run full SHARP predictor per frame, feeding aligned depth as guidance
5. Enforce temporal consistency: lock static Gaussians to reference frame values
6. Save per-frame .ply files as 4DGS sequence
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, save_ply, unproject_gaussians

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# --- Depth Alignment Utilities ---


def compute_static_mask(
    depths: list[np.ndarray],
    frames: list[np.ndarray] | None = None,
    depth_var_percentile: float = 30.0,
    rgb_diff_threshold: float = 15.0,
) -> np.ndarray:
    """Identify static pixels using temporal depth variance and optional RGB differencing.

    Args:
        depths: List of HxW depth maps.
        frames: Optional list of HxWx3 uint8 RGB frames.
        depth_var_percentile: Percentile threshold for depth variance (lower = stricter).
        rgb_diff_threshold: Max RGB L1 difference to consider a pixel static.

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
    """RANSAC-style robust affine fit: target ≈ scale * source + shift.

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


# --- 4DGS Prediction ---


@torch.no_grad()
def predict_frame_with_depth(
    predictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    aligned_depth: np.ndarray | None = None,
) -> Gaussians3D:
    """Run full SHARP prediction on a frame, optionally using aligned depth as guidance.

    Args:
        predictor: RGBGaussianPredictor model.
        image: HxWx3 uint8 image.
        f_px: Focal length in pixels.
        device: Torch device.
        aligned_depth: Optional HxW aligned depth map for cross-frame consistency.

    Returns:
        Gaussians3D in metric space.
    """
    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized = F.interpolate(
        image_pt[None], size=internal_shape, mode="bilinear", align_corners=True
    )

    # Prepare aligned depth tensor if provided
    depth_input = None
    if aligned_depth is not None:
        depth_tensor = torch.from_numpy(aligned_depth).float().to(device)
        depth_tensor = depth_tensor[None, None]  # [1, 1, H, W]
        depth_input = F.interpolate(
            depth_tensor, size=internal_shape, mode="bilinear", align_corners=True
        )

    # Run predictor with optional depth guidance
    gaussians_ndc = predictor(image_resized, disparity_factor, depth=depth_input)

    # Unproject to metric space
    intrinsics = torch.tensor([
        [f_px, 0, width / 2, 0],
        [0, f_px, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).float().to(device)

    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )
    return gaussians


def create_gaussian_static_mask(
    pixel_static_mask: np.ndarray,
    gaussian_grid_size: int = 768,
    num_layers: int = 2,
    stride: int = 2,
) -> np.ndarray:
    """Convert pixel-level static mask to Gaussian-level mask.

    SHARP generates Gaussians on a grid at stride=2 from 1536x1536 → 768x768,
    with num_layers=2 per pixel → total 768*768*2 = 1,179,648 Gaussians.

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

    # Expand to all layers: [num_layers, H, W] → flatten to [num_layers * H * W]
    mask_expanded = np.tile(mask_grid_bool[None], (num_layers, 1, 1))  # [2, 768, 768]
    return mask_expanded.flatten()


def enforce_static_consistency(
    gaussians_list: list[Gaussians3D],
    reference_idx: int,
    static_mask_1d: np.ndarray,
) -> list[Gaussians3D]:
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


# --- Main Pipeline ---


def main():
    parser = argparse.ArgumentParser(description="Generate 4DGS from extracted frames and depths")
    parser.add_argument("-i", "--input-path", type=Path, required=True,
                        help="Output directory from extract_depths.py")
    parser.add_argument("-o", "--output-path", type=Path, required=True,
                        help="Output directory for 4DGS .ply sequence")
    parser.add_argument("-c", "--checkpoint-path", type=Path, default=None,
                        help="Path to SHARP .pt checkpoint")
    parser.add_argument("--device", type=str, default="default")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip depth alignment (use raw depths)")
    parser.add_argument("--no-static-lock", action="store_true",
                        help="Skip static Gaussian consistency enforcement")
    parser.add_argument("--depth-var-percentile", type=float, default=30.0,
                        help="Percentile for depth variance threshold (lower = stricter static mask)")
    parser.add_argument("--rgb-diff-threshold", type=float, default=15.0,
                        help="RGB difference threshold for static detection")
    parser.add_argument("--reference-frame", type=int, default=None,
                        help="Reference frame index (auto-selected if omitted)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames to process")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve device
    device = args.device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    LOGGER.info("Using device: %s", device)

    input_path: Path = args.input_path
    output_path: Path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Load extraction metadata
    meta_path = input_path / "extraction_metadata.json"
    if not meta_path.exists():
        LOGGER.error("extraction_metadata.json not found in %s. Run extract_depths.py first.", input_path)
        return
    with open(meta_path) as f:
        meta = json.load(f)

    num_frames = meta["num_frames"]
    if args.max_frames:
        num_frames = min(num_frames, args.max_frames)

    focal_lengths = meta["focal_lengths_px"]
    fps = meta.get("fps", 30.0)

    # Load frames and depth maps
    LOGGER.info("Loading %d frames and depth maps...", num_frames)
    frames = []
    depths = []
    frame_paths = []

    frames_dir = input_path / "frames"
    depths_dir = input_path / "depths"

    for i in range(num_frames):
        # Find frame file
        frame_candidates = list(frames_dir.glob(f"frame_{i:05d}.*"))
        if not frame_candidates:
            LOGGER.error("Frame %d not found", i)
            return
        frame_path = frame_candidates[0]
        frame_paths.append(frame_path)

        image, _, _ = sharp_io.load_rgb(frame_path)
        frames.append(image)

        depth_path = depths_dir / f"frame_{i:05d}.npy"
        if not depth_path.exists():
            LOGGER.error("Depth map %s not found", depth_path)
            return
        depths.append(np.load(depth_path))

    # Step 1: Compute static mask
    LOGGER.info("Computing static mask...")
    static_mask = compute_static_mask(
        depths, frames,
        depth_var_percentile=args.depth_var_percentile,
        rgb_diff_threshold=args.rgb_diff_threshold,
    )
    np.save(output_path / "static_mask.npy", static_mask)

    # Step 2: Align depth maps
    if args.no_align:
        LOGGER.info("Skipping depth alignment (--no-align)")
        aligned_depths = depths
        reference_idx = 0
    else:
        LOGGER.info("Aligning depth maps across frames...")
        aligned_depths, reference_idx = align_depth_maps(
            depths, static_mask, reference_idx=args.reference_frame
        )
        # Save aligned depths
        aligned_dir = output_path / "aligned_depths"
        aligned_dir.mkdir(exist_ok=True)
        for i, ad in enumerate(aligned_depths):
            np.save(aligned_dir / f"frame_{i:05d}.npy", ad)

    # Step 3: Load SHARP model
    LOGGER.info("Loading SHARP model...")
    if args.checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(args.checkpoint_path, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)

    # Step 4: Generate per-frame Gaussians
    LOGGER.info("Generating per-frame Gaussians...")
    gaussians_list = []
    ply_dir = output_path / "ply_sequence"
    ply_dir.mkdir(exist_ok=True)

    for i in range(num_frames):
        LOGGER.info("Predicting frame %d/%d", i + 1, num_frames)

        f_px = focal_lengths[i]
        image = frames[i]
        aligned_depth = aligned_depths[i] if not args.no_align else None

        gaussians = predict_frame_with_depth(
            predictor, image, f_px, device, aligned_depth=aligned_depth
        )
        gaussians_list.append(gaussians)

        if device.type == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Step 5: Enforce static consistency
    if not args.no_static_lock:
        LOGGER.info("Enforcing static Gaussian consistency...")
        gaussian_static_mask = create_gaussian_static_mask(static_mask)
        gaussians_list = enforce_static_consistency(
            gaussians_list, reference_idx, gaussian_static_mask
        )

    # Step 6: Save per-frame .ply files
    LOGGER.info("Saving 4DGS sequence...")
    height, width = frames[0].shape[:2]
    for i, gaussians in enumerate(gaussians_list):
        ply_path = ply_dir / f"frame_{i:05d}.ply"
        save_ply(gaussians, focal_lengths[i], (height, width), ply_path)

    # Save 4DGS metadata
    metadata_4dgs = {
        "num_frames": num_frames,
        "fps": fps,
        "resolution": [width, height],
        "focal_lengths_px": focal_lengths[:num_frames],
        "color_space": "linearRGB",
        "reference_frame": reference_idx,
        "static_mask_path": "static_mask.npy",
        "depth_aligned": not args.no_align,
        "static_locked": not args.no_static_lock,
    }
    with open(output_path / "metadata_4dgs.json", "w") as f:
        json.dump(metadata_4dgs, f, indent=2)

    LOGGER.info("Done! Saved %d frames to %s", num_frames, ply_dir)
    LOGGER.info("To render, use: sharp render -i %s -o rendered/", ply_dir)


if __name__ == "__main__":
    main()
