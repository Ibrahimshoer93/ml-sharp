"""Contains `sharp predict-4dgs` CLI implementation.

Align depth maps across frames and generate 4D Gaussian Splat sequences.

Expects the output directory from ``sharp extract-depths`` containing:
    extracted_output_dir/
    ├── frames/frame_00000.png, frame_00001.png, ...
    ├── depths/frame_00000.npy, frame_00001.npy, ...
    └── extraction_metadata.json

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils import logging as logging_utils
from sharp.utils.depth_alignment import (
    align_depth_maps,
    compute_static_mask,
    create_gaussian_static_mask,
    enforce_static_consistency,
)
from sharp.utils.gaussians import save_ply

from .predict import predict_image_with_depth

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Output directory from extract-depths (contains frames/, depths/, extraction_metadata.json).",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory for 4DGS .ply sequence.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to SHARP .pt checkpoint (downloads default if omitted).",
    required=False,
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device: 'cpu', 'mps', 'cuda', or 'default'.",
)
@click.option(
    "--no-align",
    is_flag=True,
    default=False,
    help="Skip depth alignment (use raw depths).",
)
@click.option(
    "--no-static-lock",
    is_flag=True,
    default=False,
    help="Skip static Gaussian consistency enforcement.",
)
@click.option(
    "--depth-var-percentile",
    type=float,
    default=30.0,
    help="Percentile for depth variance threshold (lower = stricter static mask).",
)
@click.option(
    "--rgb-diff-threshold",
    type=float,
    default=15.0,
    help="RGB difference threshold for static detection.",
)
@click.option(
    "--reference-frame",
    type=int,
    default=None,
    help="Reference frame index (auto-selected if omitted).",
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Limit number of frames to process.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_4dgs_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    device: str,
    no_align: bool,
    no_static_lock: bool,
    depth_var_percentile: float,
    rgb_diff_threshold: float,
    reference_frame: int | None,
    max_frames: int | None,
    verbose: bool,
):
    """Generate 4D Gaussian Splats from extracted frames and depth maps."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    # Resolve device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    LOGGER.info("Using device: %s", device)

    output_path.mkdir(parents=True, exist_ok=True)

    # Load extraction metadata
    meta_path = input_path / "extraction_metadata.json"
    if not meta_path.exists():
        LOGGER.error("extraction_metadata.json not found in %s. Run extract-depths first.", input_path)
        return
    with open(meta_path) as f:
        meta = json.load(f)

    num_frames = meta["num_frames"]
    if max_frames:
        num_frames = min(num_frames, max_frames)

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
        depth_var_percentile=depth_var_percentile,
        rgb_diff_threshold=rgb_diff_threshold,
    )
    np.save(output_path / "static_mask.npy", static_mask)

    # Step 2: Align depth maps
    if no_align:
        LOGGER.info("Skipping depth alignment (--no-align)")
        aligned_depths = depths
        reference_idx = 0
    else:
        LOGGER.info("Aligning depth maps across frames...")
        aligned_depths, reference_idx = align_depth_maps(
            depths, static_mask, reference_idx=reference_frame
        )
        # Save aligned depths
        aligned_dir = output_path / "aligned_depths"
        aligned_dir.mkdir(exist_ok=True)
        for i, ad in enumerate(aligned_depths):
            np.save(aligned_dir / f"frame_{i:05d}.npy", ad)

    # Step 3: Load SHARP model
    LOGGER.info("Loading SHARP model...")
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

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

        # Prepare aligned depth tensor if available
        aligned_depth_tensor = None
        if not no_align:
            aligned_depth_tensor = torch.from_numpy(aligned_depths[i]).float().to(device)
            aligned_depth_tensor = aligned_depth_tensor[None, None]  # [1, 1, H, W]

        gaussians = predict_image_with_depth(
            predictor, image, f_px, device, aligned_depth=aligned_depth_tensor
        )
        gaussians_list.append(gaussians)

        if device.type == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Step 5: Enforce static consistency
    if not no_static_lock:
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
        "depth_aligned": not no_align,
        "static_locked": not no_static_lock,
    }
    with open(output_path / "metadata_4dgs.json", "w") as f:
        json.dump(metadata_4dgs, f, indent=2)

    LOGGER.info("Done! Saved %d frames to %s", num_frames, ply_dir)
    LOGGER.info("To render, use: sharp render -i %s -o rendered/ --sequence", ply_dir)
