"""Contains `sharp extract-depths` CLI implementation.

Extract video frames and depth maps using SHARP's monodepth model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def extract_frames_from_video(video_path: Path, output_dir: Path, fps: float | None = None) -> list[Path]:
    """Extract frames from video file using imageio."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    reader = iio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    video_fps = meta.get("fps", 30.0)

    if fps is None:
        fps = video_fps

    frame_interval = max(1, int(round(video_fps / fps)))
    LOGGER.info("Video FPS: %.1f, extracting at %.1f FPS (every %d frames)", video_fps, fps, frame_interval)

    saved_paths = []
    frame_idx = 0
    save_idx = 0

    for frame in reader:
        if frame_idx % frame_interval == 0:
            frame_path = frames_dir / f"frame_{save_idx:05d}.png"
            iio.imwrite(str(frame_path), frame[:, :, :3])  # Drop alpha if present
            saved_paths.append(frame_path)
            save_idx += 1
        frame_idx += 1

    reader.close()
    LOGGER.info("Extracted %d frames from %d total video frames", len(saved_paths), frame_idx)
    return saved_paths


def load_frames_from_directory(frames_dir: Path) -> list[Path]:
    """Load frame paths from a directory of images."""
    extensions = sharp_io.get_supported_image_extensions()
    paths = []
    for ext in extensions:
        paths.extend(frames_dir.glob(f"*{ext}"))
    paths = sorted(paths)
    LOGGER.info("Found %d frames in %s", len(paths), frames_dir)
    return paths


@torch.no_grad()
def extract_depth_map(
    monodepth_model,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    internal_shape: tuple[int, int] = (1536, 1536),
) -> np.ndarray:
    """Run monodepth on a single image and return metric depth map.

    Args:
        monodepth_model: The MonodepthWithEncodingAdaptor from SHARP.
        image: HxWx3 uint8 numpy array.
        f_px: Focal length in pixels.
        device: Torch device.
        internal_shape: Resolution to resize to for inference.

    Returns:
        Depth map as HxW float32 numpy array (at original resolution).
    """
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = f_px / width

    # Resize to internal resolution
    image_resized = F.interpolate(
        image_pt[None], size=internal_shape, mode="bilinear", align_corners=True
    )

    # Run monodepth -- returns MonodepthOutput with .disparity field
    monodepth_output = monodepth_model(image_resized)
    disparity = monodepth_output.disparity  # [1, num_layers, H_internal, W_internal]

    # Take first layer disparity for alignment purposes
    disparity_layer0 = disparity[:, 0:1]  # [1, 1, H, W]

    # Convert disparity to metric depth
    depth = disparity_factor / disparity_layer0.clamp(min=1e-4, max=1e4)

    # Resize back to original resolution
    depth_original = F.interpolate(
        depth, size=(height, width), mode="bilinear", align_corners=True
    )

    return depth_original[0, 0].cpu().numpy()


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to video file or directory of frames.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory for frames and depths.",
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
    "--fps",
    type=float,
    default=None,
    help="FPS to extract from video (default: video's native FPS).",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device: 'cpu', 'mps', 'cuda', or 'default'.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def extract_depths_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    fps: float | None,
    device: str,
    verbose: bool,
):
    """Extract frames and depth maps from video using SHARP's monodepth model."""
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

    # Extract or load frames
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if input_path.is_file() and input_path.suffix.lower() in video_extensions:
        frame_paths = extract_frames_from_video(input_path, output_path, fps=fps)
    elif input_path.is_dir():
        frame_paths = load_frames_from_directory(input_path)
        # Copy frames to output if needed
        frames_dir = output_path / "frames"
        frames_dir.mkdir(exist_ok=True)
        for i, fp in enumerate(frame_paths):
            dst = frames_dir / f"frame_{i:05d}{fp.suffix}"
            if not dst.exists():
                shutil.copy2(fp, dst)
        frame_paths = sorted(frames_dir.glob("*"))
    else:
        LOGGER.error("Input must be a video file or directory of frames")
        return

    if len(frame_paths) == 0:
        LOGGER.error("No frames found")
        return

    # Load SHARP model
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

    # Access the monodepth sub-model
    monodepth_model = predictor.monodepth_model

    # Extract depth maps
    depths_dir = output_path / "depths"
    depths_dir.mkdir(exist_ok=True)

    focal_lengths = []

    for i, frame_path in enumerate(frame_paths):
        LOGGER.info("Processing frame %d/%d: %s", i + 1, len(frame_paths), frame_path.name)

        image, _, f_px = sharp_io.load_rgb(frame_path)
        focal_lengths.append(f_px)

        depth_map = extract_depth_map(monodepth_model, image, f_px, device)

        # Save depth as numpy
        depth_path = depths_dir / f"frame_{i:05d}.npy"
        np.save(depth_path, depth_map)

        if i == 0 or (i + 1) % 10 == 0:
            LOGGER.info("  Depth range: [%.2f, %.2f] meters", depth_map.min(), depth_map.max())

        # Clear cache periodically
        if device.type == "cuda" and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "num_frames": len(frame_paths),
        "fps": fps or 30.0,
        "focal_lengths_px": focal_lengths,
        "frame_paths": [str(p.name) for p in frame_paths],
    }
    with open(output_path / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Done. Saved %d depth maps to %s", len(frame_paths), depths_dir)
