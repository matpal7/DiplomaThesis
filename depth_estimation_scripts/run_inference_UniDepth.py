from __future__ import annotations

import argparse
import logging
import time as time_module
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole

from code.depth_compute.depth_utils import load_calibration, save_outputs
from code.depth_compute.run_stats import RunStats, save_run_stats
from code.image import load_l_r_images_rectified
from code.utils import load_dict
from code.prepare_paths import build_paths, prepare_output_dirs
from visualize_depth import colorize_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniDepth on folders of rectified left images."
    )
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_name",
        type=str,
        default="lpiccinelli/unidepth-v2-vitl14",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--interpolation_mode",
        type=str,
        default="bilinear",
    )
    parser.add_argument("--resolution_level", type=int, default=None)
    parser.add_argument(
        "--run_name",
        type=str,
        default="UniDepth_mono",
    )
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> UniDepthV2:
    model = UniDepthV2.from_pretrained(model_name)
    model = model.to(device).eval()
    return model

def warmup_model(
    model:  UniDepthV2,
    K_np:   np.ndarray,
    inference_resolution: tuple[int, int],
    device: torch.device,
    n_warmup: int = 5,
) -> None:
    logging.info(
        "Warming up UniDepth (%d passes, shape=%s)...", n_warmup, inference_resolution
    )
    w, h = inference_resolution

    dummy_np = np.zeros((h, w, 3), dtype=np.float32)
    dummy_t  = torch.from_numpy(dummy_np).permute(2, 0, 1).to(device)

    intrinsics_t = torch.from_numpy(K_np)
    if isinstance(model, (UniDepthV2old, UniDepthV1)):
        camera = intrinsics_t.to(device)
    else:
        camera = Pinhole(K=intrinsics_t.unsqueeze(0))

    for _ in range(n_warmup):
        _ = model.infer(dummy_t, camera)

    if device.type == "cuda":
        torch.cuda.synchronize()

    logging.info("Warm-up complete.")

def infer_depth(
    model: UniDepthV2,
    image_rgb: np.ndarray,
    K_np: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    rgb_torch = torch.from_numpy(image_rgb).permute(2, 0, 1).to(device)

    intrinsics_torch = torch.from_numpy(K_np)
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))

    t0 = time_module.perf_counter()
    predictions = model.infer(rgb_torch, camera)
    elapsed = time_module.perf_counter() - t0

    depth = predictions["depth"].squeeze().detach().cpu().numpy()
    return depth.astype(np.float32), elapsed


def main() -> None:
    args = parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.autograd.set_grad_enabled(False)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    root_dir = Path(__file__).resolve().parents[3]
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, args.run_name)
    out_dirs = prepare_output_dirs(out_dir)

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dir)
    logging.info("Device: %s", device)

    calib_dict = load_dict(calib_dict_file)
    model = load_model(args.model_name, device)

    if args.resolution_level is not None and hasattr(model, "resolution_level"):
        model.resolution_level = args.resolution_level
    if hasattr(model, "interpolation_mode"):
        model.interpolation_mode = args.interpolation_mode

    imgs_l, _ = load_l_r_images_rectified(
        calib_dict,
        img_dir,
        max_imgs=args.max_imgs,
    )

    if len(imgs_l) == 0:
        raise RuntimeError("No images found from the provided input arguments")

    logging.info("Found %d images", len(imgs_l))

    img_full = imgs_l[0].get_img()
    img_small = imgs_l[0].get_small_img(scale=args.scale)

    input_resolution = (img_full.shape[1], img_full.shape[0])
    inference_resolution = (img_small.shape[1], img_small.shape[0])

    K, _, _ = load_calibration(
        calib_dict_file=calib_dict_file,
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
    )

    logging.info("Scaled fx: %.6f", float(K[0, 0]))

    stats = RunStats(
        run_name=args.run_name,
        date=args.date,
        model_ckpt=args.model_name,
        network_type="mono",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    warmup_model(
        model=model,
        K_np=K,
        inference_resolution=inference_resolution,
        device=device,
        n_warmup=5,
    )

    for img_l in tqdm(imgs_l, desc="mono depth inference"):
        img_number = img_l.get_image_number()
        img_small = img_l.get_small_img(scale=args.scale)

        depth, elapsed = infer_depth(
            model=model,
            image_rgb=img_small,
            K_np=K,
            device=device,
        )

        stats.per_image_stats.append({
            "image_id": img_number,
            "time_s": round(elapsed, 4),
            "depth_min": round(float(np.nanmin(depth)), 4),
            "depth_max": round(float(np.nanmax(depth)), 4),
            "depth_mean": round(float(np.nanmean(depth)), 4),
            "valid_px": int(np.isfinite(depth).sum()),
            "total_px": int(depth.size),
        })

        save_outputs(
            img_number=img_number,
            img_left=img_small,
            depth=depth,
            out_dirs=out_dirs,
        )

    save_run_stats(stats, out_dir)


if __name__ == "__main__":
    main()