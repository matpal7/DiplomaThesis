from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
import time as time_module

from depth_compute.depth_utils import save_outputs
from depth_compute.run_stats import RunStats, save_run_stats
from moge.model.v2 import MoGeModel

from code.image import load_l_r_images_undistorted, load_l_r_images_rectified
from code.utils import load_dict
from prepare_paths import build_paths, prepare_output_dirs
from visualize_depth import colorize_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoGe v2 on folders of undistorted left images."
    )
    parser.add_argument("--max_imgs",   type=int, default=None)
    parser.add_argument("--scale",      type=int, default=6)
    parser.add_argument("--date",       type=str, default="24042026")
    parser.add_argument("--model_name", type=str, default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--run_name",   type=str, default="MoGe_mono")
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> MoGeModel:
    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    return model

def warmup_model(
    model: MoGeModel,
    inference_resolution: tuple[int, int],
    device: torch.device,
    n_warmup: int = 5,
) -> None:
    logging.info(
        "Warming up MoGe (%d passes, shape=%s)...", n_warmup, inference_resolution
    )
    w, h = inference_resolution

    # MoGe expects a float32 tensor in [0, 1] with shape (3, H, W)
    dummy = torch.zeros(3, h, w, dtype=torch.float32, device=device)

    for _ in range(n_warmup):
        _ = model.infer(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    logging.info("Warm-up complete.")

def prepare_run_paths(root_dir: Path, date: str, run_name: str) -> tuple[Path, Path, dict[str, Path]]:
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, date, run_name)
    out_dirs = prepare_output_dirs(out_dir)
    return img_dir, calib_dict_file, out_dirs


def infer_depth(
    model: MoGeModel,
    image_rgb: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    input_tensor = (
        torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
    )
    t0 = time_module.perf_counter()
    output = model.infer(input_tensor)
    elapsed = time_module.perf_counter() - t0

    depth = output["depth"].detach().cpu().numpy()
    mask = output.get("mask")
    mask = mask.detach().cpu().numpy() if mask is not None else None

    if mask is not None:
        depth[~mask] = np.nan

    return depth, mask, elapsed



def main() -> None:
    args = parse_args()

    if args.scale < 1:
        raise ValueError("scale must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.autograd.set_grad_enabled(False)

    root_dir = Path(__file__).resolve().parents[3]
    device = torch.device(args.device)

    img_dir, calib_dict_file, out_dirs = prepare_run_paths(
        root_dir=root_dir,
        date=args.date,
        run_name=args.run_name,
    )

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dirs["root"])
    logging.info("Device: %s", device)

    calib_dict = load_dict(calib_dict_file)
    model = load_model(args.model_name, device)

    left_images, _ = load_l_r_images_rectified(
        calib_dict,
        img_dir,
        max_imgs=args.max_imgs,
    )

    if len(left_images) == 0:
        raise RuntimeError("No images found from the provided input arguments")

    logging.info("Found %d images", len(left_images))

    img_full = left_images[0].get_img()
    img_small = left_images[0].get_small_img(scale=args.scale)

    input_resolution = (img_full.shape[1], img_full.shape[0])
    inference_resolution = (img_small.shape[1], img_small.shape[0])

    stats = RunStats(
        run_name=args.run_name,
        date=args.date,
        model_ckpt=args.model_name,
        network_type="mono",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(left_images),
    )

    warmup_model(
        model=model,
        inference_resolution=inference_resolution,
        device=device,
        n_warmup=5,
    )

    for left_image in tqdm(left_images, desc="mono depth inference"):
        img_number = left_image.get_image_number()
        image_small = left_image.get_small_img(scale=args.scale)

        depth, mask, elapsed = infer_depth(
            model=model,
            image_rgb=image_small,
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
            img_left=image_small,
            depth=depth,
            out_dirs=out_dirs,
        )

    save_run_stats(stats, out_dirs["root"])


if __name__ == "__main__":
    main()