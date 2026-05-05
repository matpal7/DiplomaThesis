# run_bridgedepth.py
# Inference script for BridgeDepth network, modeled after FoundationStereo pipeline.
from __future__ import annotations

import argparse
import logging
import os
import time as time_module
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
from tqdm.auto import tqdm

from bridgedepth.bridgedepth import BridgeDepth
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils import visualization

from code.depth_compute.depth_utils import load_calibration, disparity_to_depth, save_outputs
from code.depth_compute.run_stats import RunStats, save_run_stats
from code.image import load_l_r_images_rectified
from code.utils import load_dict
from code.prepare_paths import build_paths, prepare_output_dirs

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BridgeDepth on folders of rectified stereo images.")
    name = "rvc"

    parser.add_argument("--model_name", choices=["rvc", "rvc_pretrain", "eth3d_pretrain", "middlebury_pretrain"], default="rvc_pretrain")
    parser.add_argument("--checkpoint_path", default=f"/checkpoints/bridge_{name}_pretrain.pth", type=str, help="Path to local .pth checkpoint (overrides --model_name)")
    parser.add_argument("--scale", type=int, default=6, help="Downscale factor for inference (>= 1)")
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument("--min_disp", type=float, default=1e-3, help="Minimum disparity clamp")
    parser.add_argument("--z_far", type=float, default=10.0, help="Max depth to keep in point cloud (meters)")
    parser.add_argument("--n_warmup", type=int, default=3, help="Number of warmup forward passes")
    parser.add_argument("--run_name", type=str, default=f"BridgeDepth_{name}")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Model loading & warmup
# ─────────────────────────────────────────────

def load_model(args: argparse.Namespace) -> BridgeDepth:
    code_dir = Path(os.path.realpath(__file__)).parent.parent

    pretrained_model_name_or_path = args.model_name
    if args.checkpoint_path is not None:
        args.checkpoint_path = str(code_dir) + args.checkpoint_path
        assert os.path.exists(args.checkpoint_path), f"Checkpoint not found: {args.checkpoint_path}"
        pretrained_model_name_or_path = args.checkpoint_path

    logging.info("Loading BridgeDepth from: %s", pretrained_model_name_or_path)
    model = BridgeDepth.from_pretrained(pretrained_model_name_or_path)
    model = model.to(torch.device("cuda")).eval()
    return model


def warmup_model(model: BridgeDepth, inference_resolution: tuple[int, int], n_warmup: int = 3) -> None:
    logging.info("Warming up BridgeDepth (%d passes, shape=%s)...", n_warmup, inference_resolution)
    w, h = inference_resolution
    dummy = torch.zeros(1, 3, h, w, device="cuda", dtype=torch.float32)
    sample = {"img1": dummy, "img2": dummy}
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample)
    torch.cuda.synchronize()
    logging.info("Warm-up complete.")


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def compute_disparity(
    model: BridgeDepth,
    img0: np.ndarray,
    img1: np.ndarray,
    min_disp: float,
) -> tuple[np.ndarray, float]:
    H, W = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda(non_blocking=True).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda(non_blocking=True).float()[None].permute(0, 3, 1, 2)

    sample = {"img1": img0_t, "img2": img1_t}

    torch.cuda.synchronize()
    t0 = time_module.perf_counter()
    with torch.no_grad():
        results_dict = model(sample)
    torch.cuda.synchronize()
    elapsed = time_module.perf_counter() - t0

    disp = results_dict["disp_pred"].clamp_min(min_disp).cpu().numpy().reshape(H, W)
    return disp.astype(np.float32), elapsed


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    assert args.scale >= 1, "scale must be >= 1"

    logger = setup_logger(name="bridgedepth")
    logging.basicConfig(level=logging.INFO)

    torch.autograd.set_grad_enabled(False)

    root_dir = Path(__file__).resolve().parents[3]
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, args.run_name)
    out_dirs = prepare_output_dirs(out_dir, disp_dir=True)

    logging.info("Image dir:        %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir:       %s", out_dir)

    calib_dict = load_dict(calib_dict_file)

    imgs_l, imgs_r = load_l_r_images_rectified(calib_dict, img_dir, max_imgs=args.max_imgs)
    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found.")
    logging.info("Found %d pairs", len(imgs_l))

    img_full = imgs_l[0].get_img()
    img_small = imgs_l[0].get_small_img(scale=args.scale)
    input_resolution = (img_full.shape[1], img_full.shape[0])
    inference_resolution = (img_small.shape[1], img_small.shape[0])

    K, baseline_m, calib_dict = load_calibration(
        calib_dict_file=calib_dict_file,
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
    )
    fx = float(K[0, 0])
    logging.info("Scaled fx: %.6f", fx)
    logging.info("Baseline:  %.6f m", baseline_m)

    model = load_model(args)
    warmup_model(model, inference_resolution, n_warmup=args.n_warmup)

    stats = RunStats(
        run_name=args.run_name,
        date=args.date,
        model_ckpt=args.checkpoint_path or args.model_name,
        network_type="stereo",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc="inference"):
        img_number = img_l.get_image_number()
        img0 = img_l.get_small_img(scale=args.scale)
        img1 = img_r.get_small_img(scale=args.scale)

        disp, elapsed = compute_disparity(model, img0, img1, min_disp=args.min_disp)

        depth = disparity_to_depth(disp=disp, fx=fx, baseline_m=baseline_m, min_disp=args.min_disp)

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
            img_left=img0,
            depth=depth,
            out_dirs=out_dirs,
            disp=disp,
        )


    save_run_stats(stats, out_dir)
    logging.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()