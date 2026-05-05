from __future__ import annotations

import argparse
import logging
import os
import sys
import time as time_module
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# DEFOM-Stereo repo layout
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "core"))

from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder

from code.depth_compute.depth_utils import load_calibration, disparity_to_depth, save_outputs
from code.depth_compute.run_stats import RunStats, save_run_stats
from code.image import load_l_r_images_rectified
from code.utils import load_dict
from code.prepare_paths import build_paths, prepare_output_dirs


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run DEFOM-Stereo on folders of rectified left/right images."
    )
    parser.add_argument(
        "--restore_ckpt",
        type=str,
        default=str(code_dir / "models" / "defomstereo_vitl_sceneflow.pth"),
    )
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min_disp",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--run_name", type=str, default="DEFOM_stereo")
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--scale_iters", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")

    # DEFOM-Stereo architecture args
    parser.add_argument("--dinov2_encoder", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--corr_implementation",
                        choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--scale_list", type=float, nargs="+",
                        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--scale_corr_radius", type=int, default=2)
    parser.add_argument("--n_downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument("--context_norm", type=str, default="batch",
                        choices=["group", "batch", "instance", "none"])
    parser.add_argument("--n_gru_layers", type=int, default=3)

    return parser.parse_args()


def load_model(args: argparse.Namespace) -> DEFOMStereo:
    model = DEFOMStereo(args)

    checkpoint = torch.load(args.restore_ckpt, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    model.cuda().eval()
    return model

def warmup_model(
    model: DEFOMStereo,
    inference_resolution: tuple[int, int],
    valid_iters: int,
    scale_iters: int,
    n_warmup: int = 5,
) -> None:
    logging.info("Warming up model (%d passes, shape=%s)...", n_warmup, inference_resolution)
    w, h = inference_resolution

    dummy = torch.zeros(1, 3, h, w, device="cuda", dtype=torch.float32)
    padder = InputPadder(dummy.shape, divis_by=32)
    dummy_pad, _ = padder.pad(dummy, dummy)

    for _ in range(n_warmup):
        _ = model(dummy_pad, dummy_pad,
                  iters=valid_iters, scale_iters=scale_iters, test_mode=True)

    torch.cuda.synchronize()
    logging.info("Warm-up complete.")

def compute_disparity(
    model: DEFOMStereo,
    img0: np.ndarray,
    img1: np.ndarray,
    valid_iters: int,
    scale_iters: int,
) -> tuple[np.ndarray, float]:
    h, w = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0_t.shape, divis_by=32)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    t0 = time_module.perf_counter()
    disp = model(img0_t, img1_t, iters=valid_iters, scale_iters=scale_iters, test_mode=True)
    elapsed = time_module.perf_counter() - t0

    disp = padder.unpad(disp.float()).cpu().numpy().reshape(h, w)
    return disp.astype(np.float32), elapsed


def main() -> None:
    args = parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    torch.autograd.set_grad_enabled(False)

    root_dir = Path(__file__).resolve().parents[3]
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, args.run_name)
    out_dirs = prepare_output_dirs(out_dir, disp_dir=True)

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dir)
    logging.info("Checkpoint: %s", args.restore_ckpt)

    calib_dict = load_dict(calib_dict_file)

    model = load_model(args)

    imgs_l, imgs_r = load_l_r_images_rectified(
        calib_dict,
        img_dir,
        max_imgs=args.max_imgs,
    )

    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found from the provided input arguments")

    logging.info("Found %d pairs", len(imgs_l))

    img_full = imgs_l[0].get_img()
    img_small = imgs_l[0].get_small_img(scale=args.scale)

    input_resolution = (img_full.shape[1], img_full.shape[0])
    inference_resolution = (img_small.shape[1], img_small.shape[0])

    K, baseline_m, _ = load_calibration(
        calib_dict_file=calib_dict_file,
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
    )
    fx = float(K[0, 0])

    logging.info("Scaled fx: %.6f", fx)
    logging.info("Baseline: %.6f m", baseline_m)

    stats = RunStats(
        run_name=args.run_name,
        date=args.date,
        model_ckpt=args.restore_ckpt,
        network_type="stereo",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    warmup_model(
        model=model,
        inference_resolution=inference_resolution,
        valid_iters=args.valid_iters,
        scale_iters=args.scale_iters,
        n_warmup=5,
    )

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc="defom-stereo inference"):
        img_number = img_l.get_image_number()

        img0 = img_l.get_small_img(scale=args.scale)
        img1 = img_r.get_small_img(scale=args.scale)

        disp, elapsed = compute_disparity(
            model=model,
            img0=img0,
            img1=img1,
            valid_iters=args.valid_iters,
            scale_iters=args.scale_iters,
        )

        depth = disparity_to_depth(
            disp=disp,
            fx=fx,
            baseline_m=baseline_m,
            min_disp=args.min_disp,
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
            img_left=img0,
            depth=depth,
            out_dirs=out_dirs,
            disp=disp,
        )

    save_run_stats(stats, out_dir)


if __name__ == "__main__":
    main()