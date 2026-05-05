from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import time as time_module
from datetime import time

from code.depth_compute.depth_utils import load_calibration, disparity_to_depth, save_outputs
from code.depth_compute.run_stats import RunStats, save_run_stats
from code.visualize_depth import colorize_depth
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import set_logging_format, vis_disparity, set_seed

from code.image import load_l_r_images_rectified
from code.utils import load_dict, scale_intrinsics
from code.prepare_paths import build_paths, prepare_output_dirs


def parse_args() -> argparse.Namespace:
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description="Run FoundationStereo on folders of rectified left/right images."
    )
    parser.add_argument(
        "--ckpt_dir",
        default=f"{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth",
        type=str,
        help="Pretrained model path",
    )
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument(
        "--min_disp",
        type=float,
        default=1e-6,
        help="Minimum disparity used to avoid division by zero.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="FoundationStereo",
        help="Output folder name for this model run.",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> FoundationStereo:
    cfg = OmegaConf.load(f"{os.path.dirname(args.ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    for key, val in vars(args).items():
        cfg[key] = val
    cfg = OmegaConf.create(cfg)

    model = FoundationStereo(cfg)
    ckpt = torch.load(args.ckpt_dir, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    # model = torch.compile(model, mode="reduce-overhead")
    return model

def warmup_model(
    model: FoundationStereo,
    inference_resolution: tuple[int, int],
    valid_iters: int,
    hiera: int,
    n_warmup: int = 5,
) -> None:
    logging.info(
        "Warming up FoundationStereo (%d passes, shape=%s)...",
        n_warmup, inference_resolution,
    )
    w, h = inference_resolution

    dummy = torch.zeros(1, 3, h, w, device="cuda", dtype=torch.float32)
    padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
    dummy_pad, _ = padder.pad(dummy, dummy)

    with torch.cuda.amp.autocast(True):
        for _ in range(n_warmup):
            if not hiera:
                _ = model.forward(dummy_pad, dummy_pad,
                                  iters=valid_iters, test_mode=True)
            else:
                _ = model.run_hierachical(
                    dummy_pad, dummy_pad,
                    iters=valid_iters, test_mode=True, small_ratio=0.5,
                )

    torch.cuda.synchronize()
    logging.info("Warm-up complete.")


def compute_disparity(
    model: FoundationStereo,
    img0: np.ndarray,
    img1: np.ndarray,
    valid_iters: int,
    hiera: int,
) -> tuple[Any, float]:
    h, w = img0.shape[:2]

    # img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    # img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    img0_t = torch.as_tensor(img0).cuda(non_blocking=True).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda(non_blocking=True).float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    torch.cuda.synchronize()
    t0 = time_module.perf_counter()
    with torch.cuda.amp.autocast(True):
        if not hiera:
            disp = model.forward(img0_t, img1_t, iters=valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(
                img0_t,
                img1_t,
                iters=valid_iters,
                test_mode=True,
                small_ratio=0.5,
            )
    torch.cuda.synchronize()
    elapsed = time_module.perf_counter() - t0

    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h, w)
    return disp.astype(np.float32), elapsed


def main() -> None:
    args = parse_args()

    assert args.scale >= 1, "scale must be >= 1"

    set_logging_format()
    set_seed(args.seed)
    torch.autograd.set_grad_enabled(False)
    root_dir = Path(__file__).resolve().parents[3]
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, args.run_name)
    out_dirs = prepare_output_dirs(out_dir, disp_dir=True)

    logging.info("Image dir: %s", img_dir)
    logging.info("Calibration file: %s", calib_dict_file)
    logging.info("Output dir: %s", out_dir)

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

    input_resolution = (img_full.shape[1], img_full.shape[0])  # (width, height)
    inference_resolution = (img_small.shape[1], img_small.shape[0])  # (width, height)

    K, baseline_m, calib_dict = load_calibration(
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
        model_ckpt=args.ckpt_dir,
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
        hiera=args.hiera,
        n_warmup=5,
    )

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc="inference"):
        img_number = img_l.get_image_number()

        img0 = img_l.get_small_img(scale=args.scale)
        img1 = img_r.get_small_img(scale=args.scale)

        disp, elapsed = compute_disparity(
            model=model,
            img0=img0,
            img1=img1,
            valid_iters=args.valid_iters,
            hiera=args.hiera,
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