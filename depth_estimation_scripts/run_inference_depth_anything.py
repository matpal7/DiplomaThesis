from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import time as time_module

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from code.depth_compute.depth_utils import save_outputs, load_calibration
from code.depth_compute.run_stats import RunStats, save_run_stats
from code.image import load_l_r_images_rectified
from code.prepare_paths import build_paths, prepare_output_dirs
from code.utils import load_dict, scale_intrinsics
from code.visualize_depth import colorize_depth
from src.depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DepthAnything3 on folders of rectified left/right images."
    )

    parent_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--max_imgs",       type=int,  default=None)
    parser.add_argument("--monocular",      action="store_true")
    parser.add_argument("--date",           type=str,  default="24042026")
    parser.add_argument("--save_depth_vis", action="store_true", default=True)
    parser.add_argument("--show_preview",   action="store_true")
    parser.add_argument("--device",         type=str,  default="cuda")
    parser.add_argument("--scale",          type=int,  default=6)
    return parser.parse_args()


def get_output_model_name(monocular: bool) -> str:
    return "DepthAnything3_mono" if monocular else "DepthAnything3_stereo"

def load_calibration_local(
    calib_dict_path: Path,
    input_resolution: tuple[int, int],
    inference_resolution: tuple[int, int],
    monocular: bool,
) -> tuple[np.ndarray | None, float | None]:

    K_scaled, baseline_m, calib_dict = load_calibration(calib_dict_path, input_resolution, inference_resolution)

    intrinsics = None
    focal = None

    if monocular:
        # Extract focal from scaled left intrinsics for metric conversion
        K_l = np.asarray(calib_dict["new_K_l"], dtype=np.float64).copy()
        K_l = scale_intrinsics(K_l, input_resolution, inference_resolution)
        fx, fy = K_l[0, 0], K_l[1, 1]
        focal = (fx + fy) / 2.0
        logging.info("Monocular focal length (scaled): fx=%.2f fy=%.2f → focal=%.2f", fx, fy, focal)
    else:
        K_l = np.asarray(calib_dict["new_K_l"], dtype=np.float64).copy()
        K_r = np.asarray(calib_dict["new_K_r"], dtype=np.float64).copy()
        K_l = scale_intrinsics(K_l, input_resolution, inference_resolution)
        K_r = scale_intrinsics(K_r, input_resolution, inference_resolution)
        intrinsics = np.stack([K_l, K_r], axis=0)

    return intrinsics, focal


def load_model(model_name: str, device: str) -> DepthAnything3:
    device_t = torch.device(device)

    logging.info("Loading model: %s", model_name)
    model = DepthAnything3.from_pretrained(model_name).to(device_t)

    first_param = next(model.parameters())
    logging.info("Model loaded on device: %s", first_param.device)

    model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    return model

def warmup_model(
    model: DepthAnything3,
    inference_resolution: tuple[int, int],
    monocular: bool,
    intrinsics: np.ndarray | None,
    device: str,
    n_warmup: int = 5,
) -> None:
    mode = "monocular" if monocular else "stereo"
    logging.info(
        "Warming up DepthAnything3 (%s, %d passes, shape=%s)...",
        mode, n_warmup, inference_resolution,
    )
    w, h = inference_resolution
    dummy = np.zeros((h, w, 3), dtype=np.uint8)

    for _ in range(n_warmup):
        if monocular:
            _ = model.inference([dummy])
        else:
            _ = model.inference(
                [dummy, dummy],
                intrinsics=intrinsics,
                extrinsics=None,
                align_to_input_ext_scale=False,
            )

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    logging.info("Warm-up complete (%s).", mode)



def compute_depth_monocular(model, img_left, focal):
    t0 = time_module.perf_counter()
    with torch.inference_mode():
        prediction = model.inference([img_left])
    elapsed = time_module.perf_counter() - t0
    net_output = np.asarray(prediction.depth[0], dtype=np.float32)

    logging.info(
        "net_output stats — min: %.4f  max: %.4f  mean: %.4f",
        np.nanmin(net_output), np.nanmax(net_output), np.nanmean(net_output),
    )

    # Check if already metric (values ~0.3–10m for tabletop scenes)
    # before deciding to apply the formula
    depth = focal * net_output / 300.0

    logging.info(
        "depth after formula — min: %.4f  max: %.4f  mean: %.4f",
        np.nanmin(depth), np.nanmax(depth), np.nanmean(depth),
    )
    return depth, elapsed


def compute_depth_stereo(
    model: DepthAnything3,
    img_left: np.ndarray,
    img_right: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, float]:
    t0 = time_module.perf_counter()
    with torch.inference_mode():
        prediction = model.inference(
        [img_left, img_right],
        intrinsics=intrinsics,
        extrinsics=None,
        align_to_input_ext_scale=False,
    )
    depth = prediction.depth[0]
    return np.asarray(depth, dtype=np.float32), time_module.perf_counter() - t0


def run_depth_anything(args: argparse.Namespace, monocular: bool) -> None:
    torch.autograd.set_grad_enabled(False)

    model_output_name = get_output_model_name(monocular)
    root_dir = Path(__file__).resolve().parents[3]
    img_dir, calib_dict_file, out_dir = build_paths(root_dir, args.date, model_output_name)
    out_dirs = prepare_output_dirs(out_dir)

    logging.info("Running mode: %s", "monocular" if monocular else "stereo")

    calib_dict = load_dict(calib_dict_file)
    model = load_model(args.model, args.device)

    imgs_l, imgs_r = load_l_r_images_rectified(calib_dict, img_dir, max_imgs=args.max_imgs)
    if len(imgs_l) == 0:
        raise RuntimeError("No stereo pairs found from the provided input arguments")

    logging.info("Found %d pairs", len(imgs_l))

    img_full  = imgs_l[0].get_img()
    img_small = imgs_l[0].get_small_img(scale=args.scale)
    input_resolution     = (img_full.shape[1],  img_full.shape[0])
    inference_resolution = (img_small.shape[1], img_small.shape[0])

    # focal is None for stereo (not needed), float for monocular
    intrinsics, focal = load_calibration_local(
        calib_dict_path=calib_dict_file,
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        monocular=monocular,
    )

    stats = RunStats(
        run_name=model_output_name,
        date=args.date,
        model_ckpt=args.model,
        network_type="mono" if monocular else "stereo",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    warmup_model(
        model=model,
        inference_resolution=inference_resolution,
        monocular=monocular,
        intrinsics=intrinsics,
        device=args.device,
        n_warmup=5,
    )

    desc = f"inference ({'mono' if monocular else 'stereo'})"
    for img_l_obj, img_r_obj in tqdm(list(zip(imgs_l, imgs_r)), desc=desc):
        img_number = img_l_obj.get_image_number()
        img_left   = img_l_obj.get_small_img(scale=args.scale)

        if monocular:
            depth, elapsed = compute_depth_monocular(
                model=model,
                img_left=img_left,
                focal=focal,                       # metric conversion applied inside
            )
        else:
            img_right = img_r_obj.get_small_img(scale=args.scale)
            depth, elapsed = compute_depth_stereo(
                model=model,
                img_left=img_left,
                img_right=img_right,
                intrinsics=intrinsics,
            )

        stats.per_image_stats.append({
            "image_id":   img_number,
            "time_s":     round(elapsed, 4),
            "depth_min":  round(float(np.nanmin(depth)), 4),
            "depth_max":  round(float(np.nanmax(depth)), 4),
            "depth_mean": round(float(np.nanmean(depth)), 4),
            "valid_px":   int(np.isfinite(depth).sum()),
            "total_px":   int(depth.size),
        })

        save_outputs(img_number=img_number, img_left=img_left, depth=depth, out_dirs=out_dirs)

        if args.show_preview:
            depth_vis = colorize_depth(depth)
            cv2.imshow(f"Depth {img_number} ({'mono' if monocular else 'stereo'})", depth_vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    save_run_stats(stats, out_dir)
    if args.show_preview:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # run_depth_anything(args, monocular=False)  # stereo
    run_depth_anything(args, monocular=True)  # monocular