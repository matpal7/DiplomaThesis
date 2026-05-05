from __future__ import annotations

import argparse
import logging
import time as time_module
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from code.depth_compute.depth_utils import load_calibration, disparity_to_depth, save_outputs
from code.depth_compute.run_stats import RunStats, save_run_stats
from code.image import load_l_r_images_rectified
from code.utils import load_dict
from code.prepare_paths import build_paths, prepare_output_dirs
from code.visualize_depth import colorize_depth

from s2m2.core.model.s2m2 import S2M2
from s2m2.core.utils.image_utils import image_pad, image_crop


_MODEL_CONFIG: dict[str, dict] = {
    "S":  {"feature_channels": 128, "n_transformer": 1},
    "M":  {"feature_channels": 192, "n_transformer": 2},
    "L":  {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run S2M2 on folders of rectified left/right images."
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default=str(code_dir / "weights" / "pretrain_weights"),
    )
    parser.add_argument("--model_type", type=str, default="L",
                        choices=list(_MODEL_CONFIG))
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--date", type=str, default="24042026")
    parser.add_argument("--min_disp", type=float, default=1e-6)
    parser.add_argument("--run_name", type=str, default="S2M2_stereo")
    parser.add_argument("--use_positivity", action="store_true", default=True)
    parser.add_argument("--refine_iter", type=int, default=3)

    return parser.parse_args()


def load_model(args: argparse.Namespace) -> S2M2:
    config = _MODEL_CONFIG[args.model_type]

    ckpt_name = f'CH{config["feature_channels"]}NTR{config["n_transformer"]}.pth'
    ckpt_path = Path(args.pretrain_dir) / ckpt_name

    model = S2M2(
        feature_channels=config["feature_channels"],
        dim_expansion=1,
        num_transformer=config["n_transformer"],
        use_positivity=args.use_positivity,
        refine_iter=args.refine_iter,
    )

    checkpoint = torch.load(ckpt_path, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.my_load_state_dict(state_dict)
    model.cuda().eval()

    logging.info("Loaded model: %s", ckpt_path)
    return model

def warmup_model(
    model: S2M2,
    inference_resolution: tuple[int, int],
    n_warmup: int = 5,
) -> None:

    logging.info("Warming up model (%d passes)...", n_warmup)
    w, h = inference_resolution

    dummy = torch.zeros(1, 3, h, w, device="cuda", dtype=torch.float32)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for _ in range(n_warmup):
            _ = model(image_pad(dummy, 32), image_pad(dummy, 32))

    torch.cuda.synchronize()
    logging.info("Warm-up complete.")

def compute_disparity(
    model: S2M2,
    img0: np.ndarray,
    img1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h, w = img0.shape[:2]

    left_t  = torch.as_tensor(img0).cuda().float().permute(2, 0, 1).unsqueeze(0)
    right_t = torch.as_tensor(img1).cuda().float().permute(2, 0, 1).unsqueeze(0)

    left_pad  = image_pad(left_t,  32)
    right_pad = image_pad(right_t, 32)

    t0 = time_module.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred_disp, pred_occ, pred_conf = model(left_pad, right_pad)
    elapsed = time_module.perf_counter() - t0

    pred_disp = image_crop(pred_disp, (h, w)).squeeze().float().cpu().numpy()
    pred_occ  = image_crop(pred_occ,  (h, w)).squeeze().float().cpu().numpy()
    pred_conf = image_crop(pred_conf, (h, w)).squeeze().float().cpu().numpy()

    return (
        pred_disp.astype(np.float32),
        pred_occ.astype(np.float32),
        pred_conf.astype(np.float32),
        elapsed,
    )




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
    logging.info("Pretrain dir: %s", args.pretrain_dir)
    logging.info("Model type: %s", args.model_type)

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

    img_full  = imgs_l[0].get_img()
    img_small = imgs_l[0].get_small_img(scale=args.scale)

    input_resolution     = (img_full.shape[1],  img_full.shape[0])
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
        model_ckpt=args.pretrain_dir,
        network_type="stereo",
        input_resolution=input_resolution,
        inference_resolution=inference_resolution,
        scale=args.scale,
        n_images=len(imgs_l),
    )

    warmup_model(model, inference_resolution, n_warmup=5)

    for img_l, img_r in tqdm(list(zip(imgs_l, imgs_r)), desc="s2m2 inference"):
        img_number = img_l.get_image_number()

        img0 = img_l.get_small_img(scale=args.scale)
        img1 = img_r.get_small_img(scale=args.scale)

        disp, occ, conf, elapsed = compute_disparity(
            model=model,
            img0=img0,
            img1=img1,
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