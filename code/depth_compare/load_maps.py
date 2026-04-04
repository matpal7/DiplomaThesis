from pathlib import Path
import logging
import pandas as pd

import cv2
import numpy as np
from tqdm import tqdm

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.compare_depth_between_cameras import warp_depth_to_target, _read_transform, _compute_metrics
# import torch

# from code.depth_compare.evaluation import eval_depth
from code.image import load_rgbd_images
from code.utils import load_estimated_depth_map, scale_intrinsics
from code.visualize_depth import colorize_depth
from prepare_paths import get_depth_estimation_network_names, \
    prepare_depth_comparison_paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

parent_dir = Path(__file__).resolve().parent.parent.parent.parent
date = "27032026"
rgbd_suffix = "zed"



max_imgs = None

(
    gt_data_dir,
    relative_pose_path,
    calib_rgbd_path,
    calib_stereo_path,
    depth_estimation_dir,
    depth_comparison_dir,
) = prepare_depth_comparison_paths(parent_dir, date, rgbd_suffix)

imgs_rgb = load_rgbd_images(gt_data_dir, suffix=rgbd_suffix, max_imgs=max_imgs)


k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
k_target, d_target = load_camera_calibration(calib_rgbd_path)
pose_convention = "cam1_from_cam2"
transform_target_from_source = _read_transform(relative_pose_path, pose_convention)

transform_target_from_source[:3, 3] /= 1000.0
norm = np.linalg.norm(transform_target_from_source[:3, 3])
logger.debug(f"translation norm: {norm}")

def resize_depth_safe(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    valid = np.isfinite(depth) & (depth > 0)

    # resize valid mask and depth separately
    depth_filled = depth.copy()
    depth_filled[~valid] = 0.0

    depth_sum  = cv2.resize(depth_filled.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    valid_sum  = cv2.resize(valid.astype(np.float32),        (w, h), interpolation=cv2.INTER_LINEAR)

    out = np.full((h, w), np.nan, dtype=np.float32)
    valid_out = valid_sum > 0.1   # at least some valid contribution
    out[valid_out] = depth_sum[valid_out] / valid_sum[valid_out]
    return out


estimated_depth_maps = {}
print("DEPTH EST DIR:", depth_estimation_dir)
for nn_name in get_depth_estimation_network_names():
    estimated_depth_map = load_estimated_depth_map(depth_estimation_dir, nn_name, max_imgs=max_imgs)
    estimated_depth_maps[nn_name] = estimated_depth_map
    logger.debug(f"{nn_name}")
    logger.debug(f"Estimated depth mapslegth: {len(estimated_depth_map)}")
    frame_size = estimated_depth_map[0].shape[::-1]
    logger.debug(f"NN frame size: {frame_size}")

metrics_out_dir = depth_comparison_dir / "metrics"
metrics_out_dir.mkdir(parents=True, exist_ok=True)
all_metrics = []

all_networks_summary = []   # one row per (nn_name, aggregated metric)

for nn_name in estimated_depth_maps.keys():
    print(f"\nNetwork: {nn_name}")
    frame_size = estimated_depth_maps[nn_name][0].shape[::-1]
    logger.debug(f"NN frame size: {frame_size}")
    k_source_scaled = scale_intrinsics(k_source.copy(), (3840, 2160), frame_size)
    k_target_scaled = scale_intrinsics(k_target.copy(), (1280, 720), frame_size)

    all_gt  = []
    all_est = []
    per_image_rows = []     # one dict per image for this network

    # ── per-image pass ────────────────────────────────────────────────────────
    for img_gt, depth_est in tqdm(
        zip(imgs_rgb, estimated_depth_maps[nn_name]),
        desc=f"[{nn_name}] evaluating",
    ):
        image_number = img_gt.get_image_number()
        print(img_gt.get_path())
        depth_gt = img_gt.get_depth()

        depth_gt = resize_depth_safe(depth_gt, depth_est.shape)


        valid = (
            np.isfinite(depth_est) & (depth_est > 0) &
            np.isfinite(depth_gt)  & (depth_gt  > 0)
        )

        pred_warped_m, valid_pred = warp_depth_to_target(
            source_depth=depth_est,
            k_source=k_source_scaled,
            d_source=d_source,
            k_target=k_target_scaled,
            d_target=d_target,
            t_target_source=transform_target_from_source,
            source_depth_scale=1.0,
            target_hw=(depth_gt.shape[0], depth_gt.shape[1]),
        )

        vis1 = colorize_depth(depth_gt)
        vis2 = colorize_depth(depth_est)
        # cv2.imshow("depth NN", vis2)
        # cv2.imshow("depth GT", vis1)
        vis1 = cv2.resize(vis1, frame_size)
        vis2 = cv2.resize(vis2, frame_size)
        vis3 = colorize_depth(pred_warped_m)
        vis = cv2.hconcat([vis1, vis2, vis3])

        # cv2.imshow(f"Depth map {rgbd_suffix} | {nn_name}", vis)
        # cv2.waitKey(0)

        # per-image metrics (warped prediction vs GT)
        metrics = _compute_metrics(pred_warped_m, depth_gt, valid_pred)
        per_image_rows.append({"image_id": image_number, "nn_name": nn_name, **metrics})

        if np.any(valid):
            all_gt.append(depth_gt[valid])
            all_est.append(depth_est[valid])

    # ── save per-image CSV for this network ───────────────────────────────────
    df_per_image = pd.DataFrame(per_image_rows)
    per_image_csv = metrics_out_dir / f"{nn_name}_per_image_metrics.csv"
    df_per_image.to_csv(per_image_csv, index=False)
    print(f"  Saved per-image metrics → {per_image_csv}")

    # ── global scale + aggregated metrics ─────────────────────────────────────
    if len(all_gt) == 0:
        print("  No valid overlapping depth values.")
        continue

    all_gt_cat  = np.concatenate(all_gt)
    all_est_cat = np.concatenate(all_est)

    # aggregate per-image metrics (mean ± std across images)
    REPORT_COLS = ["abs_rel", "rmse_m", "median_abs_error_m", "delta_1_25"]

    global_scale = float(np.median(all_gt_cat) / np.median(all_est_cat))
    print(f"  Global scale: {global_scale:.6f}")

    aggregated = {"nn_name": nn_name, "global_scale": global_scale}
    for col in REPORT_COLS:
        aggregated[f"{col}_mean"] = float(df_per_image[col].mean())

    all_networks_summary.append(aggregated)

    # save per-network summary CSV
    df_network_summary = pd.DataFrame([aggregated])
    network_summary_csv = metrics_out_dir / f"{nn_name}_summary_metrics.csv"
    df_network_summary.to_csv(network_summary_csv, index=False)
    print(f"  Saved network summary → {network_summary_csv}")

# ── global summary across ALL networks ───────────────────────────────────────
if all_networks_summary:
    df_global = pd.DataFrame(all_networks_summary)
    global_csv = metrics_out_dir / "all_networks_global_summary.csv"
    df_global.to_csv(global_csv, index=False)
    print(f"\nSaved global summary for all networks → {global_csv}")
    print(df_global.to_string(index=False))
# # --- Load ---
# gt_np   = np.load("gt_depth.npy")
# pred_np = np.load("pred_depth.npy")
#
# # --- Handle uint16 from RGB-D cameras (depth in mm → convert to meters) ---
# if gt_np.dtype == np.uint16:
#     gt_np   = gt_np.astype(np.float32) / 1000.0   # mm → meters
# if pred_np.dtype == np.uint16:
#     pred_np = pred_np.astype(np.float32) / 1000.0
#
# # --- To tensor (B, 1, H, W) ---
# gt   = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0)
# pred = torch.from_numpy(pred_np).float().unsqueeze(0).unsqueeze(0)
#
# # --- Mask ---
# min_depth, max_depth = 0.1, 10.0
# mask = (gt > min_depth) & (gt < max_depth) & torch.isfinite(gt)
#
# # --- Evaluate ---
# results = eval_depth(gt, pred, mask, max_depth=max_depth)
#
# # --- Print per-metric mean ---
# for name, vals in results.items():
#     print(f"{name:15s}: {vals.mean().item():.4f}")