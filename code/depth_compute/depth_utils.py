import logging
from pathlib import Path

import cv2
import numpy as np

from code.utils import load_dict, scale_intrinsics
from code.visualize_depth import colorize_depth


def load_calibration(calib_dict_file: Path, input_resolution: tuple[int, int], inference_resolution: tuple[int, int]) -> tuple[np.ndarray, float, dict]:
    calib_dict = load_dict(calib_dict_file)

    K = np.asarray(calib_dict["new_K_l"], dtype=np.float64)
    tvec = np.asarray(calib_dict["tvec"], dtype=np.float64).reshape(-1)

    baseline_m = float(np.linalg.norm(tvec)) / 1000.0
    K_scaled = scale_intrinsics(K, input_resolution, inference_resolution)

    return K_scaled, baseline_m, calib_dict

def disparity_to_depth(
    disp: np.ndarray,
    fx: float,
    baseline_m: float,
    min_disp: float = 1e-6,
) -> np.ndarray:
    depth = np.full_like(disp, np.nan, dtype=np.float32)

    valid = np.isfinite(disp) & (disp > min_disp)
    depth[valid] = (fx * baseline_m) / disp[valid]

    return depth

def save_outputs(
    img_number: str,
    img_left: np.ndarray,
    depth: np.ndarray,
    out_dirs: dict[str, Path],
    disp: np.ndarray = None,
) -> None:
    depth_path = out_dirs["depth"] / f"{img_number}_depth.npy"
    depth_vis  = colorize_depth(depth)

    # match sizes — resize img_left to depth_vis if they differ
    h_vis, w_vis = depth_vis.shape[:2]
    h_img, w_img = img_left.shape[:2]
    if (h_img, w_img) != (h_vis, w_vis):
        target_h = min(h_vis, h_img)
        target_w = min(w_vis, w_img)
        if (h_img, w_img) != (target_h, target_w):
            img_left  = cv2.resize(img_left,  (target_w, target_h), interpolation=cv2.INTER_AREA)
        if (h_vis, w_vis) != (target_h, target_w):
            depth_vis = cv2.resize(depth_vis, (target_w, target_h), interpolation=cv2.INTER_AREA)

    preview = cv2.hconcat([img_left, depth_vis])

    np.save(depth_path, depth)
    # cv2.imwrite(str(out_dirs["vis"] / f"{img_number}_vis.png"),     depth_vis)
    cv2.imwrite(str(out_dirs["vis"] / f"{img_number}_preview.png"), preview)
    # logging.info("Saved depth: %s", depth_path)

    if disp is not None:
        disp_path = out_dirs["disp"] / f"{img_number}_disp.npy"
        np.save(disp_path, disp)
        # logging.info("Saved disparity: %s", disp_path)