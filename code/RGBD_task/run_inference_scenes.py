import argparse
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import cv2

# import trimesh, imageio
# import nvdiffrast.torch as dr
# from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
# from datareader import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis

# ─── Resolution constants ─────────────────────────────────────────────────────
SOURCE_CALIB_HW = (2160, 3840)  # (H, W) — ZED Mini left calibration resolution
DEPTH_HW        = (360,  640)   # (H, W) — NN depth map native resolution
RGBD_HW         = (720,  1280)  # (H, W) — RealSense RGB native resolution


# ─── Diagnostic helpers ───────────────────────────────────────────────────────

def _sep(label: str) -> None:
    print("\n" + "─" * 60)
    print("  {}".format(label))
    print("─" * 60)


def print_depth_stats(depth: np.ndarray, label: str) -> None:
    """Print shape, dtype, value range and unit-scale sanity for a depth map."""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    _sep("Depth stats — {}".format(label))
    print("  shape     : {}".format(depth.shape))
    print("  dtype     : {}".format(depth.dtype))
    print("  valid px  : {} / {}  ({:.1f}%)".format(
        valid.size, depth.size, 100.0 * valid.size / depth.size))
    if valid.size > 0:
        print("  min       : {:.4f}".format(valid.min()))
        print("  max       : {:.4f}".format(valid.max()))
        print("  mean      : {:.4f}".format(valid.mean()))
        print("  median    : {:.4f}".format(np.median(valid)))
        # Unit-scale heuristic
        if valid.max() > 100.0:
            print("  ⚠  Values look like MILLIMETRES (max > 100). Expected METRES.")
        elif valid.max() < 0.01:
            print("  ⚠  Values look unusually small (max < 0.01 m). Check depth_scale.")
        else:
            print("  ✓  Scale looks like METRES.")
    else:
        print("  ⚠  No valid depth pixels!")


def print_K(K: np.ndarray, label: str) -> None:
    """Pretty-print a 3x3 intrinsic matrix with a label."""
    _sep("K — {}".format(label))
    print("  fx={:.2f}  fy={:.2f}  cx={:.2f}  cy={:.2f}".format(
        K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print("  full matrix:\n{}".format(
        "\n".join("    " + "  ".join("{:10.4f}".format(v) for v in row)
                  for row in K)))


def print_warp_stats(depth_before: np.ndarray,
                     depth_after: np.ndarray,
                     valid: np.ndarray) -> None:
    """Compare depth maps before and after warping for scale/value consistency."""
    _sep("Warp consistency check")
    valid_before = depth_before[np.isfinite(depth_before) & (depth_before > 0)]
    # FIX: depth_after is float32 metres here; use valid mask directly
    valid_after  = depth_after[valid & (depth_after > 0)]

    print("  BEFORE warp  shape={} valid={:.1f}%  range=[{:.3f}, {:.3f}] m".format(
        depth_before.shape,
        100.0 * valid_before.size / depth_before.size,
        valid_before.min() if valid_before.size else 0.0,
        valid_before.max() if valid_before.size else 0.0))

    print("  AFTER  warp  shape={} valid={:.1f}%  range=[{:.3f}, {:.3f}] m".format(
        depth_after.shape,
        100.0 * valid_after.size  / depth_after.size,
        valid_after.min()  if valid_after.size  else 0.0,
        valid_after.max()  if valid_after.size  else 0.0))

    if valid_before.size > 0 and valid_after.size > 0:
        scale_ratio = valid_after.mean() / valid_before.mean()
        print("  mean before : {:.4f} m".format(valid_before.mean()))
        print("  mean after  : {:.4f} m".format(valid_after.mean()))
        print("  mean ratio  : {:.4f}  (expected ≈1.0 if same unit)".format(scale_ratio))
        if abs(scale_ratio - 1.0) > 0.5:
            print("  ⚠  Ratio far from 1.0 — possible unit mismatch or bad extrinsic.")
        else:
            print("  ✓  Scale consistent before/after warp.")


def print_color_depth_alignment(color: np.ndarray,
                                depth: np.ndarray,
                                K: np.ndarray) -> None:
    """Confirm that colour and depth share the same spatial resolution and K."""
    _sep("Colour / depth alignment check")
    print("  color shape : {}  (H={} W={})".format(
        color.shape, color.shape[0], color.shape[1]))
    print("  depth shape : {}  (H={} W={})".format(
        depth.shape, depth.shape[0], depth.shape[1]))
    if color.shape[:2] == depth.shape[:2]:
        print("  ✓  Shapes match.")
    else:
        print("  ✗  SHAPE MISMATCH — FoundationPose will fail.")
    print("  K used      : fx={:.2f}  fy={:.2f}  cx={:.2f}  cy={:.2f}".format(
        K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    # Sanity: cx/cy should be near image centre
    exp_cx = color.shape[1] / 2.0
    exp_cy = color.shape[0] / 2.0
    if abs(K[0, 2] - exp_cx) > exp_cx * 0.3:
        print("  ⚠  cx={:.1f} far from image centre ({:.1f}). Check K scaling.".format(
            K[0, 2], exp_cx))
    else:
        print("  ✓  cx reasonable for image width {}.".format(color.shape[1]))
    if abs(K[1, 2] - exp_cy) > exp_cy * 0.3:
        print("  ⚠  cy={:.1f} far from image centre ({:.1f}). Check K scaling.".format(
            K[1, 2], exp_cy))
    else:
        print("  ✓  cy reasonable for image height {}.".format(color.shape[0]))


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def scale_intrinsics(K: np.ndarray,
                     old_hw: Tuple[int, int],
                     new_hw: Tuple[int, int]) -> np.ndarray:
    sy = new_hw[0] / float(old_hw[0])
    sx = new_hw[1] / float(old_hw[1])
    K_new = K.copy().astype(np.float64)
    K_new[0, 0] *= sx; K_new[1, 1] *= sy
    K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new


def _back_project_depth(depth: np.ndarray, K: np.ndarray, D: np.ndarray,
                         depth_scale: float = 1.0) -> np.ndarray:
    ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    uv   = np.stack([xs.astype(np.float64),
                     ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, K, D).reshape(-1, 2)
    z    = depth[ys, xs].astype(np.float64) * depth_scale
    return np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z])


def _splat_depth(uv: np.ndarray, z: np.ndarray,
                 target_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w      = target_hw
    projected = np.full((h, w), np.inf, dtype=np.float64)
    keep      = (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
                (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, z = uv[keep], z[keep]
    if len(z) == 0:
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=bool)
    u0 = np.floor(uv[:, 0]).astype(np.int32)
    v0 = np.floor(uv[:, 1]).astype(np.int32)
    for du, dv in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        uu, vv = u0 + du, v0 + dv
        inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        np.minimum.at(projected, (vv[inside], uu[inside]), z[inside])
    valid = np.isfinite(projected)
    projected[~valid] = 0.0
    return projected, valid


def _fill_depth_holes(depth: np.ndarray, valid: np.ndarray,
                      max_radius: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill small holes in the splatted depth map using a nearest-neighbour
    morphological dilation over invalid (hole) pixels.  Only fills pixels
    within ``max_radius`` pixels of a valid neighbour to avoid large
    extrapolation across object boundaries.
    """
    filled = depth.copy()
    filled_valid = valid.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max_radius + 1, 2 * max_radius + 1))

    # Dilate valid mask to find which holes are close to valid pixels
    valid_u8   = valid.astype(np.uint8) * 255
    dilated    = cv2.dilate(valid_u8, kernel)
    hole_mask  = (dilated > 0) & (~valid)          # pixels to fill

    if not hole_mask.any():
        return filled, filled_valid

    # For each hole pixel take the median of valid pixels in a small window
    # (simple but effective; for production consider inpaint or bilateral fill)
    depth_for_fill = depth.copy()
    depth_for_fill[~valid] = np.nan

    # Use a box filter on valid values only
    sum_map   = cv2.boxFilter(np.where(valid, depth, 0).astype(np.float64),
                              ddepth=-1, ksize=(2 * max_radius + 1, 2 * max_radius + 1),
                              normalize=False)
    count_map = cv2.boxFilter(valid.astype(np.float64),
                              ddepth=-1, ksize=(2 * max_radius + 1, 2 * max_radius + 1),
                              normalize=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_map = np.where(count_map > 0, sum_map / count_map, 0.0)

    filled[hole_mask]       = mean_map[hole_mask]
    filled_valid[hole_mask] = True
    return filled, filled_valid


def warp_depth_to_target(
        source_depth: np.ndarray,
        k_source: np.ndarray, d_source: np.ndarray,
        k_target: np.ndarray, d_target: np.ndarray,
        t_target_from_source: np.ndarray,
        target_hw: Tuple[int, int],
        depth_scale: float = 1.0,
        fill_holes: bool = True,
        fill_radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp source_depth (in source camera frame) into the target camera frame.

    Parameters
    ----------
    source_depth        : float32/float64 depth map in METRES
    k_source / d_source : source camera intrinsics / distortion
    k_target / d_target : target camera intrinsics / distortion
    t_target_from_source: 4×4 rigid transform — points expressed in the
                          source frame are multiplied on the right to get
                          them in the target frame  (p_tgt = T @ p_src_h)
    target_hw           : (H, W) of the output map
    depth_scale         : multiply raw depth values before processing
                          (use 1.0 if already in metres)
    fill_holes          : whether to fill small forward-splat holes
    fill_radius         : morphological fill kernel half-size in pixels

    Returns
    -------
    warped_depth : float32 depth map in METRES, zeros where invalid
    valid_mask   : bool mask of valid (filled) pixels
    """
    h, w    = target_hw
    pts_src = _back_project_depth(source_depth, k_source, d_source, depth_scale)
    if pts_src.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=bool)

    pts_h   = np.hstack([pts_src, np.ones((pts_src.shape[0], 1))])
    pts_tgt = (t_target_from_source @ pts_h.T).T[:, :3]

    front   = pts_tgt[:, 2] > 0
    pts_tgt = pts_tgt[front]
    if pts_tgt.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=bool)

    uv_proj, _ = cv2.projectPoints(
        pts_tgt.reshape(-1, 1, 3),
        np.zeros((3, 1)), np.zeros((3, 1)),
        k_target, d_target,
    )

    depth_map, valid = _splat_depth(uv_proj.reshape(-1, 2), pts_tgt[:, 2], target_hw)

    if fill_holes:
        depth_map, valid = _fill_depth_holes(depth_map, valid, max_radius=fill_radius)

    # Always return float32 metres — never apply mm conversion here
    return depth_map.astype(np.float32), valid


def depth_to_png(depth_m: np.ndarray, out_path: Path,
                 save_16bit: bool = True) -> None:
    """
    Save a float32 depth map (in METRES) as a PNG.

    FIX: previous version returned the array instead of saving it,
         and ignored out_path entirely.

    Parameters
    ----------
    depth_m   : float32 depth in METRES
    out_path  : destination path (must end in .png)
    save_16bit: if True save as uint16 (millimetres, max ~65 m);
                if False save as 8-bit visualisation (0–10 m mapped to 0–255)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if save_16bit:
        img = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    else:
        img = (np.clip(depth_m / 10.0, 0.0, 1.0) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imwrite(str(out_path), img)


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _read_transform(path: Path, direction: str = "cam2_from_cam1") -> np.ndarray:
    """
    Read a 4×4 rigid transform from an OpenCV FileStorage YAML/XML.

    Convention note
    ---------------
    The node name "T_cam2_cam1" is ambiguous in the wild — some tools store
    it as "cam2 expressed in cam1 frame" (i.e. the transform that takes
    points FROM cam1 TO cam2), others invert this convention.

    This function reads both nodes and returns whichever is requested via
    ``direction``.  After loading, always verify the translation magnitude
    and sign against the known physical camera baseline with print statements
    (see main block below).
    """
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Cannot open transform file: {}".format(path))
    try:
        t21 = fs.getNode("T_cam2_cam1").mat()
        t12 = fs.getNode("T_cam1_cam2").mat()
    finally:
        fs.release()
    if t21 is None and t12 is None:
        raise ValueError("No transform found in {}".format(path))
    t21 = np.asarray(t21, dtype=np.float64) if t21 is not None else None
    t12 = np.asarray(t12, dtype=np.float64) if t12 is not None else None

    if direction == "cam2_from_cam1":
        result = t21 if t21 is not None else np.linalg.inv(t12)
    elif direction == "cam1_from_cam2":
        result = t12 if t12 is not None else np.linalg.inv(t21)
    else:
        raise ValueError("Unsupported direction: {}".format(direction))

    # Verify the matrix is a valid rigid transform
    R = result[:3, :3]
    det = np.linalg.det(R)
    if abs(det - 1.0) > 0.01:
        logging.warning("Rotation matrix determinant = %.4f (expected 1.0). "
                        "Transform may be corrupt.", det)
    return result


def _load_yaml_calibration(yaml_path: Path) -> dict:
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Cannot open: {}".format(yaml_path))
    def _mat(n):  node = fs.getNode(n); return node.mat() if not node.empty() else None
    def _real(n): node = fs.getNode(n); return float(node.real()) if not node.empty() else None
    def _int(n):  v = _real(n); return int(v) if v is not None else None
    try:
        data = {"K": _mat("K"), "D": _mat("D"), "K_new": _mat("K_new"),
                "D_new": _mat("D_new"), "image_width": _int("image_width"),
                "image_height": _int("image_height"),
                "reprojection_error": _real("reprojection_error")}
    finally:
        fs.release()
    if data["K"] is None or data["D"] is None:
        raise ValueError("Missing K or D in {}".format(yaml_path))
    data["K"] = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    data["D"] = np.asarray(data["D"], dtype=np.float64)
    return data


def load_camera_calibration(path: Path,
                             suffix: str = "left") -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        calib = _load_yaml_calibration(path)
        K = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(calib["D"], dtype=np.float64).reshape(-1, 1)
        return K, D
    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            calib = json.load(f)
        key   = "new_K_l" if suffix == "left" else "new_K_r"
        K     = np.asarray(calib[key], dtype=np.float64).reshape(3, 3)
        D_key = "D_l" if suffix == "left" else "D_r"
        D     = np.asarray(calib[D_key], dtype=np.float64).reshape(-1, 1) \
                if D_key in calib else np.zeros((5, 1), dtype=np.float64)
        return K, D
    raise ValueError("Unsupported calibration format: {}".format(path))


def load_k_matrix(yaml_path: str, node: str = "K") -> np.ndarray:
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Cannot open: {}".format(yaml_path))
    K = fs.getNode(node).mat()
    fs.release()
    if K is None:
        raise ValueError("Node '{}' not found in {}".format(node, yaml_path))
    return K


def load_sorted_files(directory: Path, folder: str,
                      pattern: str = "*.png") -> List[Path]:
    files = list((directory / folder).glob(pattern))
    return sorted(files, key=lambda p: int(p.stem.split("_")[0]))


def load_frame(color_path: Path,
               depth_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(color_path))
    if bgr is None:
        raise FileNotFoundError("Cannot load colour image: {}".format(color_path))
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(str(depth_path)).astype(np.float32)
    return color, depth


def resize_color_to_depth(color: np.ndarray, depth: np.ndarray,
                           K_color: np.ndarray,
                           color_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    depth_h, depth_w = depth.shape[:2]
    color_ds = cv2.resize(color, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
    K_ds     = scale_intrinsics(K_color,
                                old_hw=(color_hw[0], color_hw[1]),
                                new_hw=(depth_h, depth_w))
    return color_ds, K_ds


def resize_mask_to_depth(mask: np.ndarray,
                          depth_hw: Tuple[int, int]) -> np.ndarray:
    h, w = depth_hw
    if mask.shape[:2] == (h, w):
        return mask.copy()
    return cv2.resize(mask.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)


def make_depth_vis(depth_m: np.ndarray, max_depth_m: float = 5.0) -> np.ndarray:
    """
    Visualise a float32 depth map (in METRES).
    FIX: caller must pass depth in metres — do NOT pass the uint16 mm map here.
    """
    vis = (np.clip(depth_m / max_depth_m, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_refine_iter",       type=int, default=5)
    parser.add_argument("--track_refine_iter",     type=int, default=2)
    parser.add_argument("--debug",                 type=int, default=0)
    parser.add_argument("--frame_step",            type=int, default=1)
    parser.add_argument("--save_vis",              action="store_true", default=True)
    parser.add_argument("--save_warped_depth",     action="store_true", default=True)
    parser.add_argument("--save_warped_depth_vis", action="store_true", default=True)
    parser.add_argument("--use_depth_from_rgbd",   action="store_true", default=False)
    parser.add_argument("--rgbd_camera",           type=str, default="zed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Program started")

    # ── Paths ──────────────────────────────────────────────────────────────────
    code_dir          = Path(__file__).resolve().parents[2]
    base_dir          = code_dir.parent
    date              = "24042026"
    scene_dir_main    = base_dir / "datasets" / "dataset_{}".format(date) / "downstream_task"
    depth_dir_main    = base_dir / "out" / "out_{}".format(date) / "pose_estimation_depths"
    out_dir           = base_dir / "out" / "out_{}".format(date)
    masks_dir         = out_dir / "masks"
    models_dir        = out_dir / "3D_models"
    camera_params_dir = out_dir / "cameras_parameters"
    calib_rgbd_path   = camera_params_dir / \
                        "{}_calibration_1280x720.yaml".format(args.rgbd_camera)

    folder_depth  = "depth_vis/rgbd" if args.use_depth_from_rgbd else "depth"
    pattern_depth = "*_{}.png".format(args.rgbd_camera) \
                    if args.use_depth_from_rgbd else "*.npy"

    # ── Camera intrinsics ──────────────────────────────────────────────────────
    K_full = load_k_matrix(str(calib_rgbd_path))
    print_K(K_full, "RealSense full res {}x{}".format(RGBD_HW[1], RGBD_HW[0]))

    k_source_scaled = d_source = k_target_scaled = \
        d_target = t_target_from_source = None

    subdir_name = "rgbd"
    if not args.use_depth_from_rgbd:
        subdir_name        = "nn"
        calib_stereo_path  = camera_params_dir / "calib_data.json"
        relative_pose_path = camera_params_dir / "relative_pose" / \
                             "relative_pose_{}_to_left_v3.yaml".format(args.rgbd_camera)

        k_source_full, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
        k_source_scaled = scale_intrinsics(k_source_full,
                                           old_hw=SOURCE_CALIB_HW, new_hw=DEPTH_HW)

        k_rgbd_full, d_target = load_camera_calibration(calib_rgbd_path)
        k_target_scaled = scale_intrinsics(k_rgbd_full,
                                           old_hw=RGBD_HW, new_hw=DEPTH_HW)

        # ── Load extrinsic ────────────────────────────────────────────────────
        # IMPORTANT: verify the direction convention in your YAML file.
        # "T_cam2_cam1" in some tools means "from cam1 to cam2" (what we want
        # to go from ZED-left → RealSense).  Print t and R below and cross-check
        # with the known physical baseline (ZED Mini baseline ≈ 63 mm;
        # ZED-to-RealSense distance depends on your rig — typically a few cm).
        t_target_from_source = _read_transform(relative_pose_path,
                                               direction="cam2_from_cam1")

        t_norm = np.abs(t_target_from_source[:3, 3]).max()
        if t_norm > 10.0:
            logging.warning("Translation > 10 m — assuming millimetres, dividing by 1000.")
            t_target_from_source[:3, 3] /= 1000.0
        elif t_norm < 1e-6:
            logging.warning("Translation is essentially zero — "
                            "check if the correct YAML node was read.")

        print_K(k_source_full,    "ZED left — native SOURCE_CALIB_HW {}x{}".format(
            SOURCE_CALIB_HW[1], SOURCE_CALIB_HW[0]))
        print_K(k_source_scaled,  "ZED left — scaled to DEPTH_HW {}x{}".format(
            DEPTH_HW[1], DEPTH_HW[0]))
        print_K(k_target_scaled,  "RealSense — scaled to DEPTH_HW {}x{}".format(
            DEPTH_HW[1], DEPTH_HW[0]))

        _sep("Extrinsic T_realsense_from_zed_left")
        print("  R:\n{}".format(t_target_from_source[:3, :3]))
        print("  t (m): {}  (norm={:.4f} m)".format(
            t_target_from_source[:3, 3],
            np.linalg.norm(t_target_from_source[:3, 3])))

    # K_pose: RealSense intrinsics scaled to DEPTH_HW — used by FoundationPose
    K_pose = scale_intrinsics(K_full, old_hw=RGBD_HW, new_hw=DEPTH_HW)
    print_K(K_pose, "K_pose for FoundationPose — scaled to DEPTH_HW {}x{}".format(
        DEPTH_HW[1], DEPTH_HW[0]))

    # ── Scenes ─────────────────────────────────────────────────────────────────
    scenes = sorted(
        [d.name for d in scene_dir_main.iterdir() if d.is_dir()],
        key=lambda x: int(x.split("_")[-1]),
    )
    objects_name = [
        "chips_box", "apple", "lemon", "orange",
        "rubiks_cube", "scissors", "wood_block",
    ]

    # ── Per-scene loop ─────────────────────────────────────────────────────────
    for scene in scenes[:1]:
        scene_dir   = scene_dir_main / scene
        masks_scene = masks_dir / scene

        color_paths = load_sorted_files(scene_dir, "rgb",
                                        pattern="*_{}.png".format(args.rgbd_camera))
        depth_paths = load_sorted_files(depth_dir_main / scene, folder_depth,
                                        pattern=pattern_depth)

        assert len(color_paths) == len(depth_paths), \
            "Colour/depth count mismatch: {} vs {}".format(
                len(color_paths), len(depth_paths))
        logging.info("Scene '%s': %d frames", scene, len(color_paths))

        objects_current = [f.name for f in masks_scene.iterdir() if f.is_dir()]

        # ── Per-object loop ────────────────────────────────────────────────────
        for object_name in objects_name[:1]:
            if object_name not in objects_current:
                logging.warning("Object '%s' not in scene '%s', skipping.",
                                object_name, scene)
                continue

            mask_path = masks_scene / object_name / \
                        "000_{}.png".format(args.rgbd_camera)
            # mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # if mask_full is None:
            #     raise FileNotFoundError("Mask not found: {}".format(mask_path))
            # mask_full = resize_mask_to_depth(mask_full, DEPTH_HW)

            warped_depth_dir     = out_dir / "warped_depth"     / scene / object_name
            warped_depth_vis_dir = out_dir / "warped_depth_vis" / scene / object_name
            if args.save_warped_depth:
                warped_depth_dir.mkdir(parents=True, exist_ok=True)
            if args.save_warped_depth_vis:
                warped_depth_vis_dir.mkdir(parents=True, exist_ok=True)

            # ── Frame loop ─────────────────────────────────────────────────────
            for i in range(0, len(color_paths), args.frame_step):
                logging.info("Frame %d/%d", i, len(color_paths) - 1)

                color_full, depth_full = load_frame(color_paths[i], depth_paths[i])
                depth_before_warp      = depth_full.copy()  # float32, metres

                # ── Print raw loaded depth stats (frame 0 only to avoid spam) ──
                if i == 0:
                    print_depth_stats(depth_full, "raw NN depth — frame 0")

                if not args.use_depth_from_rgbd:
                    # warp_depth_to_target returns float32 METRES — do NOT
                    # multiply by 1000 here; that conversion belongs only in
                    # depth_to_png() when saving to disk.
                    warped_depth, valid_warp = warp_depth_to_target(
                        source_depth         = depth_full,
                        k_source             = k_source_scaled,
                        d_source             = d_source,
                        k_target             = k_target_scaled,
                        d_target             = d_target,
                        t_target_from_source = t_target_from_source,
                        target_hw            = DEPTH_HW,
                        depth_scale          = 1.0,
                        fill_holes           = True,
                        fill_radius          = 3,
                    )
                    # Keep depth in float32 METRES for FoundationPose
                    depth_full = warped_depth  # float32, metres
                    depth_full[~valid_warp] = 0.0

                    # ── Warp diagnostic (frame 0 only) ─────────────────────────
                    if i == 0:
                        print_depth_stats(depth_full, "warped depth — frame 0")
                        print_warp_stats(depth_before_warp, depth_full, valid_warp)

                    # ── Save warped depth PNGs ─────────────────────────────────
                    # FIX: depth_to_png now actually writes to disk and uses the
                    #      correct float32-metre input (no double conversion).
                    stem = depth_paths[i].stem
                    if args.save_warped_depth:
                        depth_to_png(depth_full,
                                     warped_depth_dir / "{}_warped.png".format(stem),
                                     save_16bit=True)
                    if args.save_warped_depth_vis:
                        depth_to_png(depth_full,
                                     warped_depth_vis_dir / "{}_warped_vis.png".format(stem),
                                     save_16bit=False)

                # ── Downscale RGB to DEPTH_HW ──────────────────────────────────
                # depth_full is float32 metres at DEPTH_HW resolution
                color, K = resize_color_to_depth(color_full, depth_full,
                                                 K_pose, color_hw=RGBD_HW)
                depth = depth_full   # float32, metres — ready for FoundationPose
                mask  = None         # mask_full if i == 0 else None

                # ── Alignment check (frame 0 only) ─────────────────────────────
                if i == 0:
                    print_color_depth_alignment(color, depth, K)

                assert color.shape[:2] == depth.shape[:2], \
                    "Shape mismatch: color={} depth={}".format(
                        color.shape[:2], depth.shape[:2])

                # ── FoundationPose (uncomment when ready) ──────────────────────
                # NOTE: FoundationPose expects:
                #   rgb   — uint8 H×W×3 or float32 H×W×3 in [0,1]
                #   depth — float32 H×W in METRES (NOT millimetres, NOT uint16)
                #   K     — 3×3 float64 intrinsic matrix matching rgb/depth size
                #
                # if i == 0:
                #     pose = est.register(K=K, rgb=color, depth=depth,
                #                         ob_mask=mask, iteration=args.est_refine_iter)
                # else:
                #     pose = est.track_one(rgb=color, depth=depth,
                #                          K=K, iteration=args.track_refine_iter)
                # np.savetxt(str(debug_dir / "ob_in_cam" / "{:03d}.txt".format(i)),
                #            pose.reshape(4, 4))
                # if args.save_vis:
                #     center_pose = pose @ np.linalg.inv(to_origin)
                #     vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                #     vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                #                         thickness=3, transparency=0, is_input_rgb=True)
                #     imageio.imwrite(str(debug_dir / "track_vis" / "{:03d}.png".format(i)), vis)

                # ── Visualisation ──────────────────────────────────────────────
                # FIX: make_depth_vis expects float32 METRES — pass depth_full
                #      (metres), never the uint16 mm version.
                rgb_vis          = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                depth_before_vis = make_depth_vis(depth_before_warp)   # metres ✓
                depth_after_vis  = make_depth_vis(depth)                # metres ✓
                mask_vis         = (mask > 0).astype(np.uint8) * 255 \
                                   if mask is not None \
                                   else np.zeros(depth.shape[:2], dtype=np.uint8)

                cv2.imshow("RGB (downscaled to depth res)", rgb_vis)
                cv2.imshow("Depth Before Warp",             depth_before_vis)
                if not args.use_depth_from_rgbd:
                    cv2.imshow("Depth After Warp",          depth_after_vis)
                cv2.imshow("Mask",                          mask_vis)

                key = cv2.waitKey(0)
                if key == 27:
                    logging.info("ESC pressed — exiting.")
                    cv2.destroyAllWindows()
                    break

    cv2.destroyAllWindows()