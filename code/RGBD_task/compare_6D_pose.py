"""
compare_pose.py
---------------
Compare FoundationPose 6-D pose estimates from two cameras that observed
the same scene simultaneously.

Workflow
--------
1.  Load the 6-D pose T_obj_in_cam (4x4) for every frame from both cameras.
2.  Express the object-origin (or any user-supplied point in object space)
    in each camera's coordinate system.
3.  Transfer the left-camera 3-D point into the RGBD camera coordinate
    system via the known relative extrinsic T_rgbd_from_left.
4.  Compare the transferred point with the RGBD-direct point:
      - 3-D Euclidean distance [m]
      - 2-D pixel reprojection error [px]  (both points projected into
        the RGBD camera at FoundationPose inference resolution)

Intrinsic scaling
-----------------
FoundationPose downscales the input images before inference.
The calibrated intrinsics (K) must therefore be rescaled to match
the actual inference resolution used during pose estimation.

    left : calibrated at 3840x2160  →  inference at  576x324
    rgbd : calibrated at 1280x720   →  inference at  640x360
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.prepare_paths import prepare_depth_comparison_paths


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def scale_intrinsics(k: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    """
    Rescale a 3x3 camera matrix K from one image resolution to another.

    Parameters
    ----------
    k        : 3x3 intrinsic matrix calibrated at `from_wh` resolution
    from_wh  : (width, height) at which K was calibrated
    to_wh    : (width, height) at which the model actually ran

    Returns
    -------
    K_scaled : 3x3 intrinsic matrix valid at `to_wh` resolution
    """
    sx = to_wh[0] / from_wh[0]
    sy = to_wh[1] / from_wh[1]
    k_out = k.copy().astype(np.float64)
    k_out[0, 0] *= sx   # fx
    k_out[1, 1] *= sy   # fy
    k_out[0, 2] *= sx   # cx
    k_out[1, 2] *= sy   # cy
    return k_out


def read_pose(path: Path) -> np.ndarray:
    """Load a 4x4 homogeneous pose matrix from a text file."""
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose in {path}, got {pose.shape}")
    return pose


def read_transform_yaml(path: Path, direction: str = "cam2_from_cam1") -> np.ndarray:
    """
    Read a 4x4 relative-pose transform from an OpenCV FileStorage YAML/XML.

    The file should contain at least one of:
        T_cam2_cam1  (RGBD expressed in left-camera frame → RGBD-from-left)
        T_cam1_cam2  (left expressed in RGBD frame)

    Parameters
    ----------
    direction : "cam2_from_cam1"  →  T that maps points from cam1 to cam2
                "cam1_from_cam2"  →  inverse
    """
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open transform file: {path}")
    try:
        node_21 = fs.getNode("T_cam2_cam1").mat()
        node_12 = fs.getNode("T_cam1_cam2").mat()
    finally:
        fs.release()

    if node_21 is None and node_12 is None:
        raise ValueError(f"Neither T_cam2_cam1 nor T_cam1_cam2 found in {path}")

    t21 = np.asarray(node_21, dtype=np.float64) if node_21 is not None else None
    t12 = np.asarray(node_12, dtype=np.float64) if node_12 is not None else None

    if direction == "cam2_from_cam1":
        return t21 if t21 is not None else np.linalg.inv(t12)
    if direction == "cam1_from_cam2":
        return t12 if t12 is not None else np.linalg.inv(t21)
    raise ValueError(f"Unknown direction '{direction}'")


def project_point(point_3d: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    """
    Project a single 3-D point (in camera space) to 2-D pixel coordinates.

    Uses the scaled K matrix directly (no extra distortion — see NOTE below).

    NOTE: We intentionally pass zero distortion here because the calibrated K
    that is saved with suffix K_new / K_optimal is already the undistorted
    optimal matrix.  Feeding the original D alongside K_new would apply
    distortion twice and produce wrong pixel coordinates.
    """
    d_zero = np.zeros((1, 5), dtype=np.float64)
    uv, _ = cv2.projectPoints(
        point_3d.reshape(1, 1, 3),
        np.zeros((3, 1)), np.zeros((3, 1)),
        k, d_zero,
    )
    return float(uv[0, 0, 0]), float(uv[0, 0, 1])


def collect_pose_files(results_root: Path, camera: str, scene: str, obj: str) -> dict[str, Path]:
    """Return {frame_stem: path} for all pose .txt files of one camera/scene/object."""
    pose_dir = results_root / camera / scene / obj / "ob_in_cam"
    if not pose_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")
    return {p.stem: p for p in sorted(pose_dir.glob("*.txt"))}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved → {path}")


def _make_summary_row(scene: str, obj: str, rows: list[dict]) -> dict:
    """Compute summary statistics for one (scene, object) group."""
    err3 = np.array([r["err_3d_m"]  for r in rows])
    err2 = np.array([r["err_2d_px"] for r in rows])
    dx   = np.array([r["diff_x_m"]  for r in rows])
    dy   = np.array([r["diff_y_m"]  for r in rows])
    dz   = np.array([r["diff_z_m"]  for r in rows])

    bias      = np.stack([dx, dy, dz], axis=1)
    mean_bias = bias.mean(axis=0)
    residual  = np.linalg.norm(bias - mean_bias, axis=1)

    return {
        "scene":              scene,
        "object":             obj,
        "n_frames":           len(rows),
        # 3-D error
        "err3d_mean_m":       float(err3.mean()),
        "err3d_median_m":     float(np.median(err3)),
        "err3d_std_m":        float(err3.std()),
        "err3d_max_m":        float(err3.max()),
        # 2-D reprojection error
        "err2d_mean_px":      float(err2.mean()),
        "err2d_median_px":    float(np.median(err2)),
        "err2d_std_px":       float(err2.std()),
        "err2d_max_px":       float(err2.max()),
        # Systematic bias
        "bias_x_m":           float(mean_bias[0]),
        "bias_y_m":           float(mean_bias[1]),
        "bias_z_m":           float(mean_bias[2]),
        "bias_norm_m":        float(np.linalg.norm(mean_bias)),
        # Residual after bias removal
        "residual_mean_m":    float(residual.mean()),
        "residual_std_m":     float(residual.std()),
        "residual_max_m":     float(residual.max()),
    }


def _aggregate_summary(summary_rows: list[dict], group_key: str | None) -> list[dict]:
    """
    Average numeric summary fields.
    group_key=None  → single grand-total row
    group_key='object' or 'scene' → one row per unique value of that key
    """
    numeric = [
        "err3d_mean_m", "err3d_median_m", "err3d_std_m", "err3d_max_m",
        "err2d_mean_px", "err2d_median_px", "err2d_std_px", "err2d_max_px",
        "bias_x_m", "bias_y_m", "bias_z_m", "bias_norm_m",
        "residual_mean_m", "residual_std_m", "residual_max_m",
    ]

    if group_key is None:
        row = {"group": "GLOBAL", "n_scene_obj_pairs": len(summary_rows),
               "total_frames": sum(r["n_frames"] for r in summary_rows)}
        for col in numeric:
            vals = [r[col] for r in summary_rows]
            row[col] = float(np.mean(vals))
        return [row]

    groups: dict[str, list[dict]] = {}
    for r in summary_rows:
        groups.setdefault(r[group_key], []).append(r)

    result = []
    for key, grp in sorted(groups.items()):
        row = {group_key: key, "n_scene_obj_pairs": len(grp),
               "total_frames": sum(r["n_frames"] for r in grp)}
        for col in numeric:
            vals = [r[col] for r in grp]
            row[col] = float(np.mean(vals))
        result.append(row)
    return result


def _print_scene_summary(scene: str, rows: list[dict]) -> None:
    err3 = np.array([r["err_3d_m"]  for r in rows])
    err2 = np.array([r["err_2d_px"] for r in rows])
    print(f"\n  ┌─ {scene} ({len(rows)} frames across all objects)")
    print(f"  │  3-D error [mm]: mean={err3.mean()*1000:.1f}  std={err3.std()*1000:.1f}  max={err3.max()*1000:.1f}")
    print(f"  │  2-D error [px]: mean={err2.mean():.1f}   std={err2.std():.1f}   max={err2.max():.1f}")
    print(f"  └{'─'*50}\n")


def _print_global_summary(summary_rows: list[dict], save_dir: Path) -> None:
    err3 = np.array([r["err3d_mean_m"]   for r in summary_rows])
    err2 = np.array([r["err2d_mean_px"]  for r in summary_rows])
    res  = np.array([r["residual_mean_m"] for r in summary_rows])

    print(f"\n{'═'*70}")
    print(f"  GLOBAL SUMMARY  ({len(summary_rows)} scene/object pairs)")
    print(f"{'─'*70}")
    print(f"  3-D error [mm]:          mean={err3.mean()*1000:.2f}  std={err3.std()*1000:.2f}")
    print(f"  2-D error [px]:          mean={err2.mean():.2f}   std={err2.std():.2f}")
    print(f"  Residual (bias-free) [mm]: mean={res.mean()*1000:.2f}  std={res.std()*1000:.2f}")
    print(f"{'─'*70}")
    best_obj = min(summary_rows, key=lambda r: r["err3d_mean_m"])
    worst_obj = max(summary_rows, key=lambda r: r["err3d_mean_m"])
    print(f"  Best  pair: {best_obj['scene']}/{best_obj['object']}  ({best_obj['err3d_mean_m']*1000:.1f} mm)")
    print(f"  Worst pair: {worst_obj['scene']}/{worst_obj['object']}  ({worst_obj['err3d_mean_m']*1000:.1f} mm)")
    print(f"{'═'*70}")
    print(f"\n[INFO] All outputs saved to: {save_dir}")

def rotation_error_deg(T_pred: np.ndarray, T_ref: np.ndarray) -> float:
    R_pred = T_pred[:3, :3]
    R_ref = T_ref[:3, :3]

    R_delta = R_ref.T @ R_pred
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def translation_error_m(T_pred: np.ndarray, T_ref: np.ndarray) -> float:
    return float(np.linalg.norm(T_pred[:3, 3] - T_ref[:3, 3]))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── Paths ────────────────────────────────────────────────────────────────

    # parser.add_argument("--left-calib",        type=Path, required=True,
    #                     help="Path to LEFT camera calibration file (OpenCV YAML/XML). "
    #                          "Must contain K_new (or K) for the left/stereo camera.")
    # parser.add_argument("--rgbd-calib",        type=Path, required=True,
    #                     help="Path to RGBD camera calibration file (OpenCV YAML/XML). "
    #                          "Must contain K_new (or K) for the RGBD camera.")
    # parser.add_argument("--relative-pose",     type=Path, required=True,
    #                     help="OpenCV YAML/XML with T_cam2_cam1 or T_cam1_cam2 "
    #                          "(left = cam1, RGBD = cam2).")
    parser.add_argument("--save-csv",          type=Path, default=None,
                        help="Output CSV path.  Defaults to results-dir/comparison_<scene>_<obj>.csv")

    # ── Scene ────────────────────────────────────────────────────────────────
    parser.add_argument("--scene",             default="scene_001")
    parser.add_argument("--object",            dest="obj", default="rubiks_cube")
    parser.add_argument("--rgbd-camera",       default="realsense",
                        help="Sub-folder name used for the RGBD camera results.")
    parser.add_argument("--point-in-object",   nargs=3, type=float,
                        default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                        help="Point expressed in object coordinates [m] to track. "
                             "Default is the object origin.")

    # ── Camera resolutions ───────────────────────────────────────────────────
    parser.add_argument("--left-native-wh",       nargs=2, type=int, default=[3840, 2160],
                        metavar=("W", "H"),
                        help="Resolution at which the LEFT camera was calibrated.")
    parser.add_argument("--left-inference-wh",    nargs=2, type=int, default=[576, 324],
                        metavar=("W", "H"),
                        help="Resolution at which FoundationPose ran on LEFT images.")
    parser.add_argument("--rgbd-native-wh",       nargs=2, type=int, default=[1280, 720],
                        metavar=("W", "H"),
                        help="Resolution at which the RGBD camera was calibrated.")
    parser.add_argument("--rgbd-inference-wh",    nargs=2, type=int, default=[640, 360],
                        metavar=("W", "H"),
                        help="Resolution at which FoundationPose ran on RGBD images.")

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--translation-mm",    action="store_true", default=True,
                        help="Divide relative-pose translation by 1000 (mm → m).")
    parser.add_argument("--show",              action="store_true", default=False,
                        help="Show interactive per-frame visualisation.")

    args = parser.parse_args()
    date = "24042026"
    parent_dir = Path(__file__).resolve().parents[3]
    (
        gt_data_dir, relative_pose, calib_rgbd_path,
        calib_stereo_path, depth_estimation_dir, depth_comparison_dir,
    ) = prepare_depth_comparison_paths(parent_dir, date, args.rgbd_camera)

    out_dir = parent_dir / "out" / f"out_{date}"
    results_dir = out_dir / "pose_estimation" / "results"

    # ── Intrinsics ────────────────────────────────────────────────────────────
    def load_k(calib_path: Path) -> np.ndarray:
        fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open calibration: {calib_path}")
        try:
            k = fs.getNode("K_new").mat()
            if k is None:
                k = fs.getNode("K").mat()
        finally:
            fs.release()
        if k is None:
            raise ValueError(f"No K_new or K in {calib_path}")
        return np.asarray(k, dtype=np.float64)

    k_left_native, _ = load_camera_calibration(calib_stereo_path, suffix="left")
    k_rgbd_native = load_k(calib_rgbd_path)
    k_left = scale_intrinsics(k_left_native, tuple(args.left_native_wh), tuple(args.left_inference_wh))
    k_rgbd = scale_intrinsics(k_rgbd_native, tuple(args.rgbd_native_wh), tuple(args.rgbd_inference_wh))

    print(f"\n{'─' * 70}")
    print(f"  LEFT  camera: native {args.left_native_wh} → inference {args.left_inference_wh}")
    print(f"  {k_left}")
    print(f"  RGBD  camera: native {args.rgbd_native_wh} → inference {args.rgbd_inference_wh}")
    print(f"  {k_rgbd}")
    print(f"{'─' * 70}\n")

    # ── Relative pose ─────────────────────────────────────────────────────────
    t_rgbd_from_left = read_transform_yaml(relative_pose, direction="cam1_from_cam2")
    t_rgbd_from_left[:3, 3] /= 1000.0
    print(f"T_rgbd_from_left (translation in m):\n{t_rgbd_from_left}\n")

    # ── Output root ───────────────────────────────────────────────────────────
    save_dir = results_dir / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    scenes = ["scene_001", "scene_002", "scene_005", "scene_006", "scene_009"]
    objects = ["apple", "chips_box", "lemon", "orange", "rubiks_cube", "scissors", "wood_block"]

    p_obj_h = np.array([*args.point_in_object, 1.0], dtype=np.float64)

    # Accumulate one summary row per (scene, object) pair
    summary_rows: list[dict] = []
    # Accumulate every raw frame row for the global per-frame CSV
    all_frame_rows: list[dict] = []

    for scene in scenes:
        scene_dir = save_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        scene_rows: list[dict] = []  # all frames across objects within this scene

        for obj in objects:
            try:
                left_files = collect_pose_files(results_dir, "left", scene, obj)
                rgbd_files = collect_pose_files(results_dir, args.rgbd_camera, scene, obj)
            except FileNotFoundError as exc:
                print(f"[SKIP] {exc}")
                continue

            common_ids = sorted(set(left_files) & set(rgbd_files), key=int)
            if not common_ids:
                print(f"[SKIP] No common frames — scene={scene}, object={obj}")
                continue

            print(f"[INFO] {len(common_ids):3d} frames — scene={scene}, object={obj}")

            rows: list[dict] = []
            for frame_id in common_ids:
                t_left = read_pose(left_files[frame_id])
                t_rgbd = read_pose(rgbd_files[frame_id])

                point_left = (t_left @ p_obj_h)[:3]
                point_rgbd_gt = (t_rgbd @ p_obj_h)[:3]
                point_rgbd_xfer = (t_rgbd_from_left @ np.append(point_left, 1.0))[:3]

                diff_3d = point_rgbd_xfer - point_rgbd_gt
                err_3d = float(np.linalg.norm(diff_3d))

                u_gt, v_gt = project_point(point_rgbd_gt, k_rgbd)
                u_xfer, v_xfer = project_point(point_rgbd_xfer, k_rgbd)
                err_2d = float(np.hypot(u_xfer - u_gt, v_xfer - v_gt))

                T_rgbd_xfer = t_rgbd_from_left @ t_left
                T_rgbd_gt = t_rgbd

                err_t_m = translation_error_m(T_rgbd_xfer, T_rgbd_gt)
                err_R_deg = rotation_error_deg(T_rgbd_xfer, T_rgbd_gt)

                row = {
                    "scene": scene,
                    "object": obj,
                    "frame_id": int(frame_id),
                    "err_3d_m": err_3d,
                    "diff_x_m": float(diff_3d[0]),
                    "diff_y_m": float(diff_3d[1]),
                    "diff_z_m": float(diff_3d[2]),
                    "err_2d_px": err_2d,
                    "rgbd_gt_u": u_gt,
                    "rgbd_gt_v": v_gt,
                    "rgbd_xfer_u": u_xfer,
                    "rgbd_xfer_v": v_xfer,
                    "left_x_m": float(point_left[0]),
                    "left_y_m": float(point_left[1]),
                    "left_z_m": float(point_left[2]),
                    "rgbd_gt_x_m": float(point_rgbd_gt[0]),
                    "rgbd_gt_y_m": float(point_rgbd_gt[1]),
                    "rgbd_gt_z_m": float(point_rgbd_gt[2]),
                    "rgbd_xfer_x_m": float(point_rgbd_xfer[0]),
                    "rgbd_xfer_y_m": float(point_rgbd_xfer[1]),
                    "rgbd_xfer_z_m": float(point_rgbd_xfer[2]),
                    "err_translation_m": err_t_m,
                    "err_rotation_deg": err_R_deg,
                }
                rows.append(row)

                if args.show:
                    _show_frame(frame_id, obj, err_3d, err_2d, u_gt, v_gt, u_xfer, v_xfer)
                    key = cv2.waitKey(0)
                    if key in (27, ord("q")):
                        break

            if args.show:
                cv2.destroyAllWindows()

            if not rows:
                continue

            # ── per-object CSV inside scene folder ───────────────────────────
            obj_csv = scene_dir / f"{obj}.csv"
            _write_csv(obj_csv, rows)

            scene_rows.extend(rows)
            all_frame_rows.extend(rows)

            # ── per-(scene, object) summary row ──────────────────────────────
            summary_rows.append(_make_summary_row(scene, obj, rows))

        # ── per-scene CSV (all objects, all frames) ───────────────────────────
        if scene_rows:
            _write_csv(scene_dir / f"{scene}_all_objects.csv", scene_rows)
            # per-scene aggregate (one row per object, written as a mini-summary)
            scene_summary = [r for r in summary_rows if r["scene"] == scene]
            if scene_summary:
                _write_csv(scene_dir / f"{scene}_summary.csv", scene_summary)
            _print_scene_summary(scene, scene_rows)

    if not summary_rows:
        print("[WARN] No data was processed.")
        return

    # ── Global per-frame CSV ──────────────────────────────────────────────────
    _write_csv(save_dir / "all_frames.csv", all_frame_rows)

    # ── Global per-(scene, object) summary CSV ────────────────────────────────
    _write_csv(save_dir / "summary_per_scene_object.csv", summary_rows)

    # ── Global per-object summary (averaged across scenes) ───────────────────
    per_obj = _aggregate_summary(summary_rows, group_key="object")
    _write_csv(save_dir / "summary_per_object.csv", per_obj)

    # ── Global per-scene summary (averaged across objects) ───────────────────
    per_scene = _aggregate_summary(summary_rows, group_key="scene")
    _write_csv(save_dir / "summary_per_scene.csv", per_scene)

    # ── Grand global summary (single row) ────────────────────────────────────
    grand = _aggregate_summary(summary_rows, group_key=None)
    _write_csv(save_dir / "summary_global.csv", grand)

    _print_global_summary(summary_rows, save_dir)

# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _show_frame(
    frame_id: str, obj: str,
    err_3d: float, err_2d: float,
    u_gt: float, v_gt: float,
    u_xfer: float, v_xfer: float,
) -> None:
    canvas = np.zeros((500, 1000, 3), dtype=np.uint8)
    W = canvas.shape[1]

    def txt(msg, y, color=(200, 200, 200), scale=0.85):
        cv2.putText(canvas, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    txt(f"frame={frame_id}   object={obj}", 45)
    txt(f"3-D error : {err_3d*1000:.2f} mm",  100, (0, 220, 255))
    txt(f"2-D error : {err_2d:.2f} px",        155, (120, 255, 120))

    # Dot plot centred on the right side
    cx, cy = W - 200, 300
    dx = int(round(u_xfer - u_gt))
    dy = int(round(v_xfer - v_gt))

    cv2.circle(canvas, (cx, cy),          9, (255, 0, 255), -1)          # magenta = RGBD direct
    cv2.circle(canvas, (cx + dx, cy + dy), 9, (0, 255, 255), -1)        # cyan    = transferred
    cv2.line  (canvas, (cx, cy), (cx + dx, cy + dy), (255, 255, 255), 1)
    txt("● magenta = direct RGBD projection",       230, (255,  80, 255), 0.65)
    txt("● cyan    = transferred from left camera", 265, (80,  255, 255), 0.65)
    txt("(dot plot in RGBD image plane)",            300, (150, 150, 150), 0.60)

    cv2.imshow("FoundationPose pose comparison", canvas)


if __name__ == "__main__":
    main()