from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.ChArUco.charuco_relative_pose_pnp import _get_undistort_function
from calibration.image import load_calib_data


def _read_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")

    try:
        K = fs.getNode("K").mat()
        D = fs.getNode("D").mat()
    finally:
        fs.release()

    if K is None or D is None:
        raise ValueError(f"Calibration file {path} must contain nodes 'K' and 'D'.")

    return np.asarray(K, dtype=np.float64).reshape(3, 3), np.asarray(D, dtype=np.float64).reshape(-1, 1)


def _validate_transform(T: np.ndarray, path: Path) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Transform matrix in {path} must have shape (4, 4), got {T.shape}.")
    return T


def _read_relative_pose(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        matrix_keys = (
            "T_cam2_cam1",
            "T_right_left",
            "T_right_from_left",
            "T_cam1_cam2",
        )
        key = next((k for k in matrix_keys if k in data), None)

        if key is None:
            raise ValueError(f"Could not find a 4x4 transform matrix in JSON file: {path}")

        T = np.asarray(data[key], dtype=np.float64)
        if key == "T_cam1_cam2":
            T = np.linalg.inv(T)

        return _validate_transform(T, path)

    if suffix in {".yml", ".yaml"}:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open relative pose file: {path}")

        try:
            T = fs.getNode("T_cam2_cam1").mat()
            if T is None:
                T_12 = fs.getNode("T_cam1_cam2").mat()
                if T_12 is None:
                    raise ValueError(f"Could not find 'T_cam2_cam1' or 'T_cam1_cam2' in: {path}")
                T = np.linalg.inv(T_12)
        finally:
            fs.release()

        return _validate_transform(T, path)

    raise ValueError(f"Unsupported relative pose file format: {path}")


def _sample_depth(depth_map: np.ndarray, u: int, v: int, depth_scale: float = 0.001) -> float:
    if depth_map.ndim != 2:
        raise ValueError(f"Depth map must be a 2D array, got shape={depth_map.shape}.")
    if v < 0 or v >= depth_map.shape[0] or u < 0 or u >= depth_map.shape[1]:
        raise ValueError(f"Selected pixel {(u, v)} is out of bounds for depth map shape {depth_map.shape}.")

    z_raw = float(depth_map[v, u])
    if not np.isfinite(z_raw) or z_raw <= 0:
        raise ValueError(f"Invalid depth at pixel {(u, v)}: {z_raw}")

    return z_raw * depth_scale


def _pixel_to_cam_point(u: int, v: int, depth_m: float, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    uv = np.array([[[float(u), float(v)]]], dtype=np.float64)
    undist = cv2.undistortPoints(uv, K, D)
    x_n, y_n = undist.reshape(2)

    return np.array([x_n * depth_m, y_n * depth_m, depth_m], dtype=np.float64)


def _project_point(X_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[int, int]:
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    projected, _ = cv2.projectPoints(X_cam.reshape(1, 1, 3), rvec, tvec, K, D)
    uv = projected.reshape(2)
    return int(round(uv[0])), int(round(uv[1]))


def _interactive_transfer_points(
    img1: np.ndarray,
    img2: np.ndarray,
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    T_cam2_cam1: np.ndarray,
    depth_map_path: Path | None = None,
    depth_scale: float = 0.001,
    default_depth: float | None = None,
    circle_radius: int = 12,
    save_path: Path | None = None,
) -> None:
    points = []  # ((u1, v1), depth, X1, (u2, v2), X2)
    depth_map = None if depth_map_path is None else np.load(depth_map_path)

    def redraw():
        vis1 = img1.copy()
        vis2 = img2.copy()

        for idx, ((u1, v1), depth_m, _X1, (u2, v2), _X2) in enumerate(points):
            color1 = (0, 255, 255)
            color2 = (255, 0, 255)

            cv2.circle(vis1, (u1, v1), circle_radius, color1, 2)
            cv2.putText(
                vis1,
                f"{idx}: ({u1},{v1}) z={depth_m:.3f}m",
                (max(10, u1 + 10), max(25, v1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color1,
                2,
            )

            cv2.circle(vis2, (u2, v2), circle_radius, color2, 2)
            cv2.putText(
                vis2,
                f"{idx}: ({u2},{v2})",
                (max(10, u2 + 10), max(25, v2 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color2,
                2,
            )
        vis1 = cv2.resize(vis1, (1280, 720))
        vis2 = cv2.resize(vis2, (1280, 720))
        return np.hstack([vis1, vis2])

    def mouse_cb(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        u, v = int(x), int(y)
        if u >= img1.shape[1] or v >= img1.shape[0] or u < 0 or v < 0:
            return

        try:
            if depth_map is not None:
                depth_m = _sample_depth(depth_map, u, v, depth_scale=depth_scale)
            elif default_depth is not None:
                depth_m = float(default_depth)
            else:
                print("No depth source available.")
                return

            X1 = _pixel_to_cam_point(u, v, depth_m, K1, D1)
            X2 = (T_cam2_cam1 @ np.append(X1, 1.0))[:3]
            # X2 = (np.linalg.inv(T_cam2_cam1) @ np.append(X1, 1.0))[:3]
            if X2[2] <= 0:
                print(f"Point behind camera 2, skipped. z={X2[2]:.4f}")
                return

            u2, v2 = _project_point(X2, K2, D2)
            points.append(((u, v), depth_m, X1, (u2, v2), X2))

            print("-" * 60)
            print(f"cam1 pixel: ({u}, {v})")
            print(f"depth: {depth_m:.6f} m")
            print(f"cam1 point: {X1}")
            print(f"cam2 point: {X2}")
            print(f"cam2 pixel: ({u2}, {v2})")
        except Exception as e:
            print(f"Click ignored: {e}")

    window_name = "Mode A: click in LEFT panel (cam1) -> projected in RIGHT panel (cam2)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 900)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Interactive mode A")
    print("  Left mouse click in CAM1 image: transfer to CAM2")
    print("  c: clear all points")
    print("  s: save current visualization")
    print("  q or ESC: quit")

    while True:
        full = redraw()
        cv2.imshow(window_name, full)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("c"):
            points.clear()
            print("All points cleared.")
        if key == ord("s"):
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), full)
                print(f"Saved visualization to: {save_path}")

    cv2.destroyWindow(window_name)


def _build_target_to_source_lookup(
    source_depth: np.ndarray,
    K_source: np.ndarray,
    D_source: np.ndarray,
    K_target: np.ndarray,
    D_target: np.ndarray,
    T_target_source: np.ndarray,
    target_shape_hw: tuple[int, int],
    depth_scale: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build lookup from TARGET pixel -> SOURCE pixel using only source depth.

    For each valid source-depth pixel:
      source (u_s, v_s, z) -> 3D source -> 3D target -> target pixel (u_t, v_t)
    and store nearest-depth hit in target z-buffer.
    """
    target_h, target_w = target_shape_hw
    lut_u = np.full((target_h, target_w), -1, dtype=np.int32)
    lut_v = np.full((target_h, target_w), -1, dtype=np.int32)
    zbuf = np.full((target_h, target_w), np.inf, dtype=np.float64)

    ys, xs = np.where(np.isfinite(source_depth) & (source_depth > 0))
    print(f"Valid source depth pixels: {len(xs)}")

    for u_s, v_s in zip(xs.tolist(), ys.tolist()):
        z_m = float(source_depth[v_s, u_s]) * depth_scale

        X_s = _pixel_to_cam_point(int(u_s), int(v_s), z_m, K_source, D_source)
        X_t = (T_target_source @ np.append(X_s, 1.0))[:3]
        if X_t[2] <= 0:
            continue

        u_t, v_t = _project_point(X_t, K_target, D_target)
        if u_t < 0 or u_t >= target_w or v_t < 0 or v_t >= target_h:
            continue

        if X_t[2] < zbuf[v_t, u_t]:
            zbuf[v_t, u_t] = X_t[2]
            lut_u[v_t, u_t] = int(u_s)
            lut_v[v_t, u_t] = int(v_s)

    assigned = int(np.count_nonzero(lut_u >= 0))
    print(f"Assigned target pixels in lookup: {assigned} / {target_h * target_w}")
    return lut_u, lut_v, zbuf


def _interactive_click_target_show_source(
    source_img: np.ndarray,
    target_img: np.ndarray,
    lut_u: np.ndarray,
    lut_v: np.ndarray,
    circle_radius: int = 12,
    save_path: Path | None = None,
) -> None:
    """Click in target image (left stereo) and show corresponding source (RealSense) point."""
    points = []  # ((u_t,v_t),(u_s,v_s))

    def redraw() -> np.ndarray:
        vis_source = source_img.copy()
        vis_target = target_img.copy()

        for idx, ((u_t, v_t), (u_s, v_s)) in enumerate(points):
            color_t = (0, 255, 255)
            color_s = (255, 0, 255)

            cv2.circle(vis_target, (u_t, v_t), circle_radius, color_t, 2)
            cv2.putText(
                vis_target,
                f"{idx}: ({u_t},{v_t})",
                (max(10, u_t + 10), max(25, v_t - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_t,
                2,
            )

            cv2.circle(vis_source, (u_s, v_s), circle_radius, color_s, 2)
            cv2.putText(
                vis_source,
                f"{idx}: ({u_s},{v_s})",
                (max(10, u_s + 10), max(25, v_s - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_s,
                2,
            )

        return np.hstack([vis_target, vis_source])

    def mouse_cb(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        u_t, v_t = int(x), int(y)
        if u_t < 0 or v_t < 0 or u_t >= target_img.shape[1] or v_t >= target_img.shape[0]:
            return

        u_s = int(lut_u[v_t, u_t])
        v_s = int(lut_v[v_t, u_t])
        if u_s < 0 or v_s < 0:
            print(f"No RealSense correspondence for target pixel ({u_t}, {v_t}).")
            return

        points.append(((u_t, v_t), (u_s, v_s)))
        print(f"target pixel ({u_t}, {v_t}) -> source pixel ({u_s}, {v_s})")

    window_name = "Mode B: click LEFT panel (target) -> corresponding RIGHT panel (source)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 900)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Interactive mode B")
    print("  Left mouse click in TARGET image: show corresponding SOURCE pixel")
    print("  c: clear all points")
    print("  s: save current visualization")
    print("  q or ESC: quit")

    while True:
        full = redraw()
        cv2.imshow(window_name, full)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("c"):
            points.clear()
            print("All points cleared.")
        if key == ord("s"):
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), full)
                print(f"Saved visualization to: {save_path}")

    cv2.destroyWindow(window_name)

def reproject_rgbd_to_target_view(
    source_img: np.ndarray,
    source_depth: np.ndarray,
    K_source: np.ndarray,
    D_source: np.ndarray,
    K_target: np.ndarray,
    D_target: np.ndarray,
    T_target_source: np.ndarray,
    target_shape_hw: tuple[int, int],
    depth_scale: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproject RGBD source image into target camera view.

    Returns:
        warped_rgb      : RGB image in target view
        warped_depth    : Z-depth in target camera frame
        valid_mask      : valid projected pixels
    """
    target_h, target_w = target_shape_hw
    src_h, src_w = source_depth.shape[:2]

    warped_rgb = np.zeros((target_h, target_w, 3), dtype=source_img.dtype)
    warped_depth = np.full((target_h, target_w), np.inf, dtype=np.float64)
    valid_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    ys, xs = np.where(np.isfinite(source_depth) & (source_depth > 0))

    for v_s, u_s in zip(ys.tolist(), xs.tolist()):
        z_raw = float(source_depth[v_s, u_s])
        z_m = z_raw * depth_scale

        # 3D point in source camera
        X_s = _pixel_to_cam_point(u_s, v_s, z_m, K_source, D_source)

        # Transform to target camera
        X_t = (T_target_source @ np.append(X_s, 1.0))[:3]

        if X_t[2] <= 0:
            continue

        # Project to target image
        u_t, v_t = _project_point(X_t, K_target, D_target)

        if u_t < 0 or u_t >= target_w or v_t < 0 or v_t >= target_h:
            continue

        # Z-buffer: keep nearest point
        if X_t[2] < warped_depth[v_t, u_t]:
            warped_depth[v_t, u_t] = X_t[2]
            warped_rgb[v_t, u_t] = source_img[v_s, u_s]
            valid_mask[v_t, u_t] = 255

    warped_depth[~np.isfinite(warped_depth)] = 0.0
    return warped_rgb, warped_depth, valid_mask


def visualize_reprojected_result(
    source_img: np.ndarray,
    target_img: np.ndarray,
    warped_rgb: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float = 0.6,
    tile_size: tuple[int, int] = (640, 360),
) -> np.ndarray:
    """
    Create 2x2 visualization:

    top-left     : source image
    top-right    : target image
    bottom-left  : warped RGB
    bottom-right : blended overlay
    """
    mask_bool = valid_mask > 0

    blended = target_img.copy()
    blended[mask_bool] = (
        alpha * warped_rgb[mask_bool].astype(np.float32)
        + (1.0 - alpha) * target_img[mask_bool].astype(np.float32)
    ).astype(np.uint8)

    def make_tile(img: np.ndarray, title: str) -> np.ndarray:
        tile = cv2.resize(img, tile_size)
        cv2.putText(
            tile,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return tile

    top_left = make_tile(source_img, "Source RGBD")
    top_right = make_tile(target_img, "Target Stereo")
    bottom_left = make_tile(warped_rgb, "Warped RGBD -> Stereo")
    bottom_right = make_tile(blended, "Overlay")

    top_row = cv2.hconcat([top_left, top_right])
    bottom_row = cv2.hconcat([bottom_left, bottom_right])

    grid = cv2.vconcat([top_row, bottom_row])
    return grid

def _interactive_transfer_points_multi_targets(
    source_img: np.ndarray,
    source_depth: np.ndarray,
    K_source: np.ndarray,
    D_source: np.ndarray,
    targets: list[dict],
    depth_scale: float = 0.001,
    circle_radius: int = 12,
    save_path: Path | None = None,
) -> None:
    """Click in source image and project to all target images."""
    points = []  # ((u,v), depth_m, [(name,(u_t,v_t))])

    target_views = [t["image"].copy() for t in targets]

    def redraw() -> np.ndarray:
        src_vis = source_img.copy()
        panels = [src_vis]

        for idx, ((u, v), depth_m, target_hits) in enumerate(points):
            cv2.circle(src_vis, (u, v), circle_radius, (0, 255, 255), 2)
            cv2.putText(
                src_vis,
                f"{idx}: ({u},{v}) z={depth_m:.3f}m",
                (max(10, u + 10), max(25, v - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            for target_idx, (_name, (u_t, v_t)) in enumerate(target_hits):
                vis_t = target_views[target_idx]
                cv2.circle(vis_t, (u_t, v_t), circle_radius, (255, 0, 255), 2)
                cv2.putText(
                    vis_t,
                    f"{idx}: ({u_t},{v_t})",
                    (max(10, u_t + 10), max(25, v_t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                )

        cv2.putText(src_vis, 'SOURCE', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        for i, target in enumerate(targets):
            vis = target_views[i]
            cv2.putText(vis, target['name'], (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            panels.append(vis)

        resized = [cv2.resize(panel, (960, 540)) for panel in panels]
        return np.hstack(resized)

    def mouse_cb(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        u, v = int(x), int(y)
        if u < 0 or v < 0 or u >= source_img.shape[1] or v >= source_img.shape[0]:
            return

        try:
            depth_m = _sample_depth(source_depth, u, v, depth_scale=depth_scale)
            X_s = _pixel_to_cam_point(u, v, depth_m, K_source, D_source)

            target_hits = []
            for target in targets:
                X_t = (target['T_target_source'] @ np.append(X_s, 1.0))[:3]
                if X_t[2] <= 0:
                    raise ValueError(f"Point behind camera {target['name']} (z={X_t[2]:.4f}).")

                u_t, v_t = _project_point(X_t, target['K'], target['D'])
                if u_t < 0 or v_t < 0 or u_t >= target['image'].shape[1] or v_t >= target['image'].shape[0]:
                    raise ValueError(f"Projected pixel out of bounds in {target['name']}: ({u_t}, {v_t})")

                target_hits.append((target['name'], (u_t, v_t)))

            points.append(((u, v), depth_m, target_hits))
            print(f"Source pixel ({u},{v}) depth={depth_m:.4f}m")
            for target_name, (u_t, v_t) in target_hits:
                print(f"  -> {target_name}: ({u_t}, {v_t})")
        except Exception as e:
            print(f"Click ignored: {e}")

    window_name = 'Multi-target transfer: click SOURCE -> all targets'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2200, 900)
    cv2.setMouseCallback(window_name, mouse_cb)

    print('Interactive multi-target mode')
    print('  Left mouse click in SOURCE image: transfer to all target cameras')
    print('  c: clear all points')
    print('  s: save current visualization')
    print('  q or ESC: quit')

    while True:
        target_views[:] = [t['image'].copy() for t in targets]
        full = redraw()
        cv2.imshow(window_name, full)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):
            break
        if key == ord('c'):
            points.clear()
            print('All points cleared.')
        if key == ord('s') and save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), full)
            print(f'Saved visualization to: {save_path}')

    cv2.destroyWindow(window_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Interactive point transfer between calibrated cameras.')
    parser.add_argument('--mode', choices=['cam1_to_cam2', 'cam2_to_cam1_via_cam1_depth', 'multi_target_from_cam1'], default='cam1_to_cam2')

    parser.add_argument('--image-cam1', type=Path, required=False, help='Image from source camera (cam1)')
    parser.add_argument('--image-cam2', type=Path, required=False, help='Image from target camera (cam2)')
    parser.add_argument('--cam1-calib', type=Path, required=False, help='YAML intrinsics for source camera (K,D)')
    parser.add_argument('--cam2-calib', type=Path, required=False, help='YAML intrinsics for target camera (K,D)')
    parser.add_argument('--relative-pose', type=Path, required=False, help='Relative pose file (.json/.yaml) with T_cam2_cam1')

    parser.add_argument('--target-images', nargs='*', type=Path, default=None, help='Multiple target images for multi_target_from_cam1 mode')
    parser.add_argument('--target-calibs', nargs='*', type=Path, default=None, help='Multiple target intrinsics files')
    parser.add_argument('--target-relative-poses', nargs='*', type=Path, default=None, help='Relative poses (T_target_cam1) for each target')
    parser.add_argument('--target-names', nargs='*', default=None, help='Display names for targets')

    parser.add_argument('--depth-map-cam1', type=Path, default=None, help='.npy depth map for camera 1')
    parser.add_argument('--depth-scale', type=float, default=1.0)
    parser.add_argument('--depth', type=float, default=None, help='Fallback constant depth (cam1_to_cam2 only)')
    parser.add_argument('--circle-radius', type=int, default=8)
    parser.add_argument('--save', type=Path, default=None)
    parser.add_argument('--invert-relative-pose', action='store_true')
    parser.add_argument('--source-name', default='cam1')
    parser.add_argument('--target-name', default='cam2')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parent_dir = Path(__file__).resolve().parent.parent

    date = "24032026"

    dataset_dir = parent_dir / f"dataset_{date}"
    depth_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"

    args.image_cam1 = depth_dir / "22_realsense.png"
    args.image_cam2 = depth_dir / "22_zed.png"

    out_dir = parent_dir / f"out_{date}" / "cameras_parameters"

    calib_dict_realsense = out_dir / "realsense_calibration_1280x720.yaml"
    calib_dict_zed = out_dir / "zed_calibration_1280x720.yaml"
    calib_dict_NICO_left = out_dir / "left_NICO.yaml"
    calib_dict_stereo = out_dir / "calib_data.npy"



    args.relative_pose = out_dir / "relative_pose" / "relative_pose_realsense_to_zed.yaml"
    args.cam1_calib = calib_dict_realsense
    args.cam2_calib = calib_dict_zed
    args.depth_map_cam1 = dataset_dir / "stereo_4k_relative_pose" / "depth" / "22_realsense_depth.npy"

    args.mode == 'cam1_to_cam2'

    suffix1 = "realsense"
    suffix2 = "zed"

    calib_dict_cam1 = load_calib_data(args.cam1_calib, type=suffix1)
    calib_dict_cam2 = load_calib_data(args.cam2_calib, type=suffix2)

    undistort_1 = _get_undistort_function(calib_dict_cam1, suffix1)
    undistort_2 = _get_undistort_function(calib_dict_cam2, suffix2)

    img1 = cv2.imread(str(args.image_cam1))
    if img1 is None:
        raise FileNotFoundError(f'Cannot read camera 1 image: {args.image_cam1}')

    img1 = undistort_1(img1)
    K1, D1 = _read_camera_calibration(args.cam1_calib)

    if args.image_cam2 is None or args.cam2_calib is None or args.relative_pose is None:
        raise ValueError('Modes cam1_to_cam2/cam2_to_cam1_via_cam1_depth require --image-cam2 --cam2-calib --relative-pose')

    img2 = cv2.imread(str(args.image_cam2))
    if img2 is None:
        raise FileNotFoundError(f'Cannot read camera 2 image: {args.image_cam2}')
    img2 = undistort_2(img2)

    K2, D2 = _read_camera_calibration(args.cam2_calib)
    T_cam2_cam1 = _read_relative_pose(args.relative_pose)
    T_cam2_cam1[:3, 3] /= 1000.0

    print(T_cam2_cam1)
    print(np.linalg.inv(T_cam2_cam1))

    if args.invert_relative_pose:
        T_cam2_cam1 = np.linalg.inv(T_cam2_cam1)

    if args.mode == 'cam1_to_cam2':
        _interactive_transfer_points(
            img1=img1,
            img2=img2,
            K1=K1,
            D1=D1,
            K2=K2,
            D2=D2,
            T_cam2_cam1=T_cam2_cam1,
            depth_map_path=args.depth_map_cam1,
            depth_scale=args.depth_scale,
            default_depth=args.depth,
            circle_radius=args.circle_radius,
            save_path=args.save,
        )
        return

    if args.depth_map_cam1 is None:
        raise ValueError('Mode cam2_to_cam1_via_cam1_depth requires --depth-map-cam1')

    depth_cam1 = np.load(args.depth_map_cam1)
    lut_u, lut_v, _zbuf = _build_target_to_source_lookup(
        source_depth=depth_cam1,
        K_source=K1,
        D_source=D1,
        K_target=K2,
        D_target=D2,
        T_target_source=T_cam2_cam1,
        target_shape_hw=(img2.shape[0], img2.shape[1]),
        depth_scale=args.depth_scale,
    )

    _interactive_click_target_show_source(
        source_img=img1,
        target_img=img2,
        lut_u=lut_u,
        lut_v=lut_v,
        circle_radius=args.circle_radius,
        save_path=args.save,
    )


if __name__ == '__main__':
    main()
