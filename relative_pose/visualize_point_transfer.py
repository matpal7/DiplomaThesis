from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Pick point in cam1 image"


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


def _validate_transform(T: np.ndarray, path: Path) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Transform matrix in {path} must have shape (4, 4), got {T.shape}.")
    return T


def _pick_point_interactively(img: np.ndarray) -> tuple[int, int]:
    selected = {"pt": None}
    vis = img.copy()

    def _mouse_cb(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["pt"] = (int(x), int(y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_cb)

    print("Left click to choose a point. Press ENTER to confirm.")
    while True:
        frame = vis.copy()
        if selected["pt"] is not None:
            cv2.circle(frame, selected["pt"], 8, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(10) & 0xFF
        if key in (13, 10) and selected["pt"] is not None:  # Enter
            break
        if key == 27:  # ESC
            raise RuntimeError("Point selection canceled.")

    cv2.destroyWindow(WINDOW_NAME)
    return selected["pt"]


def _sample_depth(depth_map_path: Path, u: int, v: int) -> float:
    depth = np.load(depth_map_path)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be a 2D array, got shape={depth.shape}.")
    if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
        raise ValueError(f"Selected pixel {(u, v)} is out of bounds for depth map shape {depth.shape}.")

    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"Invalid depth at pixel {(u, v)}: {z}")

    return z


def _pixel_to_cam_point(u: int, v: int, depth_m: float, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=np.float64)


def _project_point(X_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[int, int]:
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    projected, _ = cv2.projectPoints(
        X_cam.reshape(1, 1, 3),
        rvec,
        tvec,
        K,
        D,
    )
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
    default_depth: float | None = None,
    circle_radius: int = 12,
    save_path: Path | None = None,
) -> None:
    points = []  # list of ((u1, v1), depth, X1, (u2, v2), X2)
    selected = {"pt": None}

    display_scale = 1.0

    def redraw():
        vis1 = img1.copy()
        vis2 = img2.copy()

        for idx, ((u1, v1), depth_m, X1, (u2, v2), X2) in enumerate(points):
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

        combined = np.hstack([vis1, vis2])

        h, w = combined.shape[:2]
        target_w = 1800
        if w > target_w:
            scale = target_w / w
            resized = cv2.resize(combined, (int(w * scale), int(h * scale)))
        else:
            resized = combined

        return resized, combined

    def mouse_cb(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        combined_preview, combined_full = redraw()
        preview_h, preview_w = combined_preview.shape[:2]
        full_h, full_w = combined_full.shape[:2]

        scale_x = full_w / preview_w
        scale_y = full_h / preview_h

        x_full = int(round(x * scale_x))
        y_full = int(round(y * scale_y))

        # click only in left image
        if x_full >= img1.shape[1]:
            return

        u, v = x_full, y_full

        try:
            if depth_map_path is not None:
                depth_m = _sample_depth(depth_map_path, u, v)
            elif default_depth is not None:
                depth_m = float(default_depth)
            else:
                print("No depth source available.")
                return

            X1 = _pixel_to_cam_point(u, v, depth_m, K1)
            X1_h = np.append(X1, 1.0)

            X2_h = T_cam2_cam1 @ X1_h
            X2 = X2_h[:3]

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

    window_name = "Transferred points: click in LEFT image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 900)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Interactive mode:")
    print("  Left mouse click: transfer point from left image to right image")
    print("  c: clear all points")
    print("  s: save current visualization")
    print("  q or ESC: quit")

    while True:
        preview, full = redraw()
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            points.clear()
            print("All points cleared.")
        elif key == ord("s"):
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), full)
                print(f"Saved visualization to: {save_path}")
            else:
                print("No save path set.")

    cv2.destroyWindow(window_name)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer a user-picked point from camera 1 image to camera 2 image "
            "using saved relative pose and camera intrinsics."
        )
    )
    parser.add_argument("--image-cam1", type=Path, required=False, help="Image from camera 1")
    parser.add_argument("--image-cam2", type=Path, required=False, help="Image from camera 2")
    parser.add_argument("--cam1-calib", type=Path, required=False, help="YAML intrinsics for camera 1 (K,D)")
    parser.add_argument("--cam2-calib", type=Path, required=False, help="YAML intrinsics for camera 2 (K,D)")
    parser.add_argument("--relative-pose", type=Path, required=False, help="Relative pose file (.json/.yaml)")

    parser.add_argument("--u", type=int, default=None, help="Pixel u in camera 1")
    parser.add_argument("--v", type=int, default=None, help="Pixel v in camera 1")
    parser.add_argument("--depth", type=float, default=None, help="Depth in meters for the selected point")
    parser.add_argument("--depth-map-cam1", type=Path, default=None, help="Optional .npy depth map for camera 1")

    parser.add_argument("--circle-radius", type=int, default=12, help="Circle radius in pixels")
    parser.add_argument("--save", type=Path, default=None, help="Optional output visualization path")
    parser.add_argument("--show", default=True, action="store_true", help="Show result window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parent_dir = Path(__file__).resolve().parent.parent

    dataset_dir = parent_dir / "dataset_11032026"
    depth_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"

    args.image_cam1 = depth_dir / "0_realsense.png"
    args.image_cam2 = depth_dir / "0_zed.png"

    out_dir = parent_dir / "out" / "cameras_parameters"

    calib_dict_realsense = out_dir / "realsense_calibration.yaml"
    calib_dict_zed = out_dir / "zed_left_calibration.yaml"

    args.relative_pose = out_dir / "relative_pose" / "relative_pose_realsense_to_zed.yaml"
    args.cam1_calib = calib_dict_realsense
    args.cam2_calib = calib_dict_zed
    args.depth_map_cam1 = dataset_dir / "stereo_4k_relative_pose" / "depth" / "0_realsense_depth.npy"

    if args.save is None:
        args.save = out_dir / "relative_pose" / "clicked_points_visualization.png"

    img1 = cv2.imread(str(args.image_cam1))
    img2 = cv2.imread(str(args.image_cam2))

    if img1 is None:
        raise FileNotFoundError(f"Cannot read camera 1 image: {args.image_cam1}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read camera 2 image: {args.image_cam2}")

    K1, D1 = _read_camera_calibration(args.cam1_calib)
    K2, D2 = _read_camera_calibration(args.cam2_calib)
    T_cam2_cam1 = _read_relative_pose(args.relative_pose)

    _interactive_transfer_points(
        img1=img1,
        img2=img2,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        T_cam2_cam1=T_cam2_cam1,
        depth_map_path=args.depth_map_cam1,
        default_depth=args.depth,
        circle_radius=args.circle_radius,
        save_path=args.save,
    )

if __name__ == "__main__":
    main()
