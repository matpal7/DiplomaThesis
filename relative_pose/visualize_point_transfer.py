from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Point transfer between two calibrated cameras. "
            "Use mode A for cam1->cam2 transfer (click cam1). "
            "Use mode B for inverse query via cam1 depth only (click cam2)."
        )
    )
    parser.add_argument("--image-cam1", type=Path, required=True, help="Image from source camera (cam1)")
    parser.add_argument("--image-cam2", type=Path, required=True, help="Image from target camera (cam2)")
    parser.add_argument("--cam1-calib", type=Path, required=True, help="YAML intrinsics for source camera (K,D)")
    parser.add_argument("--cam2-calib", type=Path, required=True, help="YAML intrinsics for target camera (K,D)")
    parser.add_argument("--relative-pose", type=Path, required=True, help="Relative pose file (.json/.yaml)")

    parser.add_argument(
        "--mode",
        choices=["cam1_to_cam2", "cam2_to_cam1_via_cam1_depth"],
        default="cam1_to_cam2",
        help=(
            "cam1_to_cam2: click in cam1 (needs cam1 depth). "
            "cam2_to_cam1_via_cam1_depth: click in cam2 and recover cam1 pixel using lookup built from cam1 depth."
        ),
    )

    parser.add_argument("--depth-map-cam1", type=Path, default=None, help=".npy depth map for camera 1")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth scale to convert raw depth to meters")
    parser.add_argument("--depth", type=float, default=None, help="Fallback constant depth in meters")

    parser.add_argument("--circle-radius", type=int, default=12, help="Circle radius in pixels")
    parser.add_argument("--save", type=Path, default=None, help="Optional output visualization path")
    parser.add_argument(
        "--invert-relative-pose",
        action="store_true",
        help=(
            "Invert loaded transform before use. Useful when your file contains "
            "T_cam1_cam2 but you need T_cam2_cam1."
        ),
    )
    parser.add_argument("--source-name", default="cam1", help="Display name for source camera")
    parser.add_argument("--target-name", default="cam2", help="Display name for target camera")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    img1 = cv2.imread(str(args.image_cam1))
    img2 = cv2.imread(str(args.image_cam2))
    if img1 is None:
        raise FileNotFoundError(f"Cannot read camera 1 image: {args.image_cam1}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read camera 2 image: {args.image_cam2}")

    K1, D1 = _read_camera_calibration(args.cam1_calib)
    K2, D2 = _read_camera_calibration(args.cam2_calib)
    T_cam2_cam1 = _read_relative_pose(args.relative_pose)
    if args.invert_relative_pose:
        T_cam2_cam1 = np.linalg.inv(T_cam2_cam1)

    print("=" * 70)
    print("RGBD -> 3D -> transform -> projection pipeline")
    print(f"Source camera(cam1): {args.source_name}")
    print(f"Target camera(cam2): {args.target_name}")
    print(f"Mode: {args.mode}")
    print(f"Image cam1: {args.image_cam1}")
    print(f"Image cam2: {args.image_cam2}")
    print(f"Depth map cam1: {args.depth_map_cam1}")
    print("Using transform T_cam2_cam1:")
    print(T_cam2_cam1)
    print("=" * 70)

    if args.save is None:
        args.save = Path("out") / f"transfer_{args.source_name}_to_{args.target_name}_{args.mode}.png"

    if args.mode == "cam1_to_cam2":
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
        raise ValueError("Mode cam2_to_cam1_via_cam1_depth requires --depth-map-cam1")

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


if __name__ == "__main__":
    main()
