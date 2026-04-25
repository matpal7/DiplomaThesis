from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from code.image import load_l_r_images_undistorted
from code.prepare_paths import prepare_relative_pose_paths
from code.utils import load_dict

def draw_chessboard_corners_labeled(
    image: np.ndarray,
    corners: np.ndarray,
    pattern_size: tuple[int, int],
    title: str = "",
) -> np.ndarray:
    vis = image.copy()

    cv2.drawChessboardCorners(vis, pattern_size, corners, True)

    for i, pt in enumerate(corners.reshape(-1, 2)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(
            vis,
            str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if title:
        cv2.putText(
            vis,
            title,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def resize_to_same_height(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 == h2:
        return img1, img2

    target_h = min(h1, h2)

    def resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(round(w * scale)), target_h))

    return resize_keep_aspect(img1, target_h), resize_keep_aspect(img2, target_h)


def create_detection_verification_vis(
    img_l: np.ndarray,
    img_r: np.ndarray,
    corners_l: np.ndarray,
    corners_r: np.ndarray,
    pattern_size: tuple[int, int],
    image_id: str,
) -> np.ndarray:
    vis_l = draw_chessboard_corners_labeled(
        img_l,
        corners_l,
        pattern_size=pattern_size,
        title=f"Left | image {image_id}",
    )
    vis_l = cv2.cvtColor(vis_l, cv2.COLOR_BGR2RGB)
    vis_r = draw_chessboard_corners_labeled(
        img_r,
        corners_r,
        pattern_size=pattern_size,
        title=f"Right | image {image_id}",
    )
    vis_r = cv2.cvtColor(vis_r, cv2.COLOR_BGR2RGB)

    vis_l, vis_r = resize_to_same_height(vis_l, vis_r)
    canvas = cv2.hconcat([vis_l, vis_r])

    divider_x = vis_l.shape[1]
    cv2.line(canvas, (divider_x, 0), (divider_x, canvas.shape[0] - 1), (255, 255, 255), 2)

    return canvas

def reorder_chessboard_corners(
    corners: np.ndarray,
    pattern_size: tuple[int, int],
    mode: str,
) -> np.ndarray:
    cols, rows = pattern_size
    pts = corners.reshape(rows, cols, 2).copy()

    if mode == "none":
        pass
    elif mode == "flip_h":
        pts = pts[:, ::-1, :]
    elif mode == "flip_v":
        pts = pts[::-1, :, :]
    elif mode == "flip_hv":
        pts = pts[::-1, ::-1, :]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return pts.reshape(-1, 2)


def collect_chessboard_distance_errors(
    pts_3d: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> np.ndarray:
    cols, rows = pattern_size

    def idx(x: int, y: int) -> int:
        return y * cols + x

    errors: list[float] = []

    for y in range(rows):
        for x in range(cols):
            i = idx(x, y)

            if x + 1 < cols:
                d = np.linalg.norm(pts_3d[i] - pts_3d[idx(x + 1, y)])
                errors.append(float(d - square_size_m))

            if y + 1 < rows:
                d = np.linalg.norm(pts_3d[i] - pts_3d[idx(x, y + 1)])
                errors.append(float(d - square_size_m))

    return np.asarray(errors, dtype=np.float64)

def triangulate_chessboard_corners_unrectified(
    img_l: np.ndarray,
    img_r: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
    R_rl: np.ndarray,
    t_rl: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

    if not ret_l or not ret_r:
        raise RuntimeError("Chessboard not found in one or both images.")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), crit)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), crit)

    corners_l_2d = corners_l.reshape(-1, 2)
    corners_r_2d = corners_r.reshape(-1, 2)

    P1 = K_l @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = K_r @ np.hstack([R_rl.astype(np.float64), t_rl.reshape(3, 1).astype(np.float64)])

    best_mode = None
    best_score = np.inf
    best_corners_r = None
    best_pts_3d = None

    for mode in ("none", "flip_h", "flip_v", "flip_hv"):
        candidate_r = reorder_chessboard_corners(
            corners=corners_r_2d,
            pattern_size=pattern_size,
            mode=mode,
        )

        pts_l = corners_l_2d.T
        pts_r = candidate_r.T

        pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
        pts_3d = (pts_4d[:3] / pts_4d[3]).T

        # reject obviously invalid solutions
        if not np.all(np.isfinite(pts_3d)):
            continue

        # optional: prefer points mostly in front of the left camera
        front_ratio = float(np.mean(pts_3d[:, 2] > 0))
        if front_ratio < 0.8:
            continue

        errors = collect_chessboard_distance_errors(
            pts_3d=pts_3d,
            pattern_size=pattern_size,
            square_size_m=square_size_m,
        )

        if errors.size == 0:
            continue

        score = float(np.mean(np.abs(errors)))

        if score < best_score:
            best_score = score
            best_mode = mode
            best_corners_r = candidate_r.copy()
            best_pts_3d = pts_3d.copy()

    if best_pts_3d is None or best_corners_r is None or best_mode is None:
        raise RuntimeError("Could not find a valid chessboard corner ordering.")

    return corners_l_2d, best_corners_r, best_pts_3d, best_mode, best_score


def collect_chessboard_distance_errors(
    pts_3d: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> np.ndarray:
    cols, rows = pattern_size

    def idx(x: int, y: int) -> int:
        return y * cols + x

    errors: list[float] = []

    for y in range(rows):
        for x in range(cols):
            i = idx(x, y)

            if x + 1 < cols:
                d = np.linalg.norm(pts_3d[i] - pts_3d[idx(x + 1, y)])
                errors.append(float(d - square_size_m))

            if y + 1 < rows:
                d = np.linalg.norm(pts_3d[i] - pts_3d[idx(x, y + 1)])
                errors.append(float(d - square_size_m))

    return np.asarray(errors, dtype=np.float64)


def summarize_errors(errors: np.ndarray) -> dict:
    if errors.size == 0:
        return {
            "num_pairs": 0,
            "mean_abs_error_m": None,
            "rmse_m": None,
            "mean_signed_error_m": None,
        }

    return {
        "num_pairs": int(errors.size),
        "mean_abs_error_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "mean_signed_error_m": float(np.mean(errors)),
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]

    for row in rows:
        values = []
        for h in headers:
            v = row[h]
            values.append("" if v is None else str(v))
        lines.append(",".join(values))

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent.parent.parent
    date = "2404026"
    rgbd_suffix = "zed"

    pattern_size = (7,5)
    square_size_m = 0.030
    max_imgs = 20

    (
        dataset_dir,
        relative_pose_dir,
        out_dir,
        out_dir_save,
        calib_dict_stereo,
        calib_dict_RGBD_cam,
    ) = prepare_relative_pose_paths(parent_dir, date, rgbd_suffix)

    calibration_dir = dataset_dir / "stereo_4k_calibration" / "rgb"
    calib_dict = load_dict(calib_dict_stereo)

    K_l = np.asarray(calib_dict["new_K_l"], dtype=np.float64)
    K_r = np.asarray(calib_dict["new_K_r"], dtype=np.float64)

    rvec = np.asarray(calib_dict["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(calib_dict["tvec"], dtype=np.float64).reshape(3, 1) / 1000.0
    R_rl, _ = cv2.Rodrigues(rvec)

    baseline_m = float(np.linalg.norm(tvec.reshape(-1)))
    print("BASELINE in meters:", baseline_m)

    imgs_l, imgs_r = load_l_r_images_undistorted(
        calib_dict,
        calibration_dir,
        max_imgs=max_imgs,
        get_rectified=False,
    )

    out_root = out_dir / f"out_{date}" / "stereo_chessboard_validation"
    out_root.mkdir(parents=True, exist_ok=True)

    per_image_results: list[dict] = []
    all_errors: list[np.ndarray] = []

    show_debug = False

    for img_l, img_r in zip(imgs_l, imgs_r):
        image_id = str(img_l.get_image_number())

        result = {
            "image_id": image_id,
            "status": "ok",
            "num_pairs": 0,
            "mean_abs_error_m": None,
            "rmse_m": None,
            "mean_signed_error_m": None,
        }

        try:
            corners_l, corners_r, pts_3d, reorder_mode, reorder_score = triangulate_chessboard_corners_unrectified(
                img_l.img,
                img_r.img,
                K_l=K_l,
                K_r=K_r,
                R_rl=R_rl,
                t_rl=tvec,
                pattern_size=pattern_size,
                square_size_m=square_size_m,
            )

            print(f"best reorder mode: {reorder_mode}, score: {reorder_score:.6f}")

            errors = collect_chessboard_distance_errors(
                pts_3d=pts_3d,
                pattern_size=pattern_size,
                square_size_m=square_size_m,
            )
            stats = summarize_errors(errors)

            result.update(stats)
            all_errors.append(errors)

            detection_vis = create_detection_verification_vis(
                img_l=img_l.img,
                img_r=img_r.img,
                corners_l=corners_l,
                corners_r=corners_r,
                pattern_size=pattern_size,
                image_id=image_id,
            )

            if show_debug:
                preview = cv2.resize(detection_vis, (1600, 800))
                cv2.imshow("Chessboard detection verification", preview)
                cv2.waitKey(0)

            errors = collect_chessboard_distance_errors(
                pts_3d=pts_3d,
                pattern_size=pattern_size,
                square_size_m=square_size_m,
            )
            stats = summarize_errors(errors)

            result.update(stats)
            all_errors.append(errors)

        except RuntimeError as exc:
            result["status"] = f"skipped: {exc}"

        per_image_results.append(result)
        print(result)

    valid_error_arrays = [e for e in all_errors if e.size > 0]
    if valid_error_arrays:
        global_errors = np.concatenate(valid_error_arrays)
        global_stats = summarize_errors(global_errors)
    else:
        global_stats = summarize_errors(np.asarray([], dtype=np.float64))

    summary = {
        "date": date,
        "pattern_size": list(pattern_size),
        "square_size_m": square_size_m,
        "baseline_m": baseline_m,
        "num_images_total": len(per_image_results),
        "num_images_valid": sum(1 for r in per_image_results if r["status"] == "ok"),
        "num_images_skipped": sum(1 for r in per_image_results if r["status"] != "ok"),
        **global_stats,
    }

    save_json(out_root / "per_image_stats.json", {"images": per_image_results})
    save_json(out_root / "global_stats.json", summary)
    save_csv(out_root / "per_image_stats.csv", per_image_results)

    print("\nGlobal summary:")
    print(summary)
    print(f"Saved per-image stats to: {out_root / 'per_image_stats.json'}")
    print(f"Saved global stats to: {out_root / 'global_stats.json'}")
    print(f"Saved CSV to: {out_root / 'per_image_stats.csv'}")