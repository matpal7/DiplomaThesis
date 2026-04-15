from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from code.image import load_l_r_images_undistorted
from code.utils import load_dict
from code.prepare_paths import prepare_relative_pose_paths


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


def detect_chessboard_corners(
    img: np.ndarray,
    pattern_size: tuple[int, int],
) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        raise RuntimeError("Chessboard not found.")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return corners.reshape(-1, 2)


def triangulate_points_unrectified(
    corners_l: np.ndarray,
    corners_r: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
    R_rl: np.ndarray,
    t_rl: np.ndarray,
) -> np.ndarray:
    P1 = K_l @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = K_r @ np.hstack([R_rl.astype(np.float64), t_rl.reshape(3, 1).astype(np.float64)])

    pts_l = corners_l.T
    pts_r = corners_r.T

    pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def collect_row_end_pair_errors(
    pts_3d: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> tuple[list[dict], np.ndarray]:
    cols, rows = pattern_size

    def idx(x: int, y: int) -> int:
        return y * cols + x

    pair_results: list[dict] = []
    errors: list[float] = []

    expected = float((cols - 1) * square_size_m)

    for y in range(rows):
        i_left = idx(0, y)
        i_right = idx(cols - 1, y)
        print(i_left, i_right)

        measured = float(np.linalg.norm(pts_3d[i_left] - pts_3d[i_right]))
        error = measured - expected
        print("Measured" , measured)
        print("Expected" , expected)
        print("Error" , error)
        error_per_square = error / float(cols - 1)
        print("Error per square" , error_per_square)

        scale = measured / expected if expected > 0 else None
        scale_error = scale - 1.0 if scale is not None else None
        percent_error = (error / expected) * 100.0 if expected > 0 else None

        pair_results.append(
            {
                "pair_type": "row_endpoints",
                "row": y,
                "point_1": i_left,
                "point_2": i_right,
                "grid_1": [0, y],
                "grid_2": [cols - 1, y],
                "expected_distance_m": expected,
                "measured_distance_m": measured,
                "error_m_total": error,
                "error_m_per_square": error_per_square,
                "scale": scale,
                "scale_error": scale_error,
                "percent_error": percent_error,
            }
        )
        errors.append(error)

    return pair_results, np.asarray(errors, dtype=np.float64)


def summarize_errors(errors: np.ndarray, expected_distance_m: float | None = None) -> dict:
    if errors.size == 0:
        return {
            "num_pairs": 0,
            "mean_abs_error_m": None,
            "rmse_m": None,
            "mean_signed_error_m": None,
            "median_abs_error_m": None,
            "mean_scale": None,
            "mean_scale_error": None,
            "mean_percent_error": None,
        }

    mean_signed_error = float(np.mean(errors))

    if expected_distance_m is not None and expected_distance_m > 0:
        mean_scale = float((expected_distance_m + mean_signed_error) / expected_distance_m)
        mean_scale_error = float(mean_scale - 1.0)
        mean_percent_error = float((mean_signed_error / expected_distance_m) * 100.0)
    else:
        mean_scale = None
        mean_scale_error = None
        mean_percent_error = None

    return {
        "num_pairs": int(errors.size),
        "mean_abs_error_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "mean_signed_error_m": mean_signed_error,
        "median_abs_error_m": float(np.median(np.abs(errors))),
        "mean_scale": mean_scale,
        "mean_scale_error": mean_scale_error,
        "mean_percent_error": mean_percent_error,
    }


def find_best_corner_order(
    corners_l: np.ndarray,
    corners_r: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
    R_rl: np.ndarray,
    t_rl: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    best_mode = None
    best_score = np.inf
    best_pts_3d = None
    best_corners_r = None

    for mode in ("none", "flip_h", "flip_v", "flip_hv"):
        candidate_r = reorder_chessboard_corners(corners_r, pattern_size, mode)
        pts_3d = triangulate_points_unrectified(
            corners_l=corners_l,
            corners_r=candidate_r,
            K_l=K_l,
            K_r=K_r,
            R_rl=R_rl,
            t_rl=t_rl,
        )

        if not np.all(np.isfinite(pts_3d)):
            continue

        front_ratio = float(np.mean(pts_3d[:, 2] > 0))
        if front_ratio < 0.8:
            continue

        _, errors = collect_row_end_pair_errors(
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
            best_pts_3d = pts_3d.copy()
            best_corners_r = candidate_r.copy()

    if best_pts_3d is None or best_corners_r is None or best_mode is None:
        raise RuntimeError("Could not find a valid chessboard corner ordering.")

    return best_pts_3d, best_corners_r, best_mode, best_score


def draw_chessboard_corners_labeled(
    image: np.ndarray,
    corners: np.ndarray,
    pattern_size: tuple[int, int],
    title: str = "",
) -> np.ndarray:
    vis = image.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners.reshape(-1, 1, 2), True)

    for i, pt in enumerate(corners):
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


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent.parent.parent
    date = "11042026"
    rgbd_suffix = "realsense"

    pattern_size = (7, 5)
    square_size_m = 0.045
    max_imgs = 20
    show_debug = False

    (
        dataset_dir,
        relative_pose_dir,
        out_dir,
        out_dir_save,
        calib_dict_stereo,
        calib_dict_RGBD_cam,
    ) = prepare_relative_pose_paths(parent_dir, date, rgbd_suffix)

    calibration_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    calib_dict = load_dict(calib_dict_stereo)

    K_l = np.asarray(calib_dict["new_K_l"], dtype=np.float64)
    K_r = np.asarray(calib_dict["new_K_r"], dtype=np.float64)

    rvec = np.asarray(calib_dict["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(calib_dict["tvec"], dtype=np.float64).reshape(3, 1) / 1000.0
    R_rl, _ = cv2.Rodrigues(rvec)

    out_root = out_dir / "stereo_point_distance_validation"
    vis_dir = out_root / "detections"
    out_root.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    imgs_l, imgs_r = load_l_r_images_undistorted(
        calib_dict,
        calibration_dir,
        max_imgs=max_imgs,
        get_rectified=False,
    )

    per_image_summary: list[dict] = []
    all_errors: list[np.ndarray] = []

    expected_distance_m = (pattern_size[0] - 1) * square_size_m

    for img_l, img_r in tqdm(zip(imgs_l, imgs_r), total=len(imgs_l), desc="validating stereo distances"):
        image_id = str(img_l.get_image_number())

        try:
            corners_l = detect_chessboard_corners(img_l.img, pattern_size)
            corners_r = detect_chessboard_corners(img_r.img, pattern_size)

            pts_3d, corners_r_best, reorder_mode, reorder_score = find_best_corner_order(
                corners_l=corners_l,
                corners_r=corners_r,
                K_l=K_l,
                K_r=K_r,
                R_rl=R_rl,
                t_rl=tvec,
                pattern_size=pattern_size,
                square_size_m=square_size_m,
            )

            pair_results, errors = collect_row_end_pair_errors(
                pts_3d=pts_3d,
                pattern_size=pattern_size,
                square_size_m=square_size_m,
            )
            stats = summarize_errors(errors, expected_distance_m=expected_distance_m)

            per_image_data = {
                "image_id": image_id,
                "status": "ok",
                "reorder_mode": reorder_mode,
                "reorder_score": reorder_score,
                "summary": stats,
                "pairs": pair_results,
            }

            save_json(out_root / "per_image" / f"{image_id}.json", per_image_data)

            vis_l = draw_chessboard_corners_labeled(img_l.img, corners_l, pattern_size, f"Left | {image_id}")
            vis_r = draw_chessboard_corners_labeled(img_r.img, corners_r_best, pattern_size, f"Right | {image_id}")
            vis_l, vis_r = resize_to_same_height(vis_l, vis_r)
            detection_vis = cv2.hconcat([vis_l, vis_r])
            cv2.imwrite(str(vis_dir / f"{image_id}.png"), detection_vis)

            if show_debug:
                preview = cv2.resize(detection_vis, (1600, 800))
                cv2.imshow("Stereo point distance validation", preview)
                cv2.waitKey(0)

            per_image_summary.append(
                {
                    "image_id": image_id,
                    "status": "ok",
                    "reorder_mode": reorder_mode,
                    **stats,
                }
            )
            all_errors.append(errors)

        except RuntimeError as exc:
            per_image_summary.append(
                {
                    "image_id": image_id,
                    "status": f"skipped: {exc}",
                    "reorder_mode": None,
                    "num_pairs": 0,
                    "mean_abs_error_m": None,
                    "rmse_m": None,
                    "mean_signed_error_m": None,
                    "median_abs_error_m": None,
                }
            )

    if show_debug:
        cv2.destroyAllWindows()

    valid_error_arrays = [e for e in all_errors if e.size > 0]
    global_errors = np.concatenate(valid_error_arrays) if valid_error_arrays else np.asarray([], dtype=np.float64)
    global_stats = summarize_errors(global_errors, expected_distance_m=expected_distance_m)

    summary = {
        "date": date,
        "pattern_size": list(pattern_size),
        "square_size_m": square_size_m,
        "baseline_m": float(np.linalg.norm(tvec.reshape(-1))),
        "num_images_total": len(per_image_summary),
        "num_images_valid": sum(1 for r in per_image_summary if r["status"] == "ok"),
        "num_images_skipped": sum(1 for r in per_image_summary if r["status"] != "ok"),
        **global_stats,
    }

    save_json(out_root / "per_image_summary.json", {"images": per_image_summary})
    save_json(out_root / "global_summary.json", summary)

    print("\nGlobal summary:")
    print(summary)
    print(f"Saved per-image detailed results to: {out_root / 'per_image'}")
    print(f"Saved per-image summary to: {out_root / 'per_image_summary.json'}")
    print(f"Saved global summary to: {out_root / 'global_summary.json'}")
    print(f"Saved detection visualizations to: {vis_dir}")