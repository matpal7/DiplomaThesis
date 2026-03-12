from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sympy import false

from calibration.ChArUco.charuco_detection import (
    TRESHOLD_CORNERS,
    create_charuco_board,
)
from utils import load_dict, save_dict


@dataclass
class PoseSample:
    pair_name: str
    T_right_left: np.ndarray
    reproj_left: float
    reproj_right: float


def load_camera_calibration(path: Path, stereo_calibration=false) -> tuple[np.ndarray, np.ndarray]:
    calib = load_dict(str(path))
    if "K" not in calib or "dist" not in calib:
        raise KeyError(f"Calibration file {path} must contain keys 'K' and 'dist'.")
    return np.asarray(calib["K"], dtype=np.float64), np.asarray(calib["dist"], dtype=np.float64)


def find_image_pairs(image_dir: Path) -> list[tuple[Path, Path, str]]:
    left_images = sorted(image_dir.glob("*_left.png"))
    right_map = {
        p.name.replace("_right.png", ""): p for p in image_dir.glob("*_right.png")
    }

    pairs: list[tuple[Path, Path, str]] = []
    for left in left_images:
        key = left.name.replace("_left.png", "")
        right = right_map.get(key)
        if right is None:
            continue
        pairs.append((left, right, key))
    return pairs


def pose_from_charuco(
    image_path: Path,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.ArucoDetector,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )

    if retval is None or retval < TRESHOLD_CORNERS or charuco_ids is None:
        return None, None, None

    ids = charuco_ids.flatten().astype(np.int32)
    object_points = board.getChessboardCorners()[ids].reshape(-1, 1, 3).astype(np.float64)
    image_points = charuco_corners.reshape(-1, 1, 2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reproj_error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)))
    return rvec, tvec, reproj_error


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def T_to_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        R_sum += R

    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a @ R_b.T
    trace = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def estimate_relative_pose(
    image_dir: Path,
    left_calib: Path,
    right_calib: Path,
    output_path: Path,
) -> None:
    K_left, dist_left = load_camera_calibration(left_calib)
    K_right, dist_right = load_camera_calibration(right_calib)

    _, board, detector = create_charuco_board()
    pairs = find_image_pairs(image_dir)

    if not pairs:
        raise RuntimeError(f"No *_left.png / *_right.png pairs found in {image_dir}")

    samples: list[PoseSample] = []

    for left_path, right_path, pair_name in pairs:
        rvec_l, tvec_l, err_l = pose_from_charuco(left_path, board, detector, K_left, dist_left)
        rvec_r, tvec_r, err_r = pose_from_charuco(right_path, board, detector, K_right, dist_right)

        if rvec_l is None or rvec_r is None:
            print(f"{pair_name}: skipped (insufficient ChArUco corners in at least one image)")
            continue

        T_left_board = rt_to_T(rvec_l, tvec_l)
        T_right_board = rt_to_T(rvec_r, tvec_r)
        T_right_left = T_right_board @ np.linalg.inv(T_left_board)

        samples.append(PoseSample(pair_name=pair_name, T_right_left=T_right_left, reproj_left=err_l, reproj_right=err_r))
        print(f"{pair_name}: accepted, reproj_left={err_l:.3f}px reproj_right={err_r:.3f}px")

    if not samples:
        raise RuntimeError("Could not estimate pose from any image pair.")

    rotations = [s.T_right_left[:3, :3] for s in samples]
    translations = np.array([s.T_right_left[:3, 3] for s in samples], dtype=np.float64)

    R_avg = average_rotations(rotations)
    t_avg = np.mean(translations, axis=0)

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    rvec_avg, tvec_avg = T_to_rt(T_avg)
    baseline = float(np.linalg.norm(t_avg))

    rot_dev = [rotation_angle_deg(R, R_avg) for R in rotations]
    trans_dev = [float(np.linalg.norm(t - t_avg)) for t in translations]

    output = {
        "num_pairs_total": len(pairs),
        "num_pairs_used": len(samples),
        "T_right_left": T_avg,
        "R_right_left": R_avg,
        "t_right_left": t_avg,
        "rvec_right_left": rvec_avg,
        "tvec_right_left": tvec_avg,
        "baseline_m": baseline,
        "rotation_deviation_deg_mean": float(np.mean(rot_dev)),
        "rotation_deviation_deg_std": float(np.std(rot_dev)),
        "translation_deviation_m_mean": float(np.mean(trans_dev)),
        "translation_deviation_m_std": float(np.std(trans_dev)),
        "samples": [
            {
                "pair": s.pair_name,
                "T_right_left": s.T_right_left,
                "reproj_error_left_px": s.reproj_left,
                "reproj_error_right_px": s.reproj_right,
            }
            for s in samples
        ],
    }

    save_dict(output, str(output_path.parent), output_path.name)

    serializable = {
        "num_pairs_total": output["num_pairs_total"],
        "num_pairs_used": output["num_pairs_used"],
        "T_right_left": np.asarray(output["T_right_left"]).tolist(),
        "rvec_right_left": np.asarray(output["rvec_right_left"]).reshape(3).tolist(),
        "tvec_right_left": np.asarray(output["tvec_right_left"]).reshape(3).tolist(),
        "baseline_m": output["baseline_m"],
        "rotation_deviation_deg_mean": output["rotation_deviation_deg_mean"],
        "rotation_deviation_deg_std": output["rotation_deviation_deg_std"],
        "translation_deviation_m_mean": output["translation_deviation_m_mean"],
        "translation_deviation_m_std": output["translation_deviation_m_std"],
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print("\n=== Relative pose (right <- left) ===")
    print(T_avg)
    print("rvec_right_left:", rvec_avg.reshape(3))
    print("tvec_right_left:", tvec_avg.reshape(3))
    print(f"baseline (pure translation magnitude): {baseline:.6f} m")
    print(f"Saved: {output_path}")
    print(f"Saved: {json_path}")



if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = parent_dir / "dataset_09032026"
    image_dir = dataset_dir / "stereo" / "rgb"
    out_dir = parent_dir / "NICO" / "out_2"

    left_calib = out_dir / "left_calib.npy"
    right_calib = out_dir / "right_calib.npy"

    output = out_dir / "relative_pose_right_from_left.npy"
    estimate_relative_pose(image_dir, left_calib, right_calib, output)
