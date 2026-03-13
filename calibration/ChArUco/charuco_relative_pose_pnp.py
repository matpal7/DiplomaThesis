from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration.ChArUco.charuco_detection import (
    TRESHOLD_CORNERS,
    create_charuco_board,
)
from calibration.image import get_undistort_functions
from utils import load_dict, save_dict

MAX_REPROJ_ERR = 2.0

@dataclass
class PoseSample:
    pair_name: str
    T_cam2_cam1: np.ndarray
    reproj_cam1: float
    reproj_cam2: float

def resize_to_frame(img, frame_size=(680, 480)):
    return cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)

def _extract_first(calib: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in calib:
            return calib[key]
    return None

def load_yaml_calibration(yaml_path: Path) -> dict:
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration yaml file: {yaml_path}")

    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()

    if K is None or D is None:
        raise ValueError(f"Calibration YAML must contain nodes 'K' and 'D': {yaml_path}")

    return {
        "K": K,
        "D": D,
    }


def load_camera_calibration(path: Path, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        calib = load_yaml_calibration(path)
    else:
        calib = load_dict(str(path))

    K = _extract_first(calib, ("K", "K_l"))
    D = _extract_first(calib, ("dist", "D", "D_l"))

    if K is None or D is None:
        raise KeyError(
            f"Calibration file {path} must contain intrinsics. Supported keys: "
            "K/K_l/K_r and dist/D/D_l/D_r."
        )



    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D_arr = np.asarray(D, dtype=np.float64).reshape(-1, 1)

    return K_arr, D_arr


def find_image_pairs(image_dir: Path, cam1_suffix: str, cam2_suffix: str) -> list[tuple[Path, Path, str]]:

    def numeric_key(p: Path):
        name = p.name
        num = name[:-len(cam1_suffix)]
        return int(num)

    cam1_images = sorted(image_dir.glob(f"*{cam1_suffix}"), key=numeric_key)

    cam2_map = {
        p.name[:-len(cam2_suffix)]: p for p in image_dir.glob(f"*{cam2_suffix}")
    }

    pairs: list[tuple[Path, Path, str]] = []

    for cam1 in cam1_images:
        key = cam1.name[:-len(cam1_suffix)]
        cam2 = cam2_map.get(key)

        if cam2 is None:
            continue

        pairs.append((cam1, cam2, key))

    return pairs


def _xi_scalar(xi: np.ndarray) -> float:
    return float(np.asarray(xi, dtype=np.float64).reshape(-1)[0])


def undistort_omni_points_compat(corners_pix: np.ndarray, K: np.ndarray, D: np.ndarray, xi: np.ndarray) -> np.ndarray:
    points = corners_pix.reshape(-1, 1, 2)
    try:
        return cv2.omnidir.undistortPoints(
            distorted=points,
            K=K,
            D=D,
            xi=np.asarray(xi, dtype=np.float64).reshape(1, 1),
            R=np.eye(3, dtype=np.float64),
        )
    except cv2.error:
        return cv2.omnidir.undistortPoints(
            distorted=points,
            K=K,
            D=D,
            xi=_xi_scalar(xi),
            R=np.eye(3, dtype=np.float64),
        )


def project_omni_points_compat(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    obj = object_points.reshape(-1, 1, 3)
    try:
        projected, _ = cv2.omnidir.projectPoints(obj, rvec, tvec, K, _xi_scalar(xi), D)
        return projected
    except cv2.error:
        projected, _ = cv2.omnidir.projectPoints(
            obj,
            rvec,
            tvec,
            K,
            np.asarray(xi, dtype=np.float64).reshape(1, 1),
            D,
        )
        return projected


def pose_from_charuco(
    image_path: Path,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.ArucoDetector,
    K: np.ndarray,
    dist: np.ndarray,
    undistored_function=None,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None

    if undistored_function is not None:
        image = undistored_function(image)

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

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reproj_error = float(
        np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1))
    )

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
    cam1_calib: Path,
    cam2_calib: Path,
    output_path: Path,
    cam1_suffix: str,
    cam2_suffix: str,
    cam1_model: str,
    cam2_model: str,
) -> None:
    K_cam2, dist_cam2 = load_camera_calibration(cam2_calib, cam2_model)

    cam1_calib_dict = load_dict(cam1_calib)
    K_cam1 = cam1_calib_dict["new_K_l"]
    dist_cam1 = np.zeros((4, 1), dtype=np.float64)

    undistored_l, _ = get_undistort_functions(cam1_calib_dict, correct_horizon=False)

    _, board, detector = create_charuco_board()
    pairs = find_image_pairs(image_dir, cam1_suffix, cam2_suffix)

    if not pairs:
        raise RuntimeError(f"No matching pairs found in {image_dir} for suffixes {cam1_suffix} and {cam2_suffix}")

    samples: list[PoseSample] = []

    for cam1_path, cam2_path, pair_name in pairs:
        rvec_1, tvec_1, err_1 = pose_from_charuco(cam1_path, board, detector, K_cam1, dist_cam1, undistored_function=undistored_l)
        rvec_2, tvec_2, err_2 = pose_from_charuco(cam2_path, board, detector, K_cam2, dist_cam2)

        if rvec_1 is None or rvec_2 is None:
            print(f"{pair_name}: skipped (insufficient ChArUco corners in at least one image)")
            continue

        if err_1 > MAX_REPROJ_ERR or err_2 > MAX_REPROJ_ERR:
            print(
                f"{pair_name}: rejected (reproj_cam1={err_1:.3f}px "
                f"reproj_cam2={err_2:.3f}px)"
            )
            continue

        T_cam1_board = rt_to_T(rvec_1, tvec_1)
        T_cam2_board = rt_to_T(rvec_2, tvec_2)
        T_cam2_cam1 = T_cam2_board @ np.linalg.inv(T_cam1_board)

        t_pair = T_cam2_cam1[:3, 3]
        baseline_pair = float(np.linalg.norm(t_pair))
        print(
            f"{pair_name}: BASELINE SIZE: "
            f"baseline={baseline_pair:.4f}"
        )

        samples.append(PoseSample(pair_name=pair_name, T_cam2_cam1=T_cam2_cam1, reproj_cam1=err_1, reproj_cam2=err_2))
        print(f"{pair_name}: accepted, reproj_cam1={err_1:.3f}px reproj_cam2={err_2:.3f}px")

    if not samples:
        raise RuntimeError("Could not estimate pose from any image pair.")

    rotations = [s.T_cam2_cam1[:3, :3] for s in samples]
    translations = np.array([s.T_cam2_cam1[:3, 3] for s in samples], dtype=np.float64)

    R_avg = average_rotations(rotations)
    t_avg = np.mean(translations, axis=0)

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    rvec_avg, tvec_avg = T_to_rt(T_avg)
    baseline = float(np.linalg.norm(t_avg))
    print("BASELINE: ", baseline)

    rot_dev = [rotation_angle_deg(R, R_avg) for R in rotations]
    trans_dev = [float(np.linalg.norm(t - t_avg)) for t in translations]

    output = {
        "num_pairs_total": len(pairs),
        "num_pairs_used": len(samples),
        "T_cam2_cam1": T_avg,
        "R_cam2_cam1": R_avg,
        "t_cam2_cam1": t_avg,
        "rvec_cam2_cam1": rvec_avg,
        "tvec_cam2_cam1": tvec_avg,
        "baseline": baseline,
        "rotation_deviation_deg_mean": float(np.mean(rot_dev)),
        "rotation_deviation_deg_std": float(np.std(rot_dev)),
        "translation_deviation_mean": float(np.mean(trans_dev)),
        "translation_deviation_std": float(np.std(trans_dev)),
        "suffix_cam1": cam1_suffix,
        "suffix_cam2": cam2_suffix,
        "camera_model_cam1": cam1_model,
        "camera_model_cam2": cam2_model,
        "samples": [
            {
                "pair": s.pair_name,
                "T_cam2_cam1": s.T_cam2_cam1,
                "reproj_error_cam1_px": s.reproj_cam1,
                "reproj_error_cam2_px": s.reproj_cam2,
            }
            for s in samples
        ],
    }

    # save_dict(output, str(output_path.parent), output_path.name)

    serializable = {
        "num_pairs_total": output["num_pairs_total"],
        "num_pairs_used": output["num_pairs_used"],
        "T_cam2_cam1": np.asarray(output["T_cam2_cam1"]).tolist(),
        "rvec_cam2_cam1": np.asarray(output["rvec_cam2_cam1"]).reshape(3).tolist(),
        "tvec_cam2_cam1": np.asarray(output["tvec_cam2_cam1"]).reshape(3).tolist(),
        "baseline": output["baseline"],
        "rotation_deviation_deg_mean": output["rotation_deviation_deg_mean"],
        "rotation_deviation_deg_std": output["rotation_deviation_deg_std"],
        "translation_deviation_mean": output["translation_deviation_mean"],
        "translation_deviation_std": output["translation_deviation_std"],
        "suffix_cam1": output["suffix_cam1"],
        "suffix_cam2": output["suffix_cam2"],
        "camera_model_cam1": output["camera_model_cam1"],
        "camera_model_cam2": output["camera_model_cam2"],
    }
    # json_path = output_path.with_suffix(".json")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(serializable, f, indent=2)

    print("\n=== Relative pose (cam2 <- cam1) ===")
    print(T_avg)
    print("rvec_cam2_cam1:", rvec_avg.reshape(3))
    print("tvec_cam2_cam1:", tvec_avg.reshape(3))
    print(f"translation magnitude: {baseline:.6f}")
    print(f"Saved: {output_path}")
    # print(f"Saved: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate relative pose between two cameras from ChArUco image pairs.")
    parser.add_argument("--image-dir", type=Path, required=False, help="Directory with synchronized images from both cameras.")
    parser.add_argument("--cam1-calib", type=Path, required=False, help="Calibration .npy for reference camera (cam1).")
    parser.add_argument("--cam2-calib", type=Path, required=False, help="Calibration .npy for target camera (cam2).")
    parser.add_argument("--cam1-suffix", default="_zed.png", help="Filename suffix for camera 1 images.")
    parser.add_argument("--cam2-suffix", default="_realsense.png", help="Filename suffix for camera 2 images.")
    parser.add_argument(
        "--cam1-model",
        choices=("pinhole", "omni"),
        default="pinhole",
        help="Camera model for camera 1. Use 'omni' for omnidirectional cameras with xi.",
    )
    parser.add_argument(
        "--cam2-model",
        choices=("pinhole", "omni"),
        default="pinhole",
        help="Camera model for camera 2. Use 'omni' for omnidirectional cameras with xi.",
    )
    parser.add_argument("--output", type=Path, required=False, help="Output .npy path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parent_dir = Path(__file__).resolve().parent.parent.parent

    dataset_dir = parent_dir / 'dataset_11032026'
    depth_dir = dataset_dir / 'stereo_4k_relative_pose' / 'rgb'
    out_dir = parent_dir / "NICO" / "out_1103"
    calib_dirc_left_path =  out_dir / "calib_data.npy"
    calib_dirc_realsense = out_dir / "realsense_calibration.yaml"


    args.image_dir = depth_dir
    args.cam1_calib = calib_dirc_left_path
    args.cam2_calib = calib_dirc_realsense
    args.output = out_dir
    estimate_relative_pose(
        image_dir=args.image_dir,
        cam1_calib=args.cam1_calib,
        cam2_calib=args.cam2_calib,
        output_path=args.output,
        cam1_suffix=args.cam1_suffix,
        cam2_suffix=args.cam2_suffix,
        cam1_model=args.cam1_model,
        cam2_model=args.cam2_model,
    )
