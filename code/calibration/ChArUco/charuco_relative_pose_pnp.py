from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from code.calibration.ChArUco.charuco_detection import (
    TRESHOLD_CORNERS,
    create_charuco_board,
)
from code.calibration.image import get_undistort_functions, load_yaml_calibration, load_calib_data, \
    get_undistort_function_mono, numeric_key, extract_key
from code.utils import load_dict, save_dict

MAX_REPROJ_ERR = 2.0

@dataclass
class PoseSample:
    pair_name: str
    T_cam2_cam1: np.ndarray
    reproj_cam1: float
    reproj_cam2: float
    rvec_cam1_board: np.ndarray
    tvec_cam1_board: np.ndarray
    obj_pts_cam1: np.ndarray
    img_pts_cam1: np.ndarray
    obj_pts_cam2: np.ndarray
    img_pts_cam2: np.ndarray



def load_camera_calibration(path: Path, suffix="left") -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)

    if path.suffix.lower() in [".yaml", ".yml"]:
        calib = load_yaml_calibration(path)

        K = calib["K"]
        D = calib["D"]

        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D_arr = np.asarray(D, dtype=np.float64).reshape(-1, 1)

        return K_arr, D_arr


    elif path.suffix.lower() == ".npy":
        calib = load_dict(path)
        if suffix == "left":
            K = calib["new_K_l"]

        else:
            K = calib["new_K_r"]

        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D_arr = np.zeros((5, 1), dtype=np.float64)

        return K_arr, D_arr

    else:
        raise ValueError(f"Unsupported calibration file format: {path}")


def find_image_pairs(image_dir: Path, cam1_suffix: str, cam2_suffix: str) -> list[tuple[Path, Path, str]]:
    cam1_images = sorted(
        image_dir.glob(f"*{cam1_suffix}.png"),
        key=numeric_key
    )

    cam2_map = {
        extract_key(p): p
        for p in image_dir.glob(f"*{cam2_suffix}.png")
    }

    pairs: list[tuple[Path, Path, str]] = []

    for cam1 in cam1_images:
        key = extract_key(cam1)
        cam2 = cam2_map.get(key)

        if cam2 is None:
            continue

        pairs.append((cam1, cam2, key))

    return pairs

def find_images(image_dir: Path, cam1_suffix: str) -> list[Path]:
    cam1_images = sorted(
        image_dir.glob(f"*{cam1_suffix}.png"),
        key=numeric_key
    )

    return cam1_images

def _draw_charuco_points(
    image: np.ndarray,
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
    marker_corners=None,
    marker_ids=None,
    title: str = "",
) -> np.ndarray:
    vis = image.copy()

    if marker_ids is not None and marker_corners is not None and len(marker_ids) > 0:
        vis = cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    if charuco_ids is not None and charuco_corners is not None and len(charuco_ids) > 0:
        for corner, cid in zip(charuco_corners.reshape(-1, 2), charuco_ids.flatten()):
            x, y = int(round(corner[0])), int(round(corner[1]))
            cv2.circle(vis, (x, y), 5, (0, 255, 255), 2)
            cv2.putText(
                vis,
                str(int(cid)),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
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
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def _resize_to_same_height(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 == h2:
        return img1, img2

    target_h = min(h1, h2)

    def resize_keep_aspect(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(round(w * scale)), target_h))

    return resize_keep_aspect(img1, target_h), resize_keep_aspect(img2, target_h)


def draw_charuco_correspondences(
    image1: np.ndarray,
    charuco_corners1: np.ndarray | None,
    charuco_ids1: np.ndarray | None,
    image2: np.ndarray,
    charuco_corners2: np.ndarray | None,
    charuco_ids2: np.ndarray | None,
    title: str = "ChArUco correspondences",
) -> np.ndarray:
    vis1 = image1.copy()
    vis2 = image2.copy()

    vis1, vis2 = _resize_to_same_height(vis1, vis2)
    scale1_x = vis1.shape[1] / image1.shape[1]
    scale1_y = vis1.shape[0] / image1.shape[0]
    scale2_x = vis2.shape[1] / image2.shape[1]
    scale2_y = vis2.shape[0] / image2.shape[0]

    canvas = cv2.hconcat([vis1, vis2])
    offset_x = vis1.shape[1]

    if (
        charuco_ids1 is None or charuco_corners1 is None or
        charuco_ids2 is None or charuco_corners2 is None
    ):
        cv2.putText(
            canvas,
            title,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    pts1 = {
        int(cid): (
            int(round(pt[0] * scale1_x)),
            int(round(pt[1] * scale1_y)),
        )
        for pt, cid in zip(charuco_corners1.reshape(-1, 2), charuco_ids1.flatten())
    }

    pts2 = {
        int(cid): (
            int(round(pt[0] * scale2_x)) + offset_x,
            int(round(pt[1] * scale2_y)),
        )
        for pt, cid in zip(charuco_corners2.reshape(-1, 2), charuco_ids2.flatten())
    }

    common_ids = sorted(set(pts1.keys()) & set(pts2.keys()))

    for cid in common_ids:
        p1 = pts1[cid]
        p2 = pts2[cid]

        cv2.circle(canvas, p1, 5, (0, 255, 255), 2)
        cv2.circle(canvas, p2, 5, (255, 0, 255), 2)
        cv2.line(canvas, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            str(cid),
            (p1[0] + 5, p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(cid),
            (p2[0] + 5, p2[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"{title} | common corners: {len(common_ids)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def show_debug_window(window_name: str, image: np.ndarray, debug: int) -> None:
    if debug <= 0:
        return
    image = cv2.resize(image, (1280, 720))
    cv2.imshow(window_name, image)
    if debug >= 2:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)

def pose_from_charuco(
    image_path: Path,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.ArucoDetector,
    K: np.ndarray,
    dist: np.ndarray,
    undistored_function=None,
    debug: int = 0,
    window_name: str | None = None,
):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None, None, None, None

    if undistored_function is not None:
        image = undistored_function(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) == 0:
        if debug > 0:
            vis = image.copy()
            cv2.putText(
                vis,
                "No markers detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            show_debug_window(window_name or "Charuco debug", vis, debug)
        return None, None, None, image, None, None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )

    vis = _draw_charuco_points(
        image=image,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        marker_corners=marker_corners,
        marker_ids=marker_ids,
        title=window_name or image_path.name,
    )

    if retval is None or retval < TRESHOLD_CORNERS or charuco_ids is None:
        if debug > 0:
            cv2.putText(
                vis,
                f"Not enough ChArUco corners: {0 if retval is None else int(retval)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            show_debug_window(window_name or "Charuco debug", vis, debug)
        return None, None, None, image, charuco_corners, charuco_ids

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
        if debug > 0:
            cv2.putText(
                vis,
                "solvePnP failed",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            show_debug_window(window_name or "Charuco debug", vis, debug)
        return None, None, None, image, charuco_corners, charuco_ids

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reproj_error = float(
        np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1))
    )

    if debug > 0:
        for p in projected.reshape(-1, 2):
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

        cv2.putText(
            vis,
            f"reproj={reproj_error:.3f}px  corners={len(charuco_ids)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        show_debug_window(window_name or "Charuco debug", vis, debug)

    return rvec, tvec, reproj_error, image, charuco_corners, charuco_ids


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

def median_rotation_neighborhood(rotations: list[np.ndarray]) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0]

    best_idx = 0
    best_median = np.inf
    for i, R_i in enumerate(rotations):
        angular_distances = [rotation_angle_deg(R_i, R_j) for R_j in rotations]
        med = float(np.median(angular_distances))
        if med < best_median:
            best_median = med
            best_idx = i
    return rotations[best_idx]


def _median_mad_threshold(values: np.ndarray, sigma_mult: float, min_threshold: float) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    robust_sigma = 1.4826 * mad
    return max(med + sigma_mult * robust_sigma, min_threshold)


def compute_reference_transform(samples: list[PoseSample]) -> np.ndarray:
    transforms = [sample.T_cam2_cam1 for sample in samples]
    rotations = [T[:3, :3] for T in transforms]
    translations = np.array([T[:3, 3] for T in transforms], dtype=np.float64)

    T_ref = np.eye(4, dtype=np.float64)
    T_ref[:3, :3] = median_rotation_neighborhood(rotations)
    T_ref[:3, 3] = np.median(translations, axis=0)
    return T_ref


def reject_outliers(
    samples: list[PoseSample],
    T_ref: np.ndarray,
    trans_sigma_mult: float = 2.0,
    rot_sigma_mult: float = 2.0,
    min_trans_threshold: float = 5.0,
    min_rot_threshold: float = 0.25,
) -> tuple[list[PoseSample], np.ndarray, np.ndarray, float, float]:
    transforms = [sample.T_cam2_cam1 for sample in samples]
    trans_dev = np.array([np.linalg.norm(T[:3, 3] - T_ref[:3, 3]) for T in transforms], dtype=np.float64)
    rot_dev = np.array([rotation_angle_deg(T_ref[:3, :3], T[:3, :3]) for T in transforms], dtype=np.float64)

    trans_thr = _median_mad_threshold(trans_dev, sigma_mult=trans_sigma_mult, min_threshold=min_trans_threshold)
    rot_thr = _median_mad_threshold(rot_dev, sigma_mult=rot_sigma_mult, min_threshold=min_rot_threshold)

    inliers = [
        sample
        for sample, t_d, r_d in zip(samples, trans_dev, rot_dev)
        if t_d <= trans_thr and r_d <= rot_thr
    ]
    return inliers, trans_dev, rot_dev, trans_thr, rot_thr


def _compute_pair_transform(rvec_1, tvec_1, rvec_2, tvec_2):
    T_cam1_board = rt_to_T(rvec_1, tvec_1)
    T_cam2_board = rt_to_T(rvec_2, tvec_2)
    return T_cam2_board @ np.linalg.inv(T_cam1_board)

def _camera_suffix_from_calib_path(calib_path: Path) -> str:
    """
    Example:
        left_NICO.yaml  -> left
        right.yaml      -> right
        realsense.yaml  -> realsense
    """
    return calib_path.stem.split("_")[0].lower()


def _get_undistort_function(calib_dict, camera_suffix):
    use_stereo = False
    if camera_suffix in ("left", "right"):
        use_stereo = True
    if not use_stereo:
        return get_undistort_functions(calib_dict, False)

    undistort_l, undistort_r = get_undistort_functions(calib_dict, stereo=use_stereo)

    if camera_suffix == "left":
        return undistort_l
    if camera_suffix == "right":
        return undistort_r
    # Mono
    return undistort_l

def compute_reproj_rms_joint_basic(
    samples: list[PoseSample],
    K_cam1: np.ndarray,
    dist_cam1: np.ndarray,
    K_cam2: np.ndarray,
    dist_cam2: np.ndarray,
    T_cam2_cam1: np.ndarray,
) -> float:
    residual_blocks = []

    for sample in samples:
        rvec_1 = sample.rvec_cam1_board
        tvec_1 = sample.tvec_cam1_board

        if sample.obj_pts_cam1.shape[0] > 0:
            proj_1, _ = cv2.projectPoints(
                sample.obj_pts_cam1,
                rvec_1,
                tvec_1,
                K_cam1,
                dist_cam1,
            )
            res_1 = proj_1.reshape(-1, 2) - sample.img_pts_cam1.reshape(-1, 2)
            residual_blocks.append(res_1.reshape(-1))

        T_cam1_board = rt_to_T(rvec_1, tvec_1)
        T_cam2_board = T_cam2_cam1 @ T_cam1_board
        rvec_2, tvec_2 = T_to_rt(T_cam2_board)

        if sample.obj_pts_cam2.shape[0] > 0:
            proj_2, _ = cv2.projectPoints(
                sample.obj_pts_cam2,
                rvec_2,
                tvec_2,
                K_cam2,
                dist_cam2,
            )
            res_2 = proj_2.reshape(-1, 2) - sample.img_pts_cam2.reshape(-1, 2)
            residual_blocks.append(res_2.reshape(-1))

    if not residual_blocks:
        return float("nan")

    residuals = np.concatenate(residual_blocks).astype(np.float64)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return rms


def estimate_relative_pose(
    image_dir: Path,
    cam1_calib: Path,
    cam2_calib: Path,
    output_path: Path,
    cam1_suffix: str,
    cam2_suffix: str,
    debug: int = 2,
    squares_horizontally=6,
    squares_vertically=8,
    squares_length=32.0,
    marker_length=22.0
) -> None:
    image_dir = Path(image_dir)
    cam1_calib = Path(cam1_calib)
    cam2_calib = Path(cam2_calib)
    output_path = Path(output_path)

    print(cam2_calib)
    calib_dict_cam1 = load_yaml_calibration(cam1_calib)  # load_calib_data(cam1_calib, type=cam1_suffix)
    calib_dict_cam2 = load_camera_calibration(cam2_calib,
                                              suffix=cam2_suffix)  # load_calib_data(cam2_calib, type=cam2_suffix)
    calib_stereo = load_dict(cam2_calib)

    K_cam1, dist_cam1 = calib_dict_cam1["K"], calib_dict_cam1["D"]
    # K_cam2, dist_cam2 = calib_dict_cam2["K"], calib_dict_cam2["D"]

    K_cam2, dist_cam2 = load_camera_calibration(cam2_calib, suffix=cam2_suffix)

    undistort_1 = _get_undistort_function(calib_dict_cam1, cam1_suffix)
    undistort_2 = _get_undistort_function(calib_stereo, cam2_suffix)

    print(f"Camera 1: {cam1_suffix}")
    print(f"Camera 2: {cam2_suffix}")

    _, board, detector = create_charuco_board(squares_horizontally=squares_horizontally, squares_vertically=squares_vertically, squares_length=squares_length, marker_length=marker_length)
    pairs = find_image_pairs(image_dir, cam1_suffix, cam2_suffix)

    if not pairs:
        raise RuntimeError(
            f"No matching pairs found in {image_dir} for suffixes {cam1_suffix} and {cam2_suffix}"
        )

    samples: list[PoseSample] = []

    for cam1_path, cam2_path, pair_name in pairs:
        (
            rvec_1, tvec_1, err_1,
            img_1, charuco_corners_1, charuco_ids_1
        ) = pose_from_charuco(
            cam1_path,
            board,
            detector,
            K_cam1,
            dist_cam1,
            undistored_function=undistort_1,
            debug=0,
            window_name=f"{pair_name} | {cam1_suffix}",
        )

        (
            rvec_2, tvec_2, err_2,
            img_2, charuco_corners_2, charuco_ids_2
        ) = pose_from_charuco(
            cam2_path,
            board,
            detector,
            K_cam2,
            dist_cam2,
            undistored_function=undistort_2,
            debug=0,
            window_name=f"{pair_name} | {cam2_suffix}",
        )

        if debug > 0 and img_1 is not None and img_2 is not None:
            corr_vis = draw_charuco_correspondences(
                image1=_draw_charuco_points(
                    img_1, charuco_corners_1, charuco_ids_1, title=f"{pair_name} | {cam1_suffix}"
                ),
                charuco_corners1=charuco_corners_1,
                charuco_ids1=charuco_ids_1,
                image2=_draw_charuco_points(
                    img_2, charuco_corners_2, charuco_ids_2, title=f"{pair_name} | {cam2_suffix}"
                ),
                charuco_corners2=charuco_corners_2,
                charuco_ids2=charuco_ids_2,
                title=f"{pair_name}: {cam1_suffix} <-> {cam2_suffix}",
            )

            info_lines = []
            if err_1 is not None:
                info_lines.append(f"{cam1_suffix} err={err_1:.3f}px")
            if err_2 is not None:
                info_lines.append(f"{cam2_suffix} err={err_2:.3f}px")

            if info_lines:
                cv2.putText(
                    corr_vis,
                    " | ".join(info_lines),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            show_debug_window("ChArUco pair debug", corr_vis, debug)

        if rvec_1 is None or rvec_2 is None:
            print(f"{pair_name}: skipped (insufficient ChArUco corners in at least one image)")
            continue

        if err_1 > MAX_REPROJ_ERR or err_2 > MAX_REPROJ_ERR:
            print(
                f"{pair_name}: rejected "
                f"(reproj_cam1={err_1:.3f}px reproj_cam2={err_2:.3f}px)"
            )
            continue

        T_cam2_cam1 = _compute_pair_transform(rvec_1, tvec_1, rvec_2, tvec_2)

        baseline_pair = float(np.linalg.norm(T_cam2_cam1[:3, 3]))
        print(f"{pair_name}: baseline={baseline_pair:.4f}")
        print(f"{pair_name}: accepted, reproj_cam1={err_1:.3f}px reproj_cam2={err_2:.3f}px")

        ids1 = charuco_ids_1.flatten().astype(np.int32)
        ids2 = charuco_ids_2.flatten().astype(np.int32)

        obj_pts_1 = board.getChessboardCorners()[ids1].reshape(-1, 1, 3).astype(np.float64)
        img_pts_1 = charuco_corners_1.reshape(-1, 1, 2).astype(np.float64)

        obj_pts_2 = board.getChessboardCorners()[ids2].reshape(-1, 1, 3).astype(np.float64)
        img_pts_2 = charuco_corners_2.reshape(-1, 1, 2).astype(np.float64)
        samples.append(
            PoseSample(
                pair_name=pair_name,
                T_cam2_cam1=T_cam2_cam1,
                reproj_cam1=err_1,
                reproj_cam2=err_2,
                rvec_cam1_board=rvec_1.copy(),
                tvec_cam1_board=tvec_1.copy(),
                obj_pts_cam1=obj_pts_1,
                img_pts_cam1=img_pts_1,
                obj_pts_cam2=obj_pts_2,
                img_pts_cam2=img_pts_2,
            )
        )

    if debug > 0:
        cv2.destroyAllWindows()

    if not samples:
        raise RuntimeError("Could not estimate pose from any image pair.")

    T_ref = compute_reference_transform(samples)
    inliers, _trans_dev_ref, _rot_dev_ref, trans_thr, rot_thr = reject_outliers(samples, T_ref)

    print(f"Outlier rejection thresholds: translation={trans_thr:.3f}, rotation={rot_thr:.3f} deg")
    print(f"Inliers after robust filtering: {len(inliers)} / {len(samples)}")

    if not inliers:
        raise RuntimeError("All pair transforms were rejected as outliers.")

    rotations = [sample.T_cam2_cam1[:3, :3] for sample in inliers]
    translations = np.array([sample.T_cam2_cam1[:3, 3] for sample in inliers], dtype=np.float64)

    R_avg = average_rotations(rotations)    #použiť knižnicu, nie je trivialne | joint optimization cez všetky pozorovania naraz
    t_avg = np.mean(translations, axis=0)

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    reproj_rms = compute_reproj_rms_joint_basic(
        samples=inliers,
        K_cam1=K_cam1,
        dist_cam1=dist_cam1,
        K_cam2=K_cam2,
        dist_cam2=dist_cam2,
        T_cam2_cam1=T_avg,
    )

    baseline = float(np.linalg.norm(t_avg))
    print("BASELINE:", baseline)
    print(f"Final joint reprojection RMS: {reproj_rms:.4f}px")

    rot_dev = [rotation_angle_deg(R, R_avg) for R in rotations]
    trans_dev = [float(np.linalg.norm(t - t_avg)) for t in translations]

    yaml_path = save_relative_pose_yaml(
        out_dir=output_path,
        cam1_suffix=cam1_suffix,
        cam2_suffix=cam2_suffix,
        K_cam1=K_cam1,
        dist_cam1=dist_cam1,
        K_cam2=K_cam2,
        dist_cam2=dist_cam2,
        T_cam2_cam1=T_avg,
        R_cam2_cam1=R_avg,
        t_cam2_cam1=t_avg,
        baseline=baseline,
        reproj_rms_joint=reproj_rms,
        num_pairs_total=len(pairs),
        num_pairs_used=len(samples),
        rotation_deviation_deg_mean=float(np.mean(rot_dev)),
        rotation_deviation_deg_std=float(np.std(rot_dev)),
        translation_deviation_mean=float(np.mean(trans_dev)),
        translation_deviation_std=float(np.std(trans_dev)),
    )

    print("Relative pose saved to:", yaml_path)

def save_relative_pose_yaml(
    out_dir: Path,
    cam1_suffix: str,
    cam2_suffix: str,
    K_cam1: np.ndarray,
    dist_cam1: np.ndarray,
    K_cam2: np.ndarray,
    dist_cam2: np.ndarray,
    T_cam2_cam1: np.ndarray,
    R_cam2_cam1: np.ndarray,
    t_cam2_cam1: np.ndarray,
    baseline: float,
    reproj_rms_joint: float,
    num_pairs_total: int,
    num_pairs_used: int,
    rotation_deviation_deg_mean: float,
    rotation_deviation_deg_std: float,
    translation_deviation_mean: float,
    translation_deviation_std: float,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_cam1 = cam1_suffix.replace(".", "").replace("*", "").replace("_", "").lower()
    safe_cam2 = cam2_suffix.replace(".", "").replace("*", "").replace("_", "").lower()

    filename = f"relative_pose_{safe_cam1}_to_{safe_cam2}.yaml"
    out_path = out_dir / filename

    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open YAML file for writing: {out_path}")

    try:
        fs.write("suffix_cam1", cam1_suffix)
        fs.write("suffix_cam2", cam2_suffix)

        fs.write("K_cam1", K_cam1)
        fs.write("D_cam1", dist_cam1)

        fs.write("K_cam2", K_cam2)
        fs.write("D_cam2", dist_cam2)

        fs.write("T_cam2_cam1", T_cam2_cam1)
        fs.write("R_cam2_cam1", R_cam2_cam1)
        fs.write("t_cam2_cam1", t_cam2_cam1.reshape(3, 1))

        T_cam1_cam2 = np.linalg.inv(T_cam2_cam1)
        R_cam1_cam2 = T_cam1_cam2[:3, :3]
        t_cam1_cam2 = T_cam1_cam2[:3, 3]

        fs.write("T_cam1_cam2", T_cam1_cam2)
        fs.write("R_cam1_cam2", R_cam1_cam2)
        fs.write("t_cam1_cam2", t_cam1_cam2.reshape(3, 1))

        fs.write("baseline", float(baseline))
        fs.write("reproj_rms_joint", float(reproj_rms_joint))
        fs.write("num_pairs_total", int(num_pairs_total))
        fs.write("num_pairs_used", int(num_pairs_used))

        fs.write("rotation_deviation_deg_mean", float(rotation_deviation_deg_mean))
        fs.write("rotation_deviation_deg_std", float(rotation_deviation_deg_std))
        fs.write("translation_deviation_mean", float(translation_deviation_mean))
        fs.write("translation_deviation_std", float(translation_deviation_std))

    finally:
        fs.release()

    return out_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate relative pose between two cameras from ChArUco image pairs.")
    parser.add_argument("--image-dir", type=Path, required=False, help="Directory with synchronized images from both cameras.")
    parser.add_argument("--cam1-calib", type=Path, required=False, help="Calibration .npy for reference camera (cam1).")
    parser.add_argument("--cam2-calib", type=Path, required=False, help="Calibration .npy for target camera (cam2).")
    parser.add_argument("--output", type=Path, required=False, help="Output .npy path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parent_dir = Path(__file__).resolve().parents[3]
    date = "27032026"
    dataset_dir = parent_dir / "datasets" / f'dataset_{date}'
    depth_dir = dataset_dir / 'stereo_4k_relative_pose' / 'rgb'
    out_dir = parent_dir / "out" / f"out_{date}" / "cameras_parameters"

    calib_dict_stereo = out_dir / "calib_data.npy"

    calib_dict_NICO_left = out_dir / "left_calibration_1280x720.yaml"
    calib_dict_NICO_right = out_dir / "right_NICO.yaml"
    calib_dict_realsense = out_dir / "realsense_calibration_1280x720.yaml"
    calib_dict_zed = out_dir / "zed_calibration_1280x720.yaml"

    cam1_calib = calib_dict_realsense
    cam2_calib = calib_dict_stereo

    out_dir = out_dir / "relative_pose"

    args.image_dir = depth_dir
    args.cam1_calib = cam1_calib
    args.cam2_calib = cam2_calib
    args.output = out_dir
    estimate_relative_pose(
        image_dir=args.image_dir,
        cam1_calib=args.cam1_calib,
        cam2_calib=args.cam2_calib,
        output_path=args.output,
        cam1_suffix="realsense",
        cam2_suffix="left",
        debug=2,
        squares_horizontally=6, squares_vertically=8, squares_length=45.0, marker_length=31.0,
    )
