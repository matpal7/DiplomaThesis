from pathlib import Path

import cv2
import numpy as np

from calibration.image import load_l_r_images_rectified, get_rectify_functions
from utils import load_dict


def median_depth_at(depth, u, v, win=2, depth_scale=0.001):
    h, w = depth.shape
    u0 = max(0, u - win)
    u1 = min(w, u + win + 1)
    v0 = max(0, v - win)
    v1 = min(h, v + win + 1)

    patch = depth[v0:v1, u0:u1].astype(np.float64)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None
    return float(np.median(patch)) * depth_scale


def pixel_to_3d(u, v, z, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def chessboard_corner_distances(rgb, depth, K, pattern_size, square_size_m, depth_scale=0.001):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        raise RuntimeError("Chessboard not found")

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    pts3d = []
    valid_ids = []

    for i, c in enumerate(corners.reshape(-1, 2)):
        u, v = int(round(c[0])), int(round(c[1]))
        z = median_depth_at(depth, u, v, win=2, depth_scale=depth_scale)
        if z is None:
            pts3d.append(None)
        else:
            pts3d.append(pixel_to_3d(c[0], c[1], z, K))
            valid_ids.append(i)

    cols, rows = pattern_size
    errors = []

    def idx(x, y):
        return y * cols + x

    for y in range(rows):
        for x in range(cols):
            i = idx(x, y)

            if pts3d[i] is None:
                continue

            # horizontal neighbor
            if x + 1 < cols and pts3d[idx(x + 1, y)] is not None:
                d = np.linalg.norm(pts3d[i] - pts3d[idx(x + 1, y)])
                errors.append(d - square_size_m)

            # vertical neighbor
            if y + 1 < rows and pts3d[idx(x, y + 1)] is not None:
                d = np.linalg.norm(pts3d[i] - pts3d[idx(x, y + 1)])
                errors.append(d - square_size_m)

    errors = np.array(errors)
    return {
        "num_pairs": len(errors),
        "mean_abs_error_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "mean_signed_error_m": float(np.mean(errors)),
    }

def triangulate_chessboard_corners(img_l, img_r, K_rect_l, K_rect_r, baseline_m, pattern_size):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

    if not ret_l or not ret_r:
        raise RuntimeError("Chessboard not found in one or both images.")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), crit)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), crit)

    # rectified stereo projection matrices
    P1 = np.hstack([K_rect_l, np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([K_rect_r, np.array([[-K_rect_r[0, 0] * baseline_m], [0.0], [0.0]])])

    pts_l = corners_l.reshape(-1, 2).T   # 2xN
    pts_r = corners_r.reshape(-1, 2).T   # 2xN

    pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T  # Nx3

    return corners_l.reshape(-1, 2), corners_r.reshape(-1, 2), pts_3d

def chessboard_3d_distance_stats(pts_3d, pattern_size, square_size_m):
    cols, rows = pattern_size

    def idx(x, y):
        return y * cols + x

    errors = []

    for y in range(rows):
        for x in range(cols):
            i = idx(x, y)

            if x + 1 < cols:
                d = np.linalg.norm(pts_3d[i] - pts_3d[idx(x + 1, y)])
                errors.append(d - square_size_m)

            if y + 1 < rows:
                d = np.linalg.norm(pts_3d[i] - pts_3d[idx(x, y + 1)])
                errors.append(d - square_size_m)

    errors = np.asarray(errors)
    return {
        "num_pairs": len(errors),
        "mean_abs_error_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "mean_signed_error_m": float(np.mean(errors)),
    }


if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parent.parent

    dataset_dir = parent_dir / 'dataset_11032026'
    depth_dir = dataset_dir / 'stereo_4k_calibration' / 'rgb'
    out_dir = parent_dir / "out" / "cameras_parameters"

    calib_dict = out_dir / "calib_data.npy"
    calib_dict = load_dict(calib_dict)

    _, _, rect_data = get_rectify_functions(calib_dict)

    K_rect_l = rect_data["K_rect_l"]
    K_rect_r = rect_data["K_rect_r"]
    baseline_m = np.linalg.norm(calib_dict["tvec"].reshape(-1)) / 1000.0
    print("BASELINE in meters:", baseline_m)
    imgs_l, imgs_r = load_l_r_images_rectified(
        calib_dict, depth_dir, max_imgs=20
    )

    for img_l, img_r in zip(imgs_l, imgs_r):
        if img_l.get_image_number() != 15: continue
        corners_l, corners_r, pts_3d = triangulate_chessboard_corners(
            img_l.img, img_r.img,
            K_rect_l, K_rect_r,
            baseline_m,
            pattern_size=(8, 5),
        )

        stats = chessboard_3d_distance_stats(
            pts_3d,
            pattern_size=(8,5),
            square_size_m=0.030,
        )

        print(stats)


