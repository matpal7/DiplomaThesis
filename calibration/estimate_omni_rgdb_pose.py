import argparse
from pathlib import Path
import json

import cv2
import numpy as np


def load_realsense_calibration(yaml_path: Path):
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open RGB/RGBD calibration file: {yaml_path}")

    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()

    if K is None or D is None:
        raise ValueError("RGBD calibration YAML must contain nodes 'K' and 'D'.")

    return K.astype(np.float64), D.astype(np.float64)


def load_omni_calibration(npz_or_npy_path: Path, camera_key: str):
    data = np.load(str(npz_or_npy_path), allow_pickle=True).item()

    if camera_key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Camera key '{camera_key}' not found in {npz_or_npy_path}. Available keys: {available}")

    cam = data[camera_key]
    K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(cam["D"], dtype=np.float64).reshape(-1, 1)
    xi = np.asarray(cam["xi"], dtype=np.float64).reshape(1, 1)
    return K, D, xi




def _xi_scalar(xi):
    return float(np.asarray(xi, dtype=np.float64).reshape(-1)[0])


def undistort_omni_points_compat(corners_pix, K, D, xi):
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


def project_omni_points_compat(objp, rvec, tvec, K, D, xi):
    obj = objp.reshape(-1, 1, 3)
    try:
        return cv2.omnidir.projectPoints(obj, rvec, tvec, K, _xi_scalar(xi), D)
    except cv2.error:
        return cv2.omnidir.projectPoints(obj, rvec, tvec, K, np.asarray(xi, dtype=np.float64).reshape(1, 1), D)

def build_object_points(board_cols: int, board_rows: int, square_size: float):
    objp = np.zeros((board_cols * board_rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_corners(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    if not ok:
        return None
    return corners.astype(np.float64)


def solve_pose_rgb(objp, corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    return rvec, tvec


def solve_pose_omni(objp, corners_pix, K, D, xi):
    corners_norm = undistort_omni_points_compat(corners_pix, K, D, xi)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=objp,
        imagePoints=corners_norm,
        cameraMatrix=np.eye(3, dtype=np.float64),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return None, None
    return rvec, tvec


def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_angle_deg(R1, R2):
    dR = R1.T @ R2
    cos_theta = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def reproj_error_rgb(objp, corners, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    e = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return float(np.mean(e))


def reproj_error_omni(objp, corners, rvec, tvec, K, D, xi):
    proj, _ = project_omni_points_compat(objp, rvec, tvec, K, D, xi)
    e = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return float(np.mean(e))


def rotmat_to_quat(R):
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def average_transforms(transforms):
    if len(transforms) == 1:
        return transforms[0]

    A = np.zeros((4, 4), dtype=np.float64)
    ts = []
    for T in transforms:
        q = rotmat_to_quat(T[:3, :3])
        if q[3] < 0:
            q = -q
        A += np.outer(q, q)
        ts.append(T[:3, 3])

    _, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, -1]
    q_avg /= np.linalg.norm(q_avg)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotmat(q_avg)
    T[:3, 3] = np.mean(np.array(ts), axis=0)
    return T


def median_rotation_neighborhood(rotations):
    """Return the rotation with the smallest median angular distance to all others."""
    if len(rotations) == 1:
        return rotations[0]

    best_idx = 0
    best_med = np.inf
    for i, Ri in enumerate(rotations):
        ang = [rotation_angle_deg(Ri, Rj) for Rj in rotations]
        med = float(np.median(ang))
        if med < best_med:
            best_med = med
            best_idx = i
    return rotations[best_idx]


def compute_median_reference(candidates):
    Ts = [x["T"] for x in candidates]
    translations = np.array([T[:3, 3] for T in Ts], dtype=np.float64)
    rotations = [T[:3, :3] for T in Ts]

    t_med = np.median(translations, axis=0)
    R_med = median_rotation_neighborhood(rotations)

    T_ref = np.eye(4, dtype=np.float64)
    T_ref[:3, :3] = R_med
    T_ref[:3, 3] = t_med
    return T_ref


def load_pairs(data_dir: Path, omni_suffix: str, rgbd_suffix: str):
    omni_files = sorted(data_dir.glob(f"*{omni_suffix}"))
    rgbd_files = sorted(data_dir.glob(f"*{rgbd_suffix}"))

    rgbd_by_prefix = {p.name[: -len(rgbd_suffix)]: p for p in rgbd_files}

    pairs = []
    for omni in omni_files:
        prefix = omni.name[: -len(omni_suffix)]
        rgbd = rgbd_by_prefix.get(prefix)
        if rgbd is not None:
            pairs.append((omni, rgbd))

    return pairs




def _median_mad_threshold(values, sigma_mult=2.5, fallback=0.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float(fallback)

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    robust_sigma = 1.4826 * mad
    return med + sigma_mult * robust_sigma


def select_inliers(candidates, T_ref, trans_thr_mm, rot_thr_deg, neighborhood_scale=1.0):
    t_devs = [float(np.linalg.norm(x["T"][:3, 3] - T_ref[:3, 3])) for x in candidates]
    r_devs = [float(rotation_angle_deg(T_ref[:3, :3], x["T"][:3, :3])) for x in candidates]

    adaptive_trans_thr = _median_mad_threshold(t_devs, sigma_mult=2.0) * neighborhood_scale
    adaptive_rot_thr = _median_mad_threshold(r_devs, sigma_mult=2.0) * neighborhood_scale

    eff_trans_thr = min(trans_thr_mm, adaptive_trans_thr) if adaptive_trans_thr > 0 else trans_thr_mm
    eff_rot_thr = min(rot_thr_deg, adaptive_rot_thr) if adaptive_rot_thr > 0 else rot_thr_deg

    inliers = []
    for x, t_dev, r_dev in zip(candidates, t_devs, r_devs):
        if t_dev <= eff_trans_thr and r_dev <= eff_rot_thr:
            y = dict(x)
            y["t_dev_mm"] = t_dev
            y["r_dev_deg"] = r_dev
            inliers.append(y)
    return inliers, eff_trans_thr, eff_rot_thr


def inlier_failure_message(candidates, min_inliers, trans_thr, rot_thr):
    if not candidates:
        return "No candidates available for inlier filtering."

    Ts = [x["T"] for x in candidates]
    tdevs = []
    rdevs = []
    for i, Ti in enumerate(Ts):
        for j, Tj in enumerate(Ts):
            if i >= j:
                continue
            tdevs.append(float(np.linalg.norm(Ti[:3, 3] - Tj[:3, 3])))
            rdevs.append(float(rotation_angle_deg(Ti[:3, :3], Tj[:3, :3])))

    if tdevs and rdevs:
        p50_t, p75_t = np.percentile(tdevs, [50, 75])
        p50_r, p75_r = np.percentile(rdevs, [50, 75])
        return (
            f"Too few inliers for current thresholds (translation={trans_thr:.1f} mm, rotation={rot_thr:.2f} deg, required={min_inliers}). "
            f"Observed pairwise spread: t median={p50_t:.1f} mm, t p75={p75_t:.1f} mm, "
            f"r median={p50_r:.2f} deg, r p75={p75_r:.2f} deg. "
            f"Try: --max-trans-dev-mm {max(trans_thr, p75_t):.1f} --max-rot-dev-deg {max(rot_thr, p75_r):.2f} --min-inliers 2, "
            "or enable --auto-relax."
        )

    return (
        f"Too few inliers for current thresholds (translation={trans_thr:.1f} mm, rotation={rot_thr:.2f} deg, required={min_inliers}). "
        "Try increasing --max-trans-dev-mm / --max-rot-dev-deg, lowering --min-inliers, or enabling --auto-relax."
    )

def parse_args():
    p = argparse.ArgumentParser(description="Estimate pose between an omnidirectional camera and RGBD camera from chessboard images.")
    p.add_argument("--data-dir", required=False, type=Path, help="Folder with synchronized image pairs.")
    p.add_argument("--omni-calib", required=False, type=Path, help=".npy with omnidirectional calibration dictionary.")
    p.add_argument("--rgbd-calib", required=False, type=Path, help="YAML with RealSense calibration (K,D).")
    p.add_argument("--camera-key", default="left", help="Key inside omni calib dictionary, e.g. left/right.")
    p.add_argument("--board-cols", type=int, default=8, help="Chessboard inner corners in X direction.")
    p.add_argument("--board-rows", type=int, default=6, help="Chessboard inner corners in Y direction.")
    p.add_argument("--square-size", type=float, default=21.0, help="Chessboard square size in mm (or chosen metric).")
    p.add_argument("--omni-suffix", default="_left.png", help="Suffix for omnidirectional camera images.")
    p.add_argument("--rgbd-suffix", default="_realsense.png", help="Suffix for RGBD camera images.")
    p.add_argument("--max-rgb-reproj", type=float, default=2.5, help="Max RGB reprojection error (pixels).")
    p.add_argument("--max-omni-reproj", type=float, default=3.0, help="Max omni reprojection error (pixels).")
    p.add_argument("--max-trans-dev-mm", type=float, default=250.0, help="Max translation deviation from reference transform.")
    p.add_argument("--max-rot-dev-deg", type=float, default=8.0, help="Max rotation deviation from reference transform.")
    p.add_argument("--min-inliers", type=int, default=3, help="Minimum inlier pairs required to accept the estimate.")
    p.add_argument("--auto-relax", action="store_true", help="Automatically relax transform inlier thresholds if too few inliers are found.")
    p.add_argument("--auto-relax-steps", type=int, default=4, help="Number of auto-relax rounds to try.")
    p.add_argument("--trans-relax-factor", type=float, default=1.5, help="Multiplier applied to translation threshold per auto-relax step.")
    p.add_argument("--rot-relax-factor", type=float, default=1.4, help="Multiplier applied to rotation threshold per auto-relax step.")
    p.add_argument("--output", type=Path, default=Path("calibration/omni_to_rgbd_pose.json"), help="Output JSON path.")
    p.add_argument("--show", action="store_true", help="Visualize detected corners.")

    return p.parse_args()


def main():
    args = parse_args()
    parent_dir = Path(__file__).resolve().parent.parent
    dataset_dir = parent_dir / "dataset_05032026"
    args.data_dir = dataset_dir / "depth" / "rgb"

    out_dir = parent_dir / "NICO" / "out_2"
    args.omni_calib = out_dir / "calib_data_left.npy"

    args.rgbd_calib = parent_dir / "calibration" / "realsense_calibration.yaml"

    K_omni, D_omni, xi_omni = load_omni_calibration(args.omni_calib, args.camera_key)
    K_rgb, D_rgb = load_realsense_calibration(args.rgbd_calib)

    pairs = load_pairs(args.data_dir, args.omni_suffix, args.rgbd_suffix)
    if not pairs:
        raise RuntimeError("No matching image pairs found. Check suffixes and filenames.")

    objp = build_object_points(args.board_cols, args.board_rows, args.square_size)
    pattern_size = (args.board_cols, args.board_rows)

    candidates = []
    for omni_path, rgbd_path in pairs:
        img_omni = cv2.imread(str(omni_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.imread(str(rgbd_path), cv2.IMREAD_COLOR)
        if img_omni is None or img_rgb is None:
            continue

        corners_omni = find_corners(img_omni, pattern_size)
        corners_rgb = find_corners(img_rgb, pattern_size)
        if corners_omni is None or corners_rgb is None:
            continue

        if args.show:
            vis_o = cv2.drawChessboardCorners(img_omni.copy(), pattern_size, corners_omni.astype(np.float32), True)
            vis_r = cv2.drawChessboardCorners(img_rgb.copy(), pattern_size, corners_rgb.astype(np.float32), True)
            cv2.imshow("omni | rgbd", cv2.hconcat([cv2.resize(vis_o, (640, 480)), cv2.resize(vis_r, (640, 480))]))
            cv2.waitKey(1)

        rvec_o, tvec_o = solve_pose_omni(objp, corners_omni, K_omni, D_omni, xi_omni)
        rvec_r, tvec_r = solve_pose_rgb(objp, corners_rgb, K_rgb, D_rgb)

        if rvec_o is None or rvec_r is None:
            continue

        e_o = reproj_error_omni(objp, corners_omni, rvec_o, tvec_o, K_omni, D_omni, xi_omni)
        e_r = reproj_error_rgb(objp, corners_rgb, rvec_r, tvec_r, K_rgb, D_rgb)

        if e_o > args.max_omni_reproj or e_r > args.max_rgb_reproj:
            continue

        T_omni_board = rt_to_T(rvec_o, tvec_o)
        T_rgb_board = rt_to_T(rvec_r, tvec_r)
        T_omni_rgb = T_omni_board @ inv_T(T_rgb_board)

        candidates.append({
            "pair": (omni_path.name, rgbd_path.name),
            "T": T_omni_rgb,
            "err_omni": e_o,
            "err_rgb": e_r,
        })

    if args.show:
        cv2.destroyAllWindows()

    if len(candidates) < 3:
        raise RuntimeError(f"Only {len(candidates)} valid pairs found, need at least 3.")

    T_ref = compute_median_reference(candidates)

    trans_thr = args.max_trans_dev_mm
    rot_thr = args.max_rot_dev_deg
    neighborhood_scale = 1.0
    inliers, eff_trans_thr, eff_rot_thr = select_inliers(
        candidates, T_ref, trans_thr, rot_thr, neighborhood_scale=neighborhood_scale
    )

    if len(inliers) < args.min_inliers and args.auto_relax:
        for _ in range(args.auto_relax_steps):
            trans_thr *= args.trans_relax_factor
            rot_thr *= args.rot_relax_factor
            neighborhood_scale *= 1.15
            inliers, eff_trans_thr, eff_rot_thr = select_inliers(
                candidates, T_ref, trans_thr, rot_thr, neighborhood_scale=neighborhood_scale
            )
            if len(inliers) >= args.min_inliers:
                print(
                    f"Auto-relax accepted: translation threshold={eff_trans_thr:.1f} mm, "
                    f"rotation threshold={eff_rot_thr:.2f} deg, inliers={len(inliers)}"
                )
                break

    if len(inliers) < args.min_inliers:
        raise RuntimeError(
            f"Only {len(inliers)} inliers after robust filtering, need at least {args.min_inliers}. "
            + inlier_failure_message(candidates, args.min_inliers, eff_trans_thr, eff_rot_thr)
        )

    T_final = average_transforms([x["T"] for x in inliers])

    output = {
        "description": "Transform from RGBD camera to omnidirectional camera frame.",
        "T_omni_rgbd": T_final.tolist(),
        "R_omni_rgbd": T_final[:3, :3].tolist(),
        "t_omni_rgbd": T_final[:3, 3].tolist(),
        "num_pairs_total": len(pairs),
        "num_pairs_valid": len(candidates),
        "num_pairs_inliers": len(inliers),
        "inlier_thresholds": {
            "translation_mm": eff_trans_thr,
            "rotation_deg": eff_rot_thr,
        },
        "pairs": [
            {
                "omni": x["pair"][0],
                "rgbd": x["pair"][1],
                "err_omni_px": x["err_omni"],
                "err_rgb_px": x["err_rgb"],
                "t_dev_mm": x.get("t_dev_mm", None),
                "r_dev_deg": x.get("r_dev_deg", None),
            }
            for x in inliers
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print("=== Final transform (RGBD -> OMNI) ===")
    print("R:")
    print(T_final[:3, :3])
    print("t [mm]:")
    print(T_final[:3, 3])
    print(f"Used {len(inliers)} / {len(pairs)} pairs")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
