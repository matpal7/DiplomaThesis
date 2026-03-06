import glob
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.excentric_calibration import load_camera_calibration
from planar_nico_vision.calibrate import undistort_image
from utils import load_dict


# ============================================================
# USER INPUT
# ============================================================

# Chessboard settings
pattern_size = (8, 6)       # inner corners (cols, rows)
square_size = 23.0          # mm

parent_dir = Path(__file__).resolve().parent.parent
dataset_dir = parent_dir / "dataset_05032026"
calib_imgs_dir = dataset_dir / "depth" / "rgb" / "cluster1"

# Omni calibration
out_dir = parent_dir / "NICO" / "out_2"
calib_left = load_dict(out_dir / "calib_data_left.npy")
K_omni = np.asarray(calib_left["left"]["K"], dtype=np.float64).reshape(3, 3)
xi_omni = np.asarray(calib_left["left"]["xi"], dtype=np.float64).reshape(1, 1)
D_omni = np.asarray(calib_left["left"]["D"], dtype=np.float64).reshape(-1, 1)

# Because your rectification maps were created with P = K,
# the rectified omni image uses K_rect = K
K_omni_rect = np.asarray(calib_left["left"]["K"], dtype=np.float64).reshape(3, 3)
D_omni_rect = np.zeros((4, 1), dtype=np.float64)

# RGB camera intrinsics
calib_file = parent_dir / "calibration" / "realsense_calibration.yaml"
calib_RGBD = load_camera_calibration(calib_file)
K_rgb = np.asarray(calib_RGBD["K"], dtype=np.float64).reshape(3, 3)
D_rgb = np.asarray(calib_RGBD["D"], dtype=np.float64).reshape(-1, 1)


# ============================================================
# HELPERS
# ============================================================

def create_chessboard_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(img, pattern_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=0)
    if not ok:
        return None
    return corners.astype(np.float64)


def draw_corners(img, pattern_size, corners):
    vis = img.copy()

    if corners is None:
        return vis

    # Ensure correct shape for OpenCV
    corners_draw = corners.reshape(-1, 1, 2).astype(np.float32)

    if corners_draw.shape[0] == pattern_size[0] * pattern_size[1]:
        cv2.drawChessboardCorners(vis, pattern_size, corners_draw, True)

    return vis


def solve_board_pose(objp, corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=objp,
        imagePoints=corners,
        cameraMatrix=K,
        distCoeffs=D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None
    return rvec, tvec


def reprojection_error(objp, corners, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    obs = corners.reshape(-1, 2)
    err = np.linalg.norm(proj - obs, axis=1)
    return float(np.mean(err))


def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def average_transforms(T_list):
    Rs = [T[:3, :3] for T in T_list]
    ts = np.array([T[:3, 3] for T in T_list], dtype=np.float64)

    R_mean = Rotation.from_matrix(Rs).mean().as_matrix()
    t_mean = np.mean(ts, axis=0)

    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean


def rotation_error_deg(T, T_ref):
    dR = T_ref[:3, :3].T @ T[:3, :3]
    return np.degrees(Rotation.from_matrix(dR).magnitude())


# ============================================================
# LOAD IMAGE PAIRS
# ============================================================

omni_paths = sorted(Path(p) for p in glob.glob(str(calib_imgs_dir / "*_left.png")))
rgb_paths = sorted(Path(p) for p in glob.glob(str(calib_imgs_dir / "*_realsense.png")))

assert len(omni_paths) == len(rgb_paths), "Different number of omni and RGB images."

objp = create_chessboard_object_points(pattern_size, square_size)
pair_results = []

for omni_path, rgb_path in zip(omni_paths, rgb_paths):
    img_omni_raw = cv2.imread(str(omni_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if img_omni_raw is None or img_rgb is None:
        print(f"Skipping unreadable pair: {omni_path.name} / {rgb_path.name}")
        continue

    # Rectified perspective omni image
    img_omni_rect = undistort_image(img_omni_raw, calib_left, eye="left")

    # IMPORTANT: do not resize before corner detection / solvePnP
    corners_omni = detect_corners(img_omni_rect, pattern_size)
    corners_rgb = detect_corners(img_rgb, pattern_size)

    # Visualization
    vis_omni = draw_corners(img_omni_rect, pattern_size, corners_omni)
    vis_rgb = draw_corners(img_rgb, pattern_size, corners_rgb)
    vis_omni_small = cv2.resize(vis_omni, (640, 480))
    vis_rgb_small = cv2.resize(vis_rgb, (640, 480))
    cv2.imshow("corners", cv2.hconcat([vis_omni_small, vis_rgb_small]))
    key = cv2.waitKey(100) & 0xFF
    if key == 27:
        break

    if corners_omni is None or corners_rgb is None:
        print(f"Skipping pair: {omni_path.name} / {rgb_path.name} (corners not found)")
        continue

    # Board pose in rectified omni camera
    rvec_o, tvec_o = solve_board_pose(objp, corners_omni, K_omni_rect, D_omni_rect)
    if rvec_o is None:
        print(f"Skipping pair: {omni_path.name} / {rgb_path.name} (omni solvePnP failed)")
        continue

    # Board pose in RGB camera
    rvec_r, tvec_r = solve_board_pose(objp, corners_rgb, K_rgb, D_rgb)
    if rvec_r is None:
        print(f"Skipping pair: {omni_path.name} / {rgb_path.name} (RGB solvePnP failed)")
        continue

    # Reprojection error
    err_omni = reprojection_error(objp, corners_omni, rvec_o, tvec_o, K_omni_rect, D_omni_rect)
    err_rgb = reprojection_error(objp, corners_rgb, rvec_r, tvec_r, K_rgb, D_rgb)

    T_omni_board = rt_to_T(rvec_o, tvec_o)
    T_rgb_board = rt_to_T(rvec_r, tvec_r)

    # Transform points from RGB frame -> rectified omni frame
    T_omni_rgb = T_omni_board @ invert_T(T_rgb_board)

    pair_results.append({
        "pair_name": (omni_path.name, rgb_path.name),
        "T": T_omni_rgb,
        "err_omni": err_omni,
        "err_rgb": err_rgb,
    })

cv2.destroyAllWindows()

print(f"\nUsable pairs before filtering: {len(pair_results)}")

if len(pair_results) < 5:
    raise RuntimeError("Too few valid image pairs.")


# ============================================================
# FILTER 1: REPROJECTION ERROR
# ============================================================

reproj_inliers = []
for x in pair_results:
    if x["err_omni"] < 2.5 and x["err_rgb"] < 2.5:
        reproj_inliers.append(x)

print(f"Pairs after reprojection filtering: {len(reproj_inliers)}")

if len(reproj_inliers) < 3:
    raise RuntimeError("Too few pairs after reprojection filtering.")

T_initial = average_transforms([x["T"] for x in reproj_inliers])

def transform_distance_components(T1, T2):
    t_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    r_err = rotation_error_deg(T1, T2)
    return t_err, r_err

# ============================================================
# FILTER 2: ROBUST CONSENSUS ON TRANSFORMS
# ============================================================

print("\n========== PAIRWISE TRANSFORM CONSENSUS ==========")

Ts = [x["T"] for x in reproj_inliers]
n = len(Ts)

# Find the transform with the best agreement to the others
best_idx = None
best_score = None

for i in range(n):
    score = 0.0
    for j in range(n):
        if i == j:
            continue
        t_err, r_err = transform_distance_components(Ts[i], Ts[j])
        score += t_err + 20.0 * r_err   # tune weight if needed
    print(f"candidate {i}: total score = {score:.2f}")

    if best_score is None or score < best_score:
        best_score = score
        best_idx = i

T_ref = Ts[best_idx]
print(f"\nChosen reference pair: {reproj_inliers[best_idx]['pair_name']}")

final_inliers = []
for x in reproj_inliers:
    t_err, r_err = transform_distance_components(x["T"], T_ref)
    t = x["T"][:3, 3]

    print(
        f"{x['pair_name'][0]} <-> {x['pair_name'][1]} | "
        f"t = [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm | "
        f"t_err_to_ref = {t_err:.1f} mm | "
        f"r_err_to_ref = {r_err:.2f} deg | "
        f"err_omni = {x['err_omni']:.3f}px | err_rgb = {x['err_rgb']:.3f}px"
    )

    # relaxed thresholds
    if t_err < 300.0 and r_err < 10.0:
        final_inliers.append(x)

print(f"\nPairs after robust transform filtering: {len(final_inliers)}")

if len(final_inliers) < 3:
    raise RuntimeError("Too few inliers after robust transform filtering.")

T_final = average_transforms([x["T"] for x in final_inliers])


# ============================================================
# RESULTS
# ============================================================

print("\n========== FINAL RESULT ==========")
print("Transform from RGB camera to RECTIFIED OMNI camera:")
print("T_omni_rgb =")
print(T_final)

print("\nRotation R_omni_rgb =")
print(T_final[:3, :3])

print("\nTranslation t_omni_rgb [mm] =")
print(T_final[:3, 3])

print("\n========== INLIER PAIRS ==========")
for x in final_inliers:
    t = x["T"][:3, 3]
    print(
        f"{x['pair_name'][0]} <-> {x['pair_name'][1]} | "
        f"err_omni={x['err_omni']:.3f}px | "
        f"err_rgb={x['err_rgb']:.3f}px | "
        f"|t|={np.linalg.norm(t):.1f} mm"
    )

translations = np.array([x["T"][:3, 3] for x in final_inliers], dtype=np.float64)
print("\nTranslation mean [mm]:", translations.mean(axis=0))
print("Translation std  [mm]:", translations.std(axis=0))

rotations = Rotation.from_matrix([x["T"][:3, :3] for x in final_inliers])
rot_mean = Rotation.from_matrix(T_final[:3, :3])
angles_deg = []
for r in rotations:
    dR = rot_mean.inv() * r
    angles_deg.append(np.degrees(dR.magnitude()))

angles_deg = np.array(angles_deg)
print("Rotation deviation from mean [deg]: mean =", angles_deg.mean(), "std =", angles_deg.std())


# ============================================================
# OPTIONAL: inverse transform
# ============================================================

T_rgb_omni = invert_T(T_final)
print("\nTransform from RECTIFIED OMNI camera to RGB camera:")
print("T_rgb_omni =")
print(T_rgb_omni)