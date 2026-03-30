from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from code.calibration.ChArUco.charuco_relative_pose_pnp import find_images
from code.calibration.calibrate_stereo import extract_chessboard_points
from code.image import get_undistort_functions
from code.utils import get_l_r_image_fnames, load_dict

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def _build_object_points(chessboard_x, chessboard_y, chessboard_dim):
    obj_coords = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    obj_coords[:, :2] = (
        chessboard_dim * np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)
    )
    return obj_coords


def _collect_calibration_points(
    image_list,
    image_transform_fn=None,
    window_name="Calibration",
    chessboard_x=7,
    chessboard_y=4,
    chessboard_dim=31.0,
    debug=0,
    frame_size=(1280, 720)
):
    obj_coords = _build_object_points(chessboard_x, chessboard_y, chessboard_dim)

    obj_pts = []
    img_pts = []
    image_size = None

    for fname in tqdm(image_list, total=len(image_list)):
        img = cv2.imread(fname)
        if img is None:
            if debug > 0:
                print("Could not read:", fname)
            continue

        if image_transform_fn is not None:
            img = image_transform_fn(img)
            img = cv2.resize(img, frame_size)

            if img is None:
                if debug > 0:
                    print("Transform returned None:", fname)
                continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_x, chessboard_y), None)

        if ret:
            obj_pts.append(obj_coords.copy())
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            img_pts.append(corners)

            if image_size is None:
                image_size = gray.shape[::-1]

            if debug > 0:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, (chessboard_x, chessboard_y), corners, ret)
                vis = cv2.resize(vis, (640, 480))
                cv2.imshow(window_name, vis)

                if debug >= 2:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(1)
        elif debug > 0:
            print("Corners not found!", fname)

    cv2.destroyAllWindows()

    return obj_pts, img_pts, image_size


def _run_calibration(obj_pts, img_pts, image_size, label="camera"):
    if len(obj_pts) < 5:
        raise RuntimeError(f"Not enough valid images for calibration. Found only {len(obj_pts)}")

    print(f"Calibrating {label} on {len(obj_pts)} images")

    retval, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts,
        img_pts,
        image_size,
        None,
        None,
    )

    print("=== Calibration finished ===")
    print("Camera:", label)
    print("Reprojection error:", retval)
    print("K:\n", K)
    print("D:\n", D)

    return {
        "K": K,
        "D": D,
        "image_size": image_size,
        "reprojection_error": retval,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "num_images_used": len(obj_pts),
    }


def _save_calibration_yaml(
    out_path,
    data,
    extra_fields=None,
):
    extra_fields = extra_fields or {}

    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open file for writing: {out_path}")

    for key, value in extra_fields.items():
        fs.write(key, value)

    fs.write("K", data["K"])
    fs.write("D", data["D"])
    for attribute in("xi", "map1", "map2"):
        if attribute in data.keys():
            fs.write(attribute, data[attribute])

    fs.write("image_width", data["image_size"][0])
    fs.write("image_height", data["image_size"][1])
    fs.write("reprojection_error", data["reprojection_error"])
    fs.write("num_images_used", data["num_images_used"])
    fs.release()


def calibrate_on_undistored(
    img_folder,
    calib_dict,
    out_dir,
    camera_side="left",
    chessboard_x=6,
    chessboard_y=4,
    chessboard_dim=54.0,
    max_imgs=20,
    debug=0,
    frame_size=(1280, 720)
):
    if camera_side not in ("left", "right"):
        raise ValueError("camera_side must be 'left' or 'right'")

    undistort_l, undistort_r = get_undistort_functions(calib_dict)
    images_l, images_r = get_l_r_image_fnames(img_folder, max_imgs)

    if camera_side == "left":
        image_list = images_l
        undistort_fn = undistort_l
    else:
        image_list = images_r
        undistort_fn = undistort_r

    obj_pts, img_pts, image_size = _collect_calibration_points(
        image_list=image_list,
        image_transform_fn=undistort_fn,
        window_name=f"Calibration {camera_side}",
        chessboard_x=chessboard_x,
        chessboard_y=chessboard_y,
        chessboard_dim=chessboard_dim,
        debug=debug,
        frame_size=frame_size,
    )

    calib = _run_calibration(obj_pts, img_pts, image_size, label=f"{camera_side} undistorted")

    rectified_calib_dict = {
        "model": "pinhole_rectified",
        "camera_side": camera_side,
        "source_img_folder": str(img_folder),
        "chessboard_x": chessboard_x,
        "chessboard_y": chessboard_y,
        "chessboard_dim": chessboard_dim,
        **calib,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{camera_side}_NICO_{frame_size[0]}x{frame_size[1]}.yaml"


    _save_calibration_yaml(
        out_path=out_path,
        data=rectified_calib_dict,
        extra_fields={
            "model": rectified_calib_dict["model"],
            "camera_side": rectified_calib_dict["camera_side"],
            "chessboard_x": rectified_calib_dict["chessboard_x"],
            "chessboard_y": rectified_calib_dict["chessboard_y"],
            "chessboard_dim": rectified_calib_dict["chessboard_dim"],
        },
    )

    print("Saved calibration to:", out_path)
    return rectified_calib_dict


def calibrate(
    img_folder,
    out_dir,
    suffix="realsense",
    chessboard_x=6,
    chessboard_y=4,
    chessboard_dim=54.0,
    max_imgs=20,
    debug=0,
    file_suffix="calibration",
    frame_size=(1280, 720),
    alpha=0.0,
):
    images = find_images(img_folder, suffix)[:max_imgs]
    print(img_folder)

    obj_pts, img_pts, image_size = _collect_calibration_points(
        image_list=images,
        image_transform_fn=None,
        window_name=f"Calibration {suffix}",
        chessboard_x=chessboard_x,
        chessboard_y=chessboard_y,
        chessboard_dim=chessboard_dim,
        debug=debug,
    )

    calib = _run_calibration(obj_pts, img_pts, image_size, label=suffix)

    K = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calib["D"], dtype=np.float64).reshape(-1, 1)

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        D,
        image_size,
        alpha,
        image_size,
    )

    new_D = np.zeros_like(D, dtype=np.float64)

    rectified_calib_dict = {
        "camera": suffix,
        "source_img_folder": str(img_folder),
        "chessboard_x": chessboard_x,
        "chessboard_y": chessboard_y,
        "chessboard_dim": chessboard_dim,
        **calib,
        "new_K": new_K,
        "new_D": new_D,
        "undistort_alpha": float(alpha),
        "undistort_roi": np.asarray(roi, dtype=np.int32),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{suffix}_{file_suffix}_{frame_size[0]}x{frame_size[1]}.yaml"

    _save_calibration_yaml(
        out_path=out_path,
        data=rectified_calib_dict,
        extra_fields={
            "camera": rectified_calib_dict["camera"],
            "chessboard_x": rectified_calib_dict["chessboard_x"],
            "chessboard_y": rectified_calib_dict["chessboard_y"],
            "chessboard_dim": rectified_calib_dict["chessboard_dim"],
            "K_new": rectified_calib_dict["new_K"],
            "D_new": rectified_calib_dict["new_D"],
            "undistort_alpha": rectified_calib_dict["undistort_alpha"],
            "undistort_roi": rectified_calib_dict["undistort_roi"],
        },
    )

    print("Saved calibration to:", out_path)
    return rectified_calib_dict


def calibrate_mono(
    img_folder,
    camera_side="left",
    chessboard_x=6,
    chessboard_y=4,
    chessboard_dim=54.0,
    max_imgs=None):
    obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r = extract_chessboard_points(img_folder,
                                                                                    chessboard_x=chessboard_x,
                                                                                    chessboard_y=chessboard_y,
                                                                                    chessboard_dim=chessboard_dim,
                                                                                    debug=debug, max_imgs=max_imgs)

    img_pts = img_pts_l
    img_dim = img_dim_l
    if camera_side == "right":
        img_pts = img_pts_r
        img_dim = img_dim_r

    retval, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(obj_pts, img_pts, img_dim, None, None,
                                                                None, 0,
                                                                CRITERIA)

    print("retval: " + camera_side, retval)
    map1, map2 = cv2.omnidir.initUndistortRectifyMap(K, D, xi, np.eye(3), K, img_dim, cv2.CV_16SC2,
                                                     cv2.omnidir.RECTIFY_PERSPECTIVE)

    camera_dict = {'K': K, 'xi': xi, 'D': D, 'map1': map1, 'map2': map2, 'img_size': img_dim}

    return camera_dict

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[3]
    date = "28032026"
    dataset_dir = parent_dir / "datasets" / f"dataset_{date}"
    calib_imgs_dir = dataset_dir / "stereo_4k_calibration_stereo" / "rgb"
    relative_pose_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    out_dir = parent_dir / "out" / f"out_{date}" / "cameras_parameters"
    debug = 0

    # calib_dict = load_dict(out_dir / "calib_data.npy")


    calibrate(relative_pose_dir, str(out_dir), suffix="realsense", chessboard_dim=45.0, max_imgs=280, chessboard_x=7,
                            chessboard_y=5, debug=debug)
    calibrate(relative_pose_dir, str(out_dir), suffix="zed", chessboard_dim=45.0, max_imgs=28, chessboard_x=7,
                            chessboard_y=5, debug=debug)

    # calibrate(relative_pose_dir, str(out_dir), suffix="left", chessboard_dim=44.0, max_imgs=280, chessboard_x=7,
    #           chessboard_y=5, debug=debug)

    # left_dict = calibrate_mono(relative_pose_dir, camera_side="left", chessboard_dim=44.0,
    #                         max_imgs=15, chessboard_x=7, chessboard_y=5)

    # print("calibrate_mono", left_di44ct)

    # right_dict = calibrate_mono(calib_imgs_dir, camera_side="right", chessboard_dim=0.030,
    #                            max_imgs=40, chessboard_x=8, chessboard_y=5)