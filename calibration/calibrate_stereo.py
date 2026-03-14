import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from calibration.image import get_undistort_functions
from calibration.undistored_images import show_undistorted_images
from utils import get_l_r_image_fnames, load_dict, save_dict

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
frameSize = (640, 480)
width = 1280
height = 720

def extract_chessboard_points(calib_img_folder, undistort_l=None, undistort_r=None, chessboard_x=8, chessboard_y=6,
                              chessboard_dim=21.0, debug=0, max_imgs=None):
    obj_coords = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    obj_coords[:, :2] = chessboard_dim * np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    obj_pts = []
    img_pts_l = []
    img_pts_r = []
    print("IMAGE FOLDER:", calib_img_folder)
    images_l, images_r = get_l_r_image_fnames(calib_img_folder, max_imgs)
    print("Finding calibration patterns")
    print(len(images_l))
    for fname_l, fname_r in tqdm(zip(images_l, images_r), total=len(images_l)):
        img_l = cv2.imread(fname_l)
        if undistort_l is not None:
            img_l = undistort_l(img_l)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (chessboard_x, chessboard_y), None)

        img_r = cv2.imread(fname_r)
        if undistort_r is not None:
            img_r = undistort_r(img_r)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (chessboard_x, chessboard_y), None)

        if ret_r and ret_l:
            obj_pts.append(obj_coords)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)
            img_pts_l.append(corners_l)
            img_pts_r.append(corners_r)

            if debug > 0:
                cv2.drawChessboardCorners(img_l, (chessboard_x, chessboard_y), corners_l, ret_l)
                cv2.drawChessboardCorners(img_r, (chessboard_x, chessboard_y), corners_r, ret_r)

                resized_frame_L = cv2.resize(img_l, (640, 480))
                resized_frame_R = cv2.resize(img_r, (640, 480))
                img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
                cv2.imshow('img r', img_concat_h)
                cv2.waitKey(1)

        elif debug > 0:
            print("Corners not found!" ,fname_l)

    obj_pts = np.expand_dims(np.array(obj_pts, dtype=np.float64), -3)
    img_pts_l = np.array(img_pts_l, dtype=np.float64)
    img_pts_r = np.array(img_pts_r, dtype=np.float64)

    img_dim_l = (img_l.shape[1], img_l.shape[0])
    img_dim_r = (img_r.shape[1], img_r.shape[0])

    return obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r





def calibrate(calib_img_folder, chessboard_x=7, chessboard_y=4, chessboard_dim=31.0, debug=0, max_imgs=None):
    obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r = extract_chessboard_points(calib_img_folder,
                                                                                    chessboard_x=chessboard_x,
                                                                                    chessboard_y=chessboard_y,
                                                                                    chessboard_dim=chessboard_dim,
                                                                                    debug=debug, max_imgs=max_imgs)

    print("Calibrating camera")
    print(chessboard_dim)
    retval, objectPoints, imagePoints1, imagePoints2, K_l, xi_l, D_l, K_r, xi_r, D_r, rvec, tvec, rvecs_L, tvecs_L, idx = cv2.omnidir.stereoCalibrate(
        obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r, None, None, None, None, None, None, 0, CRITERIA)
    print("retval: ", retval)

    # new_K_l = np.copy(K_l)
    # new_K_l[0, 1] = 0.0
    # new_K_r = np.copy(K_r)
    # new_K_r[0, 1] = 0.0
    # new_K_r, _ = cv2.getOptimalNewCameraMatrix(K_r, D_r,img_dim_r, alpha=0)
    #
    # scale = 1.5
    # new_K_l_wide = np.copy(K_l)
    # new_K_l_wide[0, 1] = 0.0
    # new_K_l_wide[0, 0] = new_K_l_wide[0, 0] / scale
    # new_K_l_wide[1, 1] = new_K_l_wide[1, 1] / scale
    # new_K_r_wide = np.copy(K_r)
    # new_K_r_wide[0, 1] = 0.0
    # new_K_r_wide[0, 0] = new_K_r_wide[0, 0] / scale
    # new_K_r_wide[1, 1] = new_K_r_wide[1, 1] / scale
    #
    # I = np.eye(3)
    # zero_vector = np.zeros((3, 1))
    # Rt_l = np.hstack((I, zero_vector))
    # P_l = np.dot(K_l, Rt_l)
    # new_P_l = np.dot(new_K_l, Rt_l)
    #
    # R, _ = cv2.Rodrigues(rvec)
    # Rt_r = np.hstack((R, tvec))
    # P_r = np.dot(K_r, Rt_r)
    # new_P_r = np.dot(new_K_r, Rt_r)
    #
    # calib_dict = {'K_l': K_l, 'new_K_l': new_K_l, 'xi_l': xi_l, 'D_l': D_l, 'K_r': K_r, 'new_K_r': new_K_r,
    #               'xi_r': xi_r, 'D_r': D_r,
    #               'rvec': rvec, 'tvec': tvec, 'rvecs_L': rvecs_L, 'tvecs_L': tvecs_L, 'img_dim_l': img_dim_l,
    #               'img_dim_r': img_dim_r,
    #               'new_K_l_wide': new_K_l_wide, 'new_K_r_wide': new_K_r_wide, 'P_l': P_l, 'P_r': P_r,
    #               'new_P_l': new_P_l, 'new_P_r': new_P_r}
    scale = 2.5
    new_K_l = np.copy(K_l)
    new_K_l[0, 1] = 0.0
    # new_K_l[0, 0] = new_K_l[0, 0] / scale
    # new_K_l[1, 1] = new_K_l[1, 1] / scale
    new_K_l_wide = np.copy(K_l)
    new_K_l_wide[0, 1] = 0.0
    new_K_l_wide[0, 0] = new_K_l_wide[0, 0] / scale
    new_K_l_wide[1, 1] = new_K_l_wide[1, 1] / scale

    new_K_r = np.copy(K_r)
    new_K_r[0, 1] = 0.0
    # new_K_r[0, 0] = new_K_r[0, 0] / scale
    # new_K_r[1, 1] = new_K_r[1, 1] /scale
    new_K_r_wide = np.copy(K_r)
    new_K_r_wide[0, 1] = 0.0
    new_K_r_wide[0, 0] = new_K_r_wide[0, 0] / scale
    new_K_r_wide[1, 1] = new_K_r_wide[1, 1] / scale


    calib_dict = {'K_l': K_l, 'new_K_l': new_K_l, 'xi_l': xi_l, 'D_l': D_l, 'K_r': K_r, 'new_K_r': new_K_r,
                  'xi_r': xi_r,
                  'D_r': D_r, 'rvec': rvec, 'tvec': tvec, 'rvecs_L': rvecs_L, 'tvecs_L': tvecs_L,
                  'img_dim_l': img_dim_l, 'img_dim_r': img_dim_r, 'new_K_l_wide': new_K_l_wide,
                  'new_K_r_wide': new_K_r_wide}

    print("LEFT CAMERA")
    print("K_l:\n", K_l)
    print("xi_l:", xi_l)
    print("D_l:", D_l)

    print("\nRIGHT CAMERA")
    print("K_r:\n", K_r)
    print("xi_r:", xi_r)
    print("D_r:", D_r)

    return calib_dict


def getCorners(fname_l, undistort_l, chessboard_x=7, chessboard_y=4):
    img_l = cv2.imread(fname_l)
    img_pts_l = []
    if undistort_l is not None:
        img_l = undistort_l(img_l)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (chessboard_x, chessboard_y), None)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
    img_pts_l.append(corners_l)
    resized_frame_L = cv2.resize(img_l, (640, 480))

    img_concat_h = cv2.hconcat([resized_frame_L])
    cv2.imshow('img r', img_concat_h)
    cv2.waitKey(0)

def calibrate_on_undistored(
    img_folder,
    calib_dict,
    out_dir,
    camera_side="left",
    chessboard_x=7,
    chessboard_y=4,
    chessboard_dim=31.0,
    max_imgs=20,
    debug=0,
):
    if camera_side not in ("left", "right"):
        raise ValueError("camera_side must be 'left' or 'right'")

    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=False)
    images_l, images_r = get_l_r_image_fnames(img_folder, max_imgs)

    if camera_side == "left":
        image_list = images_l
        undistort_fn = undistort_l
    else:
        image_list = images_r
        undistort_fn = undistort_r

    obj_coords = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    obj_coords[:, :2] = chessboard_dim * np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    obj_pts = []
    img_pts = []
    image_size = None

    for fname in tqdm(image_list, total=len(image_list)):
        img = cv2.imread(fname)
        if img is None:
            if debug > 0:
                print("Could not read:", fname)
            continue

        img = undistort_fn(img)
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
                cv2.imshow(f"Calibration {camera_side}", vis)

                if debug >= 2:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(1)

        elif debug > 0:
            print("Corners not found!", fname)

    cv2.destroyAllWindows()

    if len(obj_pts) < 5:
        raise RuntimeError(f"Not enough valid images for calibration. Found only {len(obj_pts)}")

    print(f"Calibrating {camera_side} camera on {len(obj_pts)} undistorted images")

    retval, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts,
        img_pts,
        image_size,
        None,
        None,
    )

    print("=== Calibration finished ===")
    print("Camera side:", camera_side)
    print("Reprojection error:", retval)
    print("K:\n", K)
    print("D:\n", D)

    rectified_calib_dict = {
        "model": "pinhole_rectified",
        "camera_side": camera_side,
        "K": K,
        "D": D,
        "image_size": image_size,
        "reprojection_error": retval,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "num_images_used": len(obj_pts),
        "source_img_folder": str(img_folder),
        "chessboard_x": chessboard_x,
        "chessboard_y": chessboard_y,
        "chessboard_dim": chessboard_dim,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{camera_side}_NICO.yaml"
    out_path = out_dir / file_name

    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open file for writing: {out_path}")

    fs.write("model", rectified_calib_dict["model"])
    fs.write("camera_side", rectified_calib_dict["camera_side"])
    fs.write("K", rectified_calib_dict["K"])
    fs.write("D", rectified_calib_dict["D"])
    fs.write("image_width", rectified_calib_dict["image_size"][0])
    fs.write("image_height", rectified_calib_dict["image_size"][1])
    fs.write("reprojection_error", rectified_calib_dict["reprojection_error"])
    fs.write("num_images_used", rectified_calib_dict["num_images_used"])
    fs.write("chessboard_x", rectified_calib_dict["chessboard_x"])
    fs.write("chessboard_y", rectified_calib_dict["chessboard_y"])
    fs.write("chessboard_dim", rectified_calib_dict["chessboard_dim"])
    fs.release()

    print("Saved calibration to:", out_path)

    return rectified_calib_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib_imgs_dir', type=str, help='directory with calibration images')
    parser.add_argument('out_dir', type=str, help='directory where the calibration pickle gets saved to')
    parser.add_argument('-d', '--debug', type=int, default=0,
                        help='whether to debug 1 shows calib images, 2 lets you see the undistorted calib files')
    parser.add_argument('-eyeTracker_dir', '--debug', type=str, default="",
                        help='directory with eyetracking images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parent.parent
    dataset_dir = parent_dir / "dataset_11032026"
    calib_imgs_dir = dataset_dir / "stereo_4k_calibration" / "rgb"
    relative_pose_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    out_dir = parent_dir / "out" / "cameras_parameters"
    debug = 2


    # calib_dict = calibrate(calib_imgs_dir, debug=debug, chessboard_dim=30.0, max_imgs=40, chessboard_x=8, chessboard_y=5)
    # save_dict(calib_dict, out_dir)
    calib_dict = load_dict(out_dir / "calib_data.npy")
    print(calib_dict["new_K_l"])
    calibrate_on_undistored(relative_pose_dir, calib_dict, str(out_dir), camera_side="left", chessboard_dim=54.0, max_imgs=40, chessboard_x=6, chessboard_y=4, debug=1)
    calibrate_on_undistored(relative_pose_dir, calib_dict, str(out_dir), camera_side="right", chessboard_dim=54.0, max_imgs=40, chessboard_x=6, chessboard_y=4, debug=1)

    # show_undistorted_images(calib_dict, calib_imgs_dir)
