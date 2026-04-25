import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from code.image import get_undistort_functions
from code.calibration.undistored_images import show_undistorted_images
from code.utils import get_l_r_image_fnames, load_dict, save_dict

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
                cv2.waitKey(0)

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
    parent_dir = Path(__file__).resolve().parents[3]
    date = "24042026"
    dataset_dir = parent_dir /"datasets" / f"dataset_{date}"
    calib_imgs_dir = dataset_dir / "stereo_4k_calibration" / "rgb"
    relative_pose_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    out_dir = parent_dir /"out" / f"out_{date}" / "cameras_parameters"
    debug = 0


    calib_dict = calibrate(calib_imgs_dir, debug=debug, chessboard_dim=30.0, max_imgs=None, chessboard_x=8, chessboard_y=5)
    baseline_m = np.linalg.norm(calib_dict["tvec"].reshape(-1)) / 1000.0
    print("BASELINE in meters:", baseline_m)
    # print(out_dir)
    save_dict(calib_dict, out_dir)
    # calib_dict = load_dict(out_dir / "calib_data.npy")
    # #
    # show_undistorted_images(calib_dict, calib_imgs_dir, max_imgs=5)
