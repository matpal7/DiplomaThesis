import argparse
import os

import numpy as np
import cv2
from tqdm import tqdm

from image import get_undistort_functions
from utils import get_l_r_image_fnames

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def extract_chessboard_points(calib_img_folder, undistort_l=None, undistort_r=None, chessboard_size=(8,6), chessboard_dim=20.0, debug=0, max_imgs=None):
    chessboard_x, chessboard_y = chessboard_size
    obj_coords = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    obj_coords[:, :2] = chessboard_dim * np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    obj_pts = []
    img_pts_l = []
    img_pts_r = []

    images_l, images_r = get_l_r_image_fnames(calib_img_folder, max_imgs)
    print("IMAGES LEFT: ", images_l)

    print("Finding calibration patterns")
    for fname_l, fname_r in tqdm(zip(images_l, images_r), total=len(images_l)):
        img_l = cv2.imread(fname_l)
        if undistort_l is not None:
            img_l = undistort_l(img_l)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)

        img_r = cv2.imread(fname_r)
        if undistort_r is not None:
            img_r = undistort_r(img_r)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

        if ret_r and ret_l:
            obj_pts.append(obj_coords)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)
            img_pts_l.append(corners_l)
            img_pts_r.append(corners_r)

            if debug > 0:
                cv2.drawChessboardCorners(img_l, (chessboard_x, chessboard_y), corners_l, ret_l)
                cv2.drawChessboardCorners(img_r, (chessboard_x, chessboard_y), corners_r, ret_r)
                cv2.imshow('img l', cv2.resize(img_l, (1052, 780)))
                cv2.imshow('img r', cv2.resize(img_r, (1052, 780)))
                cv2.waitKey(1)
        elif debug > 0:
            print("Corners not found!")
            cv2.imshow('img l', cv2.resize(img_l, (1052, 780)))
            cv2.imshow('img r', cv2.resize(img_r, (1052, 780)))
            cv2.waitKey(0)

    obj_pts = np.expand_dims(np.array(obj_pts, dtype=np.float64), -3)
    img_pts_l = np.array(img_pts_l, dtype=np.float64)
    img_pts_r = np.array(img_pts_r, dtype=np.float64)

    img_dim_l = (img_l.shape[1], img_l.shape[0])
    img_dim_r = (img_r.shape[1], img_r.shape[0])


    return obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r


def save_calib_dict(calib_dict, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    npy_path = os.path.join(out_folder, 'calib_data.npy')
    np.save(npy_path, calib_dict)
    print("Wrote calib data to: ", npy_path)


def calibrate(calib_img_folder, chessboard_size=(8,6), chessboard_dim= 20.0, debug=0, max_imgs=None):
    obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r = extract_chessboard_points(calib_img_folder, chessboard_size=chessboard_size, chessboard_dim=chessboard_dim, debug=debug, max_imgs=max_imgs)

    print("Calibrating camera")
    retval, objectPoints, imagePoints1, imagePoints2, K_l, xi_l, D_l, K_r, xi_r, D_r, rvec, tvec, rvecs_L, tvecs_L, idx = cv2.omnidir.stereoCalibrate(obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r, None, None, None, None, None, None, 0, CRITERIA)

    new_K_l = np.copy(K_l)
    new_K_l[0, 1] = 0.0
    # new_K_l[0, 0] = new_K_l[0, 0] / 2.5
    # new_K_l[1, 1] = new_K_l[1, 1] / 2.5
    new_K_r = np.copy(K_r)
    new_K_r[0, 1] = 0.0
    # new_K_r[0, 0] = new_K_r[0, 0] / 2.5
    # new_K_r[1, 1] = new_K_r[1, 1] / 2.5

    calib_dict = {'K_l': K_l, 'new_K_l': new_K_l, 'xi_l': xi_l, 'D_l': D_l, 'K_r': K_r, 'new_K_r': new_K_r, 'xi_r': xi_r,
                  'D_r': D_r, 'rvec': rvec, 'tvec': tvec, 'rvecs_L': rvecs_L, 'tvecs_L': tvecs_L,
                  'img_dim_l': img_dim_l, 'img_dim_r': img_dim_r}


    return calib_dict


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


def get_intersects_from_four_points(p1, p2, p3, p4):
    line_12 = line(p1, p2)
    line_34 = line(p3, p4)

    line_13 = line(p1, p3)
    line_24 = line(p2, p4)

    intersection_1 = intersection(line_12, line_34)
    intersection_2 = intersection(line_13, line_24)

    return intersection_1, intersection_2


def get_horizon(img_pts, obj_pts, chessboard_x = 9, chessboard_y = 6, debug=0):
    img_pts = np.reshape(img_pts, [-1, chessboard_y, chessboard_x, 2])
    objt_pts = np.reshape(obj_pts, [-1, chessboard_y, chessboard_x, 3])

    vp_1 = []
    vp_2 = []

    for i in range(img_pts.shape[0]):
        for start_x in range(chessboard_x):
            for start_y in range(chessboard_y):
                for end_x in range(start_x + 1, chessboard_x):
                    for end_y in range(start_y + 1, chessboard_y):
                        intersection_1, intersection_2 = get_intersects_from_four_points(img_pts[i, start_y, start_x, :],
                                                                                         img_pts[i, end_y, start_x, :],
                                                                                         img_pts[i, start_y, end_x, :],
                                                                                         img_pts[i, end_y, end_x, :])

                        if not intersection_1 or not intersection_2:
                            continue
                        vp_1.append(intersection_1)
                        vp_2.append(intersection_2)

    vp_1 = np.array(vp_1)
    vp_2 = np.array(vp_2)

    m = (vp_1[:, 1] - vp_2[:, 1]) / (vp_1[:, 0] - vp_2[:, 0])
    b1 = vp_1[:, 1] - m * vp_1[:, 0]
    b2 = vp_2[:, 1] - m * vp_2[:, 0]

    med_m = np.nanmedian(m)
    med_k = np.nanmedian(np.concatenate([b1, b2]))

    if debug > 0:
        print("m median:", med_m)
        print("k median:", med_k)
        print("m std:", np.std(m))
        print("k std:", np.std(np.concatenate([b1, b2])))

    h = np.array([med_m, -1, med_k])
    return h


def calibrate_horizon(horizon_img_dir, calib_dict, chessboard_x = 9, chessboard_y = 6, chessboard_dim= 20.0, debug=0, correct_horizon=False):
    print("Starting horizon detection")
    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=correct_horizon)
    obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r = extract_chessboard_points(horizon_img_dir, undistort_l=undistort_l, undistort_r=undistort_r, chessboard_x=chessboard_x, chessboard_y=chessboard_y, chessboard_dim=chessboard_dim, debug=debug, max_imgs=None)

    print("Calculating horizons")
    horizon_r = get_horizon(img_pts_l, obj_pts, chessboard_x, chessboard_y, debug=debug)
    horizon_l = get_horizon(img_pts_r, obj_pts, chessboard_x, chessboard_y, debug=debug)

    print("Horizon R:", horizon_r)
    print("Horizon L:", horizon_l)

    calib_dict['horizon_r'] = horizon_r
    calib_dict['horizon_l'] = horizon_l

    if debug > 0:
        n_l = calib_dict['new_K_l'].T @ horizon_l
        print("n_l", n_l/np.linalg.norm(n_l))

        n_r = calib_dict['new_K_r'].T @ horizon_r
        print("n_r", n_r/np.linalg.norm(n_r))

    return calib_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib_imgs_dir', type=str, help='directory with the calibration images')
    parser.add_argument('horizon_imgs_dir', type=str, help='directory with the calibration images')
    parser.add_argument('out_dir', type=str, help='directory where the calibration pickle gets saved to')
    parser.add_argument('-d', '--debug', type=int, default=0, help='whether to debug 1 shows calib images, 2 lets you see the undistorted calib files')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    #args = parse_args()
    calib_imgs_dir = 'C:/Users/Lenovo/Desktop/DiplomaThesis/dataset/calibration/'
    out_dir = 'C:/Users/Lenovo/Desktop/DiplomaThesis_git/NICO/out/'
    debug = 1

    calib_dict = calibrate(calib_imgs_dir, debug=debug)
    # calib_dict = calibrate_horizon(args.horizon_imgs_dir, calib_dict, debug=1)
    save_calib_dict(calib_dict, out_dir)

    # calib_dict = calibrate_horizon(args.horizon_imgs_dir, calib_dict, correct_horizon=True, debug=1)