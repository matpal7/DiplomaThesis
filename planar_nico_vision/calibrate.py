import argparse
import os
import pickle

import numpy as np
import cv2
import glob

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def extract_chessboard_points(glob_string, chessboard_x = 9, chessboard_y = 6, chessboard_dim= 20.0, debug=0):
    obj_coords = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    obj_coords[:, :2] = chessboard_dim * np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    obj_pts = []
    img_pts = []
    images = glob.glob(glob_string)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            obj_pts.append(obj_coords)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            img_pts.append(corners)

            if debug > 0:
                cv2.drawChessboardCorners(img, (chessboard_x, chessboard_y), corners2, ret)
                cv2.imshow('img', cv2.resize(img, (1052, 780)))
                cv2.waitKey(1)
    return img_pts, obj_pts, img.shape


def save_calib_dict(calib_dict, out_folder):
    pkl_path = os.path.join(calib_dict, 'calib_data.pkl')
    with open(pkl_path, 'w') as f:
        pickle.dump(calib_dict, f)

    print("Wrote calib data to: ", pkl_path)


def calibrate(calib_img_folder, out_folder, chessboard_x = 9, chessboard_y = 6, chessboard_dim= 20.0, debug=0):
    calib_dict = {}

    for eye in ['l', 'r']:
        glob_string = '{}/*{}*.png'.format(calib_img_folder, eye)
        img_pts, obj_pts, img_shape = extract_chessboard_points(glob_string, chessboard_x, chessboard_y, chessboard_dim, debug)

        img_dim = (img_shape[1], img_shape[0])

        obj_pts = np.expand_dims(np.array(obj_pts, dtype=np.float64), -3)
        img_pts = np.array(img_pts, dtype=np.float64)

        retval, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(obj_pts, img_pts, img_dim, None, None, None, 0, CRITERIA)
        map1, map2 = cv2.omnidir.initUndistortRectifyMap(K, D, xi, np.eye(3), K, img_dim, cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE)

        # K_wide = np.array([[img_dim[0]/6, 0, img_dim[0]/2], [0, img_dim[1]/6, img_dim[1]/2], [0,0,1]])
        # map1_wide, map2_wide = cv2.omnidir.initUndistortRectifyMap(K, D, xi, np.eye(3), K_wide, img_dim, cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE)

        eye_dict = {'K': K, 'xi': xi, 'D': D, 'map1': map1, 'map2': map2, 'img_dim': img_dim}

        if debug > 0:
            print(eye_dict)

        calib_dict[eye] = eye_dict

        if debug > 1:
            images = glob.glob(glob_string)
            for fname in images:
                img = cv2.imread(fname)
                undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cv2.imshow('calibresult', cv2.resize(undistorted_img, (1052, 780)))
                # undistorted_img_wide = cv2.remap(img, map1_wide, map2_wide, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                # cv2.imshow('calib result wide', cv2.resize(undistorted_img_wide, (1052, 780)))
                cv2.waitKey(0)

    save_calib_dict(calib_dict, out_folder)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib_imgs_dir', type=str, help='directory with the calibration images')
    parser.add_argument('out_dir', type=str, help='directory where the calibration pickle gets saved to')
    parser.add_argument('-d', '--debug', type=int, default=0, help='whether to debug 1 shows calib images, 2 lets you see the undistorted calib files')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # calib_imgs_dir = 'D:/Research/data/NICO/calib_new_lenses'
    # out_dir = 'D:/Research/data/NICO/'
    calibrate(args.calib_imgs_dir, args.out_dir, debug=2)