import argparse
import os

import numpy as np
import cv2
from tqdm import tqdm

from utils import get_l_r_image_fnames, load_calib_data, get_undistort_functions, save_array

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
frameSize = (640, 480)
width = 1280
height = 720

def extract_chessboard_points(calib_img_folder, undistort_l=None, undistort_r=None, chessboard_x=7, chessboard_y=4,
                              chessboard_dim=31.0, debug=0, max_imgs=None):
    obj_coords = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    obj_coords[:, :2] = chessboard_dim * np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    obj_pts = []
    img_pts_l = []
    img_pts_r = []

    images_l, images_r = get_l_r_image_fnames(calib_img_folder, max_imgs, datasetFolder=False)
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
            print("Corners not found!")

    obj_pts = np.expand_dims(np.array(obj_pts, dtype=np.float64), -3)
    img_pts_l = np.array(img_pts_l, dtype=np.float64)
    img_pts_r = np.array(img_pts_r, dtype=np.float64)

    img_dim_l = (img_l.shape[1], img_l.shape[0])
    img_dim_r = (img_r.shape[1], img_r.shape[0])

    return obj_pts, img_pts_l, img_pts_r, img_dim_l, img_dim_r


def save_calib_dict(calib_dict, out_folder):
    npy_path = os.path.join(out_folder, 'calib_data.npy')
    np.save(npy_path, calib_dict)


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

    new_K_l = np.copy(K_l)
    new_K_l[0, 1] = 0.0
    new_K_r = np.copy(K_r)
    new_K_r[0, 1] = 0.0
    new_K_r, _ = cv2.getOptimalNewCameraMatrix(K_r, D_r,img_dim_r, alpha=0)

    scale = 1.5
    new_K_l_wide = np.copy(K_l)
    new_K_l_wide[0, 1] = 0.0
    new_K_l_wide[0, 0] = new_K_l_wide[0, 0] / scale
    new_K_l_wide[1, 1] = new_K_l_wide[1, 1] / scale
    new_K_r_wide = np.copy(K_r)
    new_K_r_wide[0, 1] = 0.0
    new_K_r_wide[0, 0] = new_K_r_wide[0, 0] / scale
    new_K_r_wide[1, 1] = new_K_r_wide[1, 1] / scale

    I = np.eye(3)
    zero_vector = np.zeros((3, 1))
    Rt_l = np.hstack((I, zero_vector))
    P_l = np.dot(K_l, Rt_l)
    new_P_l = np.dot(new_K_l, Rt_l)

    R, _ = cv2.Rodrigues(rvec)
    Rt_r = np.hstack((R, tvec))
    P_r = np.dot(K_r, Rt_r)
    new_P_r = np.dot(new_K_r, Rt_r)

    calib_dict = {'K_l': K_l, 'new_K_l': new_K_l, 'xi_l': xi_l, 'D_l': D_l, 'K_r': K_r, 'new_K_r': new_K_r,
                  'xi_r': xi_r, 'D_r': D_r,
                  'rvec': rvec, 'tvec': tvec, 'rvecs_L': rvecs_L, 'tvecs_L': tvecs_L, 'img_dim_l': img_dim_l,
                  'img_dim_r': img_dim_r,
                  'new_K_l_wide': new_K_l_wide, 'new_K_r_wide': new_K_r_wide, 'P_l': P_l, 'P_r': P_r,
                  'new_P_l': new_P_l, 'new_P_r': new_P_r}

    return calib_dict


def rectify_images(img_l, img_r, calib_dir, get_wide=False):
    img_size = (img_l.shape[1], img_l.shape[0])

    flag = cv2.omnidir.RECTIFY_PERSPECTIVE
    num_disparities = 16 * 8
    sad_window_size = 5 * 2
    point_type = cv2.omnidir.XYZRGB

    KNew = calib_dir['new_K_l']
    if get_wide:
        KNew = calib_dir['new_K_l_wide']

    disparity, image1Rec, image2Rec, pointCloud = cv2.omnidir.stereoReconstruct(img_l, img_r, calib_dir['K_l'],
                                                                                calib_dir['D_l'], calib_dir['xi_l'],
                                                                                calib_dir['K_r'], calib_dir['D_r'],
                                                                                calib_dir['xi_r'],
                                                                                calib_dir['rvec'], calib_dir['tvec'],
                                                                                flag=flag,
                                                                                numDisparities=num_disparities,
                                                                                SADWindowSize=sad_window_size,
                                                                                newSize=img_size, Knew=KNew,
                                                                                pointType=point_type)

    return image1Rec, image2Rec, disparity, pointCloud


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


def show_undistored(img_folder, calib_dict):
    images_l, images_r = get_l_r_image_fnames(img_folder, 10, datasetFolder=False)
    undistort_l, undistort_r = get_undistort_functions(calib_dict, get_wide=False)

    for fname_l, fname_r in tqdm(zip(images_l, images_r), total=len(images_l)):
        # rectify_images(fname_l, fname_r, calib_dict)
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)
        img_l_origin = cv2.resize(img_l.copy(), frameSize)
        img_r_origin = cv2.resize(img_r.copy(), frameSize)
        img_l = undistort_l(img_l)
        img_r = undistort_r(img_r)

        resized_frame_L = cv2.resize(img_l, frameSize)
        resized_frame_R = cv2.resize(img_r, frameSize)

        img_concat_h = cv2.hconcat([img_l_origin, resized_frame_L, resized_frame_R])
        cv2.imshow('unidistored', img_concat_h)
        cv2.waitKey(0)


def show_real_time(calib_dict, dirc):  

    cap_L = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap_R = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    num = 4
    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=False)

    while cap_L.isOpened():
        succes, image_L = cap_L.read()
        succes, image_R = cap_R.read()
        img_l = image_L.copy()
        img_r = image_R.copy()

        img_l = undistort_l(img_l)
        img_r = undistort_r(img_r)
        resized_frame_L = cv2.resize(img_l, (640, 480))
        resized_frame_R = cv2.resize(img_r, (640, 480))

        img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('unidistored', img_concat_h)
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(dirc + "left/" + str(num) + "_l.png", image_L)
            cv2.imwrite(dirc + "right/" + str(num) + "_r.png", image_R)
            cv2.imwrite(dirc + "" + str(num) + "_l.png", image_L)
            cv2.imwrite(dirc + "" + str(num) + "_r.png", image_R)
            print("{} saved".format(num))
            num += 1

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
    # par_dir = "C:/Users/Matej/Desktop/bakalarkaGit/bakalarka/BachelorThesis/nico_images/dataset_03"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(parent_dir, 'dataset')
    calib_imgs_dir = os.path.join(dataset_dir, "calibration")
    out_dir = os.path.join(parent_dir, 'out')
    # eyeTracker_dir = par_dir + '/eyetracker'
    debug = 2

    # args = parse_args()nk
    # calib_imgs_dir = args.calib_imgs_dir
    # out_dir = args.out_dir
    # debug = args.debug
    # eyeTracker_dir = args.eyeTracker_dir
    print(calib_imgs_dir)
    # calib_dict = calibrate(calib_imgs_dir, debug=debug, chessboard_dim=30.0, max_imgs=25, chessboard_x=8, chessboard_y=6)
    # save_array(calib_dict, os.path.join(out_dir, 'calib_data.npy'))
    calib_dir = load_calib_data(out_dir + "/calib_data.npy")
    # calib_dict = load_calib_data(out_dir + "/calib_data.npy")  # load_calib_data(out_dir + "/calib_data.npy")
    # show_undistored(calib_imgs_dir, calib_dict)
    depth_dir = os.path.join(dataset_dir, 'depth')
    show_undistored(depth_dir + "/rgb", calib_dir)
    #
    # show_real_time(calib_dict, eyeTracker_dir)
