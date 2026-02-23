import os
import numpy as np
import cv2

from planar_nico_vision.utils import load_calib_data


def get_rectify_maps(calib_dict, image_size, balance=0.0):
    K_l  = calib_dict['K_l']
    D_l  = calib_dict['D_l']
    xi_l = calib_dict['xi_l']

    K_r  = calib_dict['K_r']
    D_r  = calib_dict['D_r']
    xi_r = calib_dict['xi_r']

    rvec = calib_dict['rvec']
    tvec = calib_dict['tvec']

    R, _ = cv2.Rodrigues(rvec)
    T = tvec

    R1, R2 = cv2.omnidir.stereoRectify(
        R, T,
        flags=cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    map1_1, map1_2 = cv2.omnidir.initUndistortRectifyMap(
        K_l, D_l, xi_l,
        R1,
        K_l,
        image_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    map2_1, map2_2 = cv2.omnidir.initUndistortRectifyMap(
        K_r, D_r, xi_r,
        R2,
        K_r,
        image_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    rectification = {
        "map_l_x": map_l_x,
        "map_l_y": map_l_y,
        "map_r_x": map_r_x,
        "map_r_y": map_r_y
    }

    return map1_1, map1_2, map2_1, map2_2


def stereo_rectify_from_calib(calib_dict, use_new_K=False):

    # --- Parametre z calib_dict
    K_l = calib_dict['new_K_l'] if use_new_K else calib_dict['K_l']
    D_l = calib_dict['D_l']
    xi_l = np.array(calib_dict['xi_l']).reshape(1)

    K_r = calib_dict['new_K_r'] if use_new_K else calib_dict['K_r']
    D_r = calib_dict['D_r']
    xi_r = np.array(calib_dict['xi_r']).reshape(1)

    rvec = calib_dict['rvec']
    tvec = calib_dict['tvec']

    img_size = tuple(calib_dict['img_dim_l'])

    # --- rvec -> R
    R, _ = cv2.Rodrigues(rvec)
    T = tvec

    # --- Stereo rectify
    R1, R2 = cv2.omnidir.stereoRectify(
        R, T,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    # --- Ľavá kamera mapy
    map_l_x, map_l_y = cv2.omnidir.initUndistortRectifyMap(
        K_l, D_l, xi_l,
        R1,
        K_l,
        img_size,
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    # --- Pravá kamera mapy
    map_r_x, map_r_y = cv2.omnidir.initUndistortRectifyMap(
        K_r, D_r, xi_r,
        R2,
        K_r,
        img_size,
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    return {
        "map_l_x": map_l_x,
        "map_l_y": map_l_y,
        "map_r_x": map_r_x,
        "map_r_y": map_r_y
    }

def rectify_pair(img_l, img_r, rect_maps):
    rect_l = cv2.remap(img_l,
                       rect_maps["map_l_x"],
                       rect_maps["map_l_y"],
                       interpolation=cv2.INTER_LINEAR)

    rect_r = cv2.remap(img_r,
                       rect_maps["map_r_x"],
                       rect_maps["map_r_y"],
                       interpolation=cv2.INTER_LINEAR)

    return rect_l, rect_r


if __name__ == "__main__":
    out_dir = 'C:/Users/Lenovo/Desktop/DiplomaThesis_git/NICO/out/'
    calib_dict = load_calib_data(out_dir + "calib_data.npy")
    rect_maps = stereo_rectify_from_calib(calib_dict)

    img_l = cv2.imread("C:\\Users\\Lenovo\\Desktop\\DiplomaThesis\\dataset\depth\\rgb\\0_left.png")
    img_r = cv2.imread("C:\\Users\\Lenovo\\Desktop\\DiplomaThesis\\dataset\depth\\rgb\\0_right.png")

    image_size = (img_l.shape[1], img_l.shape[0])

    scale = 1/4
    unrect_l = cv2.resize(img_l, (0, 0), fx=scale, fy=scale)
    unrect_r = cv2.resize(img_r, (0, 0), fx=scale, fy=scale)
    both = np.hstack((unrect_l, unrect_r))

    # draw epipolar lines
    for y in range(0, both.shape[0], 40):
        cv2.line(both, (0, y), (both.shape[1], y), (0,255,0), 1)


    cv2.imshow("UNrectified Pair", both)

    rect_l = cv2.remap(img_l,
                       rect_maps["map_l_x"],
                       rect_maps["map_l_y"],
                       cv2.INTER_LINEAR)

    rect_r = cv2.remap(img_r,
                       rect_maps["map_r_x"],
                       rect_maps["map_r_y"],
                       cv2.INTER_LINEAR)

    rect_l_small = cv2.resize(rect_l, (0, 0), fx=scale, fy=scale)
    rect_r_small = cv2.resize(rect_r, (0, 0), fx=scale, fy=scale)
    both_rect = np.hstack((rect_l_small, rect_r_small))

    # draw epipolar lines
    for y in range(0, both.shape[0], 40):
        cv2.line(both, (0, y), (both_rect.shape[1], y), (0, 255, 0), 1)

    cv2.imshow("Rectified Pair", both)
    cv2.waitKey(0)
