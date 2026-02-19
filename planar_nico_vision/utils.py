import glob
import os

import numpy as np
import cv2
import re


def load_calib_data(npy_path):
    calib_dict = np.load(npy_path, allow_pickle=True).flat[0]
    return calib_dict


def get_l_r_image_fnames(calib_img_folder, max_imgs=None):
    glob_string_l = '{}/*_left.png'.format(calib_img_folder)
    glob_string_r = '{}/*_right.png'.format(calib_img_folder)
    images_l = sorted(glob.glob(glob_string_l))
    images_r = sorted(glob.glob(glob_string_r))

    if max_imgs is not None:
        images_l = images_l[:max_imgs]
        images_r = images_r[:max_imgs]

    return images_l, images_r

# def get_l_r_image_fnames(calib_img_folder, max_imgs=None):
#     # Get all PNG files in folder
#     all_files = glob.glob(os.path.join(calib_img_folder, '*.png'))
#
#     # Filter only *_left.png and *_right.png, ignore *_realsense.png
#     images_l = sorted([
#         f for f in all_files
#         if re.match(r'\d+_left\.png$', os.path.basename(f))
#     ])
#     images_r = sorted([
#         f for f in all_files
#         if re.match(r'\d+_right\.png$', os.path.basename(f))
#     ])
#
#     # Limit number of images if requested
#     if max_imgs is not None:
#         images_l = images_l[:max_imgs]
#         images_r = images_r[:max_imgs]
#
#     return images_l, images_r