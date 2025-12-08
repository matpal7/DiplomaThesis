import glob
import os

import cv2
import numpy as np


def load_calib_data(npy_path):
    calib_dict = np.load(npy_path, allow_pickle=True).flat[0]
    return calib_dict


def save_array(array, path):
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(path, array)


def load_data(npy_path):
    array = np.load(npy_path, allow_pickle=True)
    return array


def get_l_r_image_fnames(img_folder, max_imgs=None, datasetFolder=False, withGlasses=True, name='*', index=1):
    if name == '': name = '*'
    if not datasetFolder:
        glob_string_l = '{}/*left*.png'.format(img_folder)
        glob_string_r = '{}/*right*.png'.format(img_folder)
    else:
        glasses = 'with_glasses' if withGlasses else 'no_glasses'
        glob_string_l = '{}/{}/{}/{}/*l*.png'.format(img_folder, name, glasses, index)
        glob_string_r = '{}/{}/{}/{}/*r*.png'.format(img_folder, name, glasses, index)

    images_l = sorted(glob.glob(glob_string_l))
    images_r = sorted(glob.glob(glob_string_r))

    if max_imgs is not None:
        images_l = images_l[:max_imgs]
        images_r = images_r[:max_imgs]

    return images_l, images_r


def get_undistort_functions(calib_dict, get_wide=False):
    R_l = np.eye(3)
    R_r = np.eye(3)
    if get_wide:
        new_K_l = calib_dict['new_K_l_wide']
        new_K_r = calib_dict['new_K_r_wide']
    else:
        new_K_l = calib_dict['new_K_l']
        new_K_r = calib_dict['new_K_r']

    map1_l, map2_l = cv2.omnidir.initUndistortRectifyMap(calib_dict['K_l'], calib_dict['D_l'], calib_dict['xi_l'],
                                                         R_l, new_K_l, calib_dict['img_dim_l'],
                                                         cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE)

    def undistort_l(img):
        return cv2.remap(img, map1_l, map2_l, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    map1_r, map2_r = cv2.omnidir.initUndistortRectifyMap(calib_dict['K_r'], calib_dict['D_r'], calib_dict['xi_r'],
                                                         R_r, new_K_r, calib_dict['img_dim_r'],
                                                         cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE)

    def undistort_r(img):
        return cv2.remap(img, map1_r, map2_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistort_l, undistort_r


def convert_mm_to_pixels(point_3D, P):
    point_3D = np.append(point_3D, 1)
    point_2D_homogeneous = np.dot(P, point_3D)

    x = point_2D_homogeneous[0] / point_2D_homogeneous[2]
    y = point_2D_homogeneous[1] / point_2D_homogeneous[2]
    return np.array([x, y]).astype(np.int32)


def crop_image(img, a, b, c, d):
    return img.copy()[a:b, c:d]


def crop_image(img, head_box):
    a, b, width, height = head_box
    cropped_image = img[b:b + height, a:a + width]
    return cropped_image


def show_image(img, frame_size=None):
    if frame_size != None:
        img = cv2.resize(img, frame_size)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def get_data(save_dir, prefix=''):
    if prefix != '':
        prefix += "_"

    folder_GazeTR = "/GazeTR"
    folder_L2CS = "/L2CS"

    path_with_glasses = f"/{prefix}with_glasses.npy"
    path_no_glasses = f"/{prefix}no_glasses.npy"

    list_errors_with_glasses_L2CS = load_data(save_dir + folder_L2CS + path_with_glasses)
    list_errors_no_glasses_L2CS = load_data(save_dir + folder_L2CS + path_no_glasses)

    list_errors_with_glasses_GazeTR = load_data(save_dir + folder_GazeTR + path_with_glasses)
    list_errors_no_glasses_GazeTR = load_data(save_dir + folder_GazeTR + path_no_glasses)

    errors_with_glasses_L2CS_angle = np.array([])
    errors_with_glasses_L2CS_distance = np.array([])

    errors_no_glasses_L2CS_angle = np.array([])
    errors_no_glasses_L2CS_distance = np.array([])

    errors_with_glasses_GazeTR_angle = np.array([])
    errors_no_glasses_GazeTR_angle = np.array([])

    errors_with_glasses_GazeTR_distance = np.array([])
    errors_no_glasses_GazeTR_distance = np.array([])

    # # Process L2CS errors
    for err in list_errors_with_glasses_L2CS:
        errors_with_glasses_L2CS_angle = np.append(errors_with_glasses_L2CS_angle, err.angle_error)
        errors_with_glasses_L2CS_distance = np.append(errors_with_glasses_L2CS_distance, err.distance_error)

    for err in list_errors_no_glasses_L2CS:
        errors_no_glasses_L2CS_angle = np.append(errors_no_glasses_L2CS_angle, err.angle_error)
        errors_no_glasses_L2CS_distance = np.append(errors_no_glasses_L2CS_distance, err.distance_error)

    # Process GazeTR errors
    for err in list_errors_with_glasses_GazeTR:
        errors_with_glasses_GazeTR_angle = np.append(errors_with_glasses_GazeTR_angle, err.angle_error)
        errors_with_glasses_GazeTR_distance = np.append(errors_with_glasses_GazeTR_distance, err.distance_error)

    for err in list_errors_no_glasses_GazeTR:
        errors_no_glasses_GazeTR_angle = np.append(errors_no_glasses_GazeTR_angle, err.angle_error)
        errors_no_glasses_GazeTR_distance = np.append(errors_no_glasses_GazeTR_distance, err.distance_error)

    errors_L2CS_angle = np.concatenate((errors_with_glasses_L2CS_angle, errors_no_glasses_L2CS_angle))
    errors_L2CS_distance = np.concatenate((errors_with_glasses_L2CS_distance, errors_no_glasses_L2CS_distance))
    errors_GazeTR_angle = np.concatenate((errors_with_glasses_GazeTR_angle, errors_no_glasses_GazeTR_angle))
    errors_GazeTR_distance = np.concatenate((errors_with_glasses_GazeTR_distance, errors_no_glasses_GazeTR_distance))

    data = {
        "with_glasses": {
            "L2CS": {
                "angle": errors_with_glasses_L2CS_angle,
                "distance": errors_with_glasses_L2CS_distance
            },
            "GazeTR": {
                "angle": errors_with_glasses_GazeTR_angle,
                "distance": errors_with_glasses_GazeTR_distance
            }
        },
        "no_glasses": {
            "L2CS": {
                "angle": errors_no_glasses_L2CS_angle,
                "distance": errors_no_glasses_L2CS_distance
            },
            "GazeTR": {
                "angle": errors_no_glasses_GazeTR_angle,
                "distance": errors_no_glasses_GazeTR_distance
            }
        },
        "combined": {
            "L2CS": {
                "angle": errors_L2CS_angle,
                "distance": errors_L2CS_distance
            },
            "GazeTR": {
                "angle": errors_GazeTR_angle,
                "distance": errors_GazeTR_distance
            }
        }
    }
    return data
