import os.path

import cv2
import numpy as np
import re

from utils import get_l_r_image_fnames, get_depth_rgb_image_fnames


class Image():
    def __init__(self, img_path, undistort_function):
        self.pose = None
        self.img_path = img_path
        self.img_basename = os.path.basename(img_path)

        img_distorted = cv2.imread(img_path)
        self.img = undistort_function(img_distorted)
        self.dims = (self.img.shape[1], self.img.shape[0])

    def get_small_img(self, scale=4):
        mini_dims = self.scaled_dims(scale)
        return cv2.resize(self.img, mini_dims)

    def scaled_dims(self, scale):
        mini_dims = (self.dims[0] // scale, self.dims[1] // scale)
        return mini_dims

    def get_image_number(self):
        basename = os.path.basename(self.img_path)
        match = re.match(r'(\d+)_', basename)
        if match:
            return int(match.group(1))
        else:
            return None

    def get_path(self):
        return self.img_path


    # def set_kp_and_des(self, kp, des):
    #     self.kp = kp
    #     self.p = np.array([p.pt for p in kp])
    #     self.des = des
    #     self.bgrs = np.array([self.img[int(p[1]), int(p[0]), :] for p in self.p])
    #     self.X_ptrs = -np.ones(len(self.p)).astype(np.int)

class ImageRealsense(Image):
    def __init__(self, img_path):
        self.pose = None
        self.img_path = img_path
        self.img_basename = os.path.basename(img_path)

        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image at path: {img_path}")

        self.dims = (self.img.shape[1], self.img.shape[0])


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions


def get_corrective_rotation(K, horizon):
    n = K.T @ horizon
    n /= np.linalg.norm(n)

    R = rotation_matrix_from_vectors(n, np.array([0, -1, 0]))
    return R


def get_undistort_functions(calib_dict, correct_horizon=False):
    if correct_horizon:
        R_l = get_corrective_rotation(calib_dict['new_K_l'], calib_dict['horizon_l'])
        R_r = get_corrective_rotation(calib_dict['new_K_r'], calib_dict['horizon_r'])
    else:
        R_l = np.eye(3)
        R_r = np.eye(3)

    map1_l, map2_l = cv2.omnidir.initUndistortRectifyMap(calib_dict['K_l'], calib_dict['D_l'], calib_dict['xi_l'],
                                                         R_l, calib_dict['new_K_l'], calib_dict['img_dim_l'],
                                                         cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE)

    def undistort_l(img):
        return cv2.remap(img, map1_l, map2_l, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    map1_r, map2_r = cv2.omnidir.initUndistortRectifyMap(calib_dict['K_r'], calib_dict['D_r'], calib_dict['xi_r'],
                                                         R_r, calib_dict['new_K_r'], calib_dict['img_dim_r'],
                                                         cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE)

    def undistort_r(img):
        return cv2.remap(img, map1_r, map2_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistort_l, undistort_r


def load_l_r_images_undistorted(calib_dict, img_dir, correct_horizon=False, max_imgs=None):
    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=correct_horizon)
    fnames_l, fnames_r = get_l_r_image_fnames(img_dir, max_imgs=max_imgs)

    imgs_l = [Image(fname, undistort_l) for fname in fnames_l]
    imgs_r = [Image(fname, undistort_r) for fname in fnames_r]

    return imgs_l, imgs_r

def load_realsense_rgb_images(img_dir, max_imgs= None):
    fname_rgb = get_depth_rgb_image_fnames(img_dir, max_imgs=max_imgs)
    imgs_rgb = [ImageRealsense(fname) for fname in fname_rgb]

    return imgs_rgb

# def extract_descriptors(imgs: List[Image], scale=4):
#     sift = cv2.SIFT_create()
#
#     # bg_sub = cv2.createBackgroundSubtractorMOG2(history=3, detectShadows=False)
#     # for img in imgs:
#     #     bg_sub.apply(img.get_small_img(scale=scale))
#
#     small_imgs = np.array([img.get_small_img(scale=scale) for img in imgs])
#     img_mean = np.mean(small_imgs, axis=0)
#     img_std_sqr = np.percentile(np.std(small_imgs, axis=0), 99) ** 2
#
#     for i in range(len(imgs)):
#         # # fg[:3 * fg.shape[0]//4, :] = 0
#         # fg = 255 * np.ones(imgs[i].dims[:2], dtype=np.uint8)
#
#         # fg = np.where(np.sum((small_imgs[i] - img_mean) ** 2, axis=-1) > img_std_sqr, 255, 0).astype(np.uint8)
#         # fg[:fg.shape[0]//2, :] = 0
#         #
#         # element_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         # element_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
#         #
#         # fg = cv2.erode(fg, element_erode)
#         # fg = cv2.dilate(fg, element_dilate)
#         #
#         # fg = cv2.resize(fg, imgs[i].dims)
#         # kp, des = sift.detectAndCompute(imgs[i].img, fg)
#
#         kp, des = sift.detectAndCompute(imgs[i].img, None)
#
#         imgs[i].set_kp_and_des(kp, des)


