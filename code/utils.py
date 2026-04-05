import glob
import os
from pathlib import Path

import numpy as np
import cv2
import re


def load_dict(npy_path):
    calib_dict = np.load(npy_path, allow_pickle=True).item()
    return calib_dict

def save_dict(calib_dict, out_folder, file_name='calib_data.npy'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    npy_path = os.path.join(out_folder, file_name)
    np.save(npy_path, calib_dict)
    print("Wrote calib data to: ", npy_path)


def _idx(p):
    return int(Path(p).stem.split("_")[0])

def get_l_r_image_fnames(img_folder, max_imgs=None):
    glob_string_l = '{}/*_left.png'.format(img_folder)
    glob_string_r = '{}/*_right.png'.format(img_folder)
    images_l = glob.glob(glob_string_l)
    images_r = glob.glob(glob_string_r)

    images_l = sorted(images_l, key=_idx)
    images_r = sorted(images_r, key=_idx)

    if max_imgs is not None:
        images_l = images_l[:max_imgs]
        images_r = images_r[:max_imgs]

    return images_l, images_r

def get_depth_rgb_image_fnames(img_folder, suffix="realsense", max_imgs=None):
    glob_string_d = f"{img_folder}/*_{suffix}.png"
    images_d = sorted(glob.glob(glob_string_d), key=_idx)

    if max_imgs is not None:
        images_d = images_d[:max_imgs]

    return images_d

def scale_intrinsics(K: np.ndarray, old_size: tuple[int, int], new_size: tuple[int, int]) -> np.ndarray:
    """
    Scale camera intrinsics after resizing image.

    old_size: (width, height)
    new_size: (width, height)
    """
    old_w, old_h = old_size
    new_w, new_h = new_size

    sx = new_w / float(old_w)
    sy = new_h / float(old_h)

    K_new = K.copy().astype(np.float64)
    K_new[0, 0] *= sx  # fx
    K_new[1, 1] *= sy  # fy
    K_new[0, 2] *= sx  # cx
    K_new[1, 2] *= sy  # cy

    return K_new


def load_estimated_depth_map(out_dir: Path, NNname: str, max_imgs: int = None) -> list[np.ndarray]:
    depth_maps_dir = out_dir / NNname / "depth"
    depth_paths = sorted(
        depth_maps_dir.glob("*_depth.npy"),
        key=lambda p: _idx(p.name)
    )

    if max_imgs is not None:
        depth_paths = depth_paths[:max_imgs]

    depth_maps = [np.load(p) for p in depth_paths]

    return depth_maps




