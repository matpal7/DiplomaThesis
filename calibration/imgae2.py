import os
import re

import cv2
import numpy as np


class Image:
    def __init__(self, img_path, undistort_function=None):
        self.pose = None
        self.img_path = img_path
        self.img_basename = os.path.basename(img_path)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image at path: {img_path}")

        self.img = undistort_function(img) if undistort_function is not None else img
        self.dims = (self.img.shape[1], self.img.shape[0])

    def get_small_img(self, scale=4):
        mini_dims = self.scaled_dims(scale)
        return cv2.resize(self.img, mini_dims)

    def scaled_dims(self, scale):
        return (self.dims[0] // scale, self.dims[1] // scale)

    def get_image_number(self):
        basename = os.path.basename(self.img_path)
        match = re.match(r'(\d+)_', basename)
        return int(match.group(1)) if match else None

    def get_path(self):
        return self.img_path

    def get_resized_img(self, resolution):
        return cv2.resize(self.img, resolution, interpolation=cv2.INTER_AREA)

class ImageRGBD(Image):
    def __init__(self, img_path, undistort_function=None, undistort_depth=False):
        super().__init__(img_path, undistort_function=undistort_function)

        depth_path = self._get_depth_path()
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        self.depth = np.load(depth_path).astype(np.float32)

        if undistort_function is not None and undistort_depth:
            self.depth = undistort_function(self.depth)