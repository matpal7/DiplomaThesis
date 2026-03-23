import os.path
from pathlib import Path

import cv2
import numpy as np
import re

from utils import get_l_r_image_fnames, get_depth_rgb_image_fnames, load_dict


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
    def __init__(self, img_path, undistort_function=None, undistort_depth=True):
        super().__init__(img_path, undistort_function=undistort_function)

        depth_path = self._get_depth_path()
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        self.depth = np.load(depth_path).astype(np.float32)

        if undistort_function is not None and undistort_depth:
            self.depth = undistort_function(self.depth)

    def _get_depth_path(self):
        base = Path(self.img_path)
        depth_name = base.stem + "_depth.npy"
        depth_path = base.parent.parent / "depth" / depth_name
        return str(depth_path)

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

def load_calib_data(calib_file, type):
    allowed_types = {"left", "right", "mono", "zed", "realsense"}
    if type not in allowed_types:
        raise ValueError(
            f"Invalid calib_type '{type}'. Allowed values are: {sorted(allowed_types)}"
        )
    # mono-like YAML calibrations
    if type in ("mono", "zed", "realsense"):
        return load_yaml_calibration(calib_file)

    # stereo calib_dict
    calib = load_dict(calib_file)

    if type == "left":
        calib["K"] = calib["new_K_l"].copy()
        calib["image_size"] = tuple(int(x) for x in np.asarray(calib["img_dim_l"]).reshape(-1)[:2])
    else:
        calib["K"] = calib["new_K_r"]
        calib["image_size"] = tuple(int(x) for x in np.asarray(calib["img_dim_r"]).reshape(-1)[:2])
    calib["D"] = np.zeros((5, 1), dtype=np.float64)
    return calib



def get_undistort_functions(calib_file, stereo=True):
    if stereo:
        return get_undistort_functions_stereo(calib_file)
    return get_undistort_function_mono(calib_file)

def get_undistort_functions_stereo(calib_file, correct_horizon=False):
    if isinstance(calib_file, (str, Path, os.PathLike)):
        calib_dict = load_dict(calib_file)
    else:
        calib_dict = calib_file
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
    undistort_l, undistort_r = get_undistort_functions(calib_dict)
    fnames_l, fnames_r = get_l_r_image_fnames(img_dir, max_imgs=max_imgs)

    imgs_l = [Image(fname, undistort_l) for fname in fnames_l]
    imgs_r = [Image(fname, undistort_r) for fname in fnames_r]

    return imgs_l, imgs_r

def load_l_r_images_rectified(calib_dict, img_dir, max_imgs=None):
    undistort_l, undistort_r, _ = get_rectify_functions(calib_dict)
    fnames_l, fnames_r = get_l_r_image_fnames(img_dir, max_imgs=max_imgs)

    imgs_l = [Image(fname, undistort_l) for fname in fnames_l]
    imgs_r = [Image(fname, undistort_r) for fname in fnames_r]

    return imgs_l, imgs_r

def load_rgbd_images(img_dir, suffix="realsense", max_imgs=None):
    fname_rgb = get_depth_rgb_image_fnames(img_dir, suffix=suffix, max_imgs=max_imgs)
    imgs_rgb = [ImageRGBD(fname) for fname in fname_rgb]

    return imgs_rgb


def _as_omnidir_xi(xi) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    if xi.size != 1:
        raise ValueError(f"xi must contain exactly one value, got shape={xi.shape}, value={xi}")
    return xi.reshape(1, 1)

def _pick_new_k(calib: dict, left: bool, use_wide: bool, balance: float) -> np.ndarray:
    base_key = "K_l" if left else "K_r"
    new_key = "new_K_l" if left else "new_K_r"
    wide_key = "new_K_l_wide" if left else "new_K_r_wide"

    if use_wide and wide_key in calib:
        K = np.asarray(calib[wide_key], dtype=np.float64)
    elif new_key in calib:
        K = np.asarray(calib[new_key], dtype=np.float64)
    else:
        K = np.asarray(calib[base_key], dtype=np.float64).copy()

    K[0, 1] = 0.0
    if use_wide and wide_key not in calib and balance > 0:
        # fallback "wider" view by reducing focal length
        K[0, 0] /= (1.0 + balance)
        K[1, 1] /= (1.0 + balance)

    return K

def get_rectify_functions(
    calib_dict: dict,
    use_wide: bool = False,
    balance: float = 0.0,
):
    """
    Returns rectification functions for left/right stereo images.

    Output images are:
    - undistorted
    - rectified to a common stereo geometry
    - in perspective/pinhole model

    Returns:
        rectify_l, rectify_r, rect_data

    rect_data contains:
        R1, R2, P1, P2, Q, roi1, roi2, img_size
    """
    K_l = np.asarray(calib_dict["K_l"], dtype=np.float64).reshape(3, 3)
    D_l = np.asarray(calib_dict["D_l"], dtype=np.float64).reshape(-1)
    xi_l = _as_omnidir_xi(calib_dict["xi_l"])

    K_r = np.asarray(calib_dict["K_r"], dtype=np.float64).reshape(3, 3)
    D_r = np.asarray(calib_dict["D_r"], dtype=np.float64).reshape(-1)
    xi_r = _as_omnidir_xi(calib_dict["xi_r"])

    rvec = np.asarray(calib_dict["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(calib_dict["tvec"], dtype=np.float64).reshape(3, 1)
    R_lr, _ = cv2.Rodrigues(rvec)

    img_dim_l = tuple(int(x) for x in np.asarray(calib_dict["img_dim_l"]).reshape(-1)[:2])
    img_dim_r = tuple(int(x) for x in np.asarray(calib_dict["img_dim_r"]).reshape(-1)[:2])

    if img_dim_l != img_dim_r:
        raise ValueError(f"Left/right calibration image dimensions differ: {img_dim_l} vs {img_dim_r}")

    # OpenCV expects (width, height)
    img_size = img_dim_l

    new_K_l = _pick_new_k(calib_dict, left=True, use_wide=use_wide, balance=balance)
    new_K_r = _pick_new_k(calib_dict, left=False, use_wide=use_wide, balance=balance)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=new_K_l,
        distCoeffs1=np.zeros((4, 1), dtype=np.float64),
        cameraMatrix2=new_K_r,
        distCoeffs2=np.zeros((4, 1), dtype=np.float64),
        imageSize=img_size,
        R=R_lr,
        T=tvec,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map1_l, map2_l = cv2.omnidir.initUndistortRectifyMap(
        K_l,
        D_l,
        xi_l,
        R1,
        P1[:3, :3],
        img_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE,
    )

    map1_r, map2_r = cv2.omnidir.initUndistortRectifyMap(
        K_r,
        D_r,
        xi_r,
        R2,
        P2[:3, :3],
        img_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE,
    )

    def rectify_l(img: np.ndarray) -> np.ndarray:
        return cv2.remap(
            img, map1_l, map2_l,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

    def rectify_r(img: np.ndarray) -> np.ndarray:
        return cv2.remap(
            img, map1_r, map2_r,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

    rect_data = {
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi1": roi1,
        "roi2": roi2,
        "img_size": img_size,
        "K_rect_l": P1[:3, :3].copy(),
        "K_rect_r": P2[:3, :3].copy(),
        "D_rect_l": np.zeros((5, 1), dtype=np.float64),
        "D_rect_r": np.zeros((5, 1), dtype=np.float64),
    }

    return rectify_l, rectify_r, rect_data

def extract_key(p: Path) -> str:
    return p.stem.split("_")[0]

def numeric_key(p: Path) -> int:
    return int(extract_key(p))

def load_rgb_depth_pairs(base_dir, suffix: str, max_imgs=None):
    base_dir = Path(base_dir)
    rgb_dir = base_dir / "rgb"
    depth_dir = base_dir / "depth"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    rgb_paths = sorted(
        rgb_dir.glob(f"*_{suffix}.png"),
        key=numeric_key
    )

    depth_candidates = {}
    for p in depth_dir.glob(f"*_{suffix}_depth.npy"):
        depth_candidates[extract_key(p)] = p

    imgs = []
    depths = []

    for rgb_path in rgb_paths:
        key = extract_key(rgb_path)
        depth_path = depth_candidates.get(key)

        if depth_path is None:
            continue

        rgb_img = ImageRGBD(str(rgb_path))
        depth = np.load(depth_path)

        imgs.append(rgb_img)
        depths.append(depth)

        if max_imgs is not None and len(imgs) >= max_imgs:
            break

    return imgs, depths

def load_yaml_calibration(yaml_path: Path) -> dict:
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration yaml file: {yaml_path}")

    try:
        K = fs.getNode("K").mat()
        D = fs.getNode("D").mat()

        image_width_node = fs.getNode("image_width")
        image_height_node = fs.getNode("image_height")

        image_width = int(image_width_node.real()) if not image_width_node.empty() else None
        image_height = int(image_height_node.real()) if not image_height_node.empty() else None
    finally:
        fs.release()

    if K is None or D is None:
        raise ValueError(f"Calibration YAML must contain nodes 'K' and 'D': {yaml_path}")

    return {
        "K": np.asarray(K, dtype=np.float64).reshape(3, 3),
        "D": np.asarray(D, dtype=np.float64),
        "image_size": (image_width, image_height) if image_width is not None and image_height is not None else None,
    }


def get_undistort_function_mono(calib):
    K = calib["K"]
    D = calib["D"]
    image_size = calib["image_size"]

    if image_size is None:
        raise ValueError("YAML must contain image_width and image_height")

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=D,
        R=np.eye(3, dtype=np.float64),
        newCameraMatrix=K,
        size=image_size,
        m1type=cv2.CV_16SC2,
    )

    def undistort(img: np.ndarray) -> np.ndarray:
        if (img.shape[1], img.shape[0]) != image_size:
            raise ValueError(
                f"Image size {(img.shape[1], img.shape[0])} does not match calibration size {image_size}"
            )

        return cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    return undistort