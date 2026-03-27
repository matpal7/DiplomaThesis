from pathlib import Path

import cv2
import numpy as np
from utils import get_l_r_image_fnames, load_dict, save_dict



def show_undistort_image(imgs_path, calib_data, pattern="left", show=True):
    K = calib_data["K"]
    dist = calib_data["dist"]
    w, h = calib_data["image_size"]

    image_paths = sorted(imgs_path.glob(f"*_{pattern}.png"))
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    for img_path in image_paths:

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue


        undistorted = cv2.undistort(img, K, dist, None, new_K)

        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]

        if show:
            vis = cv2.hconcat([
                cv2.resize(img, (640, 480)),
                cv2.resize(undistorted, (640, 480))
            ])

            cv2.imshow(f"Distored|Undistored", vis)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = parent_dir / "dataset_09032026"
    calib_imgs_dir = dataset_dir / "stereo" / "rgb"
    out_dir = parent_dir / "NICO" / "out_2"

    # data_calib_left = load_dict(out_dir / "left_calib.npy")
    # show_undistort_image(calib_imgs_dir, data_calib_left, pattern="left")

    data_calib_right = load_dict(out_dir / "right_calib.npy")
    show_undistort_image(calib_imgs_dir, data_calib_right, pattern="right")
