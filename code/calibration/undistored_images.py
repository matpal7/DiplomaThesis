import os
from pathlib import Path

import cv2
import numpy as np

from code.image import load_l_r_images_undistorted, load_rgbd_images, get_undistort_functions, \
    load_l_r_images_rectified
from code.utils import load_dict

frameSize = (1280, 720)


def show_undistorted_images(calib_dict, img_dir, scale=4, max_imgs=None):
    imgs_l, imgs_r = load_l_r_images_undistorted(
        calib_dict, img_dir, max_imgs=max_imgs
    )
    print(len(imgs_l))
    imgs_rgb = load_rgbd_images(img_dir, max_imgs=max_imgs)
    for i, (img_l, img_r) in enumerate(zip(imgs_l, imgs_r)):
        img_d = None
        if len(imgs_rgb) != 0:
            img_d = imgs_rgb[i]
        if scale > 1:
            display_l = img_l.get_small_img(scale)
            display_r = img_r.get_small_img(scale)
            if img_d is not None:
                display_d = img_d.get_small_img(scale)
        else:
            display_l = img_l.img
            display_r = img_r.img
            if img_d is not None:
                display_d = img_d.img

        img_concat_h = cv2.hconcat([display_l, display_r])
        for y in range(0, img_concat_h.shape[0], 40):
            cv2.line(img_concat_h, (0, y), (img_concat_h.shape[1], y), (0, 255, 0), 1)
        cv2.imshow("Undistorted images", img_concat_h)
        if img_d is not None:
            cv2.imshow("Realsense image", display_d)
        img_number = img_l.get_image_number()

        print(f"Showing pair {img_number}. Press any key for next, ESC to exit.")
        key = cv2.waitKey(0)

        if key == ord('s'):
            out_folder = "C:/Users/matej/Desktop/DiplomaThesis_git/NICO/undistorted_output"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            left_path = os.path.join(out_folder, f"{img_number}_left_undist.png")
            right_path = os.path.join(out_folder, f"{img_number}_right_undist.png")

            scale2 = 6
            cv2.imwrite(left_path, img_l.get_small_img(scale2))
            cv2.imwrite(right_path, img_r.get_small_img(scale2))

            print("Saved undistorted images to:", out_folder)


        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

def undistorted_image_left(img, calib_dict):
    calib_dict = load_dict(calib_dict)
    undistort_l, undistort_r = get_undistort_functions(calib_dict)
    img = undistort_l(img)
    return img


if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parent.parent

    dataset_dir = parent_dir / "dataset_11032026"
    calib_imgs_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"

    out_dir = parent_dir / "out" / "cameras_parameters"

    calib_dict = load_dict(out_dir / "calib_data.npy")

    out_dir = parent_dir / "out" / "cameras_parameters"
    show_undistorted_images(calib_dict, calib_imgs_dir, scale=4, max_imgs=20)
