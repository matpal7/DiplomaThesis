import numpy as np
import cv2

from tqdm import tqdm
from planar_nico_vision.image import get_l_r_image_fnames, load_l_r_images_undistorted
from planar_nico_vision.utils import load_calib_data

frameSize = (640, 480)


def show_undistorted_images(calib_dict, img_dir, correct_horizon=False, scale=1, max_imgs=None):
    imgs_l, imgs_r = load_l_r_images_undistorted(
        calib_dict, img_dir, correct_horizon=correct_horizon, max_imgs=max_imgs
    )

    for i, (img_l, img_r) in enumerate(zip(imgs_l, imgs_r)):
        if scale > 1:
            display_l = img_l.get_small_img(scale)
            display_r = img_r.get_small_img(scale)
        else:
            display_l = img_l.img
            display_r = img_r.img

        img_concat_h = cv2.hconcat([display_l, display_r])
        cv2.imshow("Undistorted images", img_concat_h)
        img_number = img_l.get_image_number()

        print(f"Showing pair {img_number}. Press any key for next, ESC to exit.")
        key = cv2.waitKey(0)

        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    out_dir = 'C:/Users/Lenovo/Desktop/DiplomaThesis_git/NICO/out/'
    calib_dir = load_calib_data(out_dir + "/calib_data.npy")
    dataset_dir = 'C:/Users/Lenovo/Desktop/DiplomaThesis/dataset/'
    calib_imgs_dir = dataset_dir + 'calibration/'
    depth_imgs_dir = dataset_dir + 'depth/rgb/'
    show_undistorted_images(calib_dir, depth_imgs_dir, scale=5, max_imgs=5)
