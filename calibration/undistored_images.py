import cv2

from calibration.image import load_l_r_images_undistorted, load_realsense_rgb_images
from utils import load_dict

frameSize = (640, 480)


def show_undistorted_images(calib_dict, img_dir, correct_horizon=False, scale=4, max_imgs=None):
    imgs_l, imgs_r = load_l_r_images_undistorted(
        calib_dict, img_dir, correct_horizon=correct_horizon, max_imgs=max_imgs
    )
    print(len(imgs_l))
    imgs_rgb = load_realsense_rgb_images(img_dir, max_imgs=max_imgs)
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

        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    out_dir = '/NICO/out/'
    calib_dir = load_dict(out_dir + "calib_data.npy")
    dataset_dir = 'C:/Users/matej/Desktop/DiplomaThesis/dataset/'
    calib_imgs_dir = dataset_dir + 'calibration/'
    depth_imgs_dir = dataset_dir + 'depth/rgb/'
    show_undistorted_images(calib_dir, calib_imgs_dir, scale=4, max_imgs=20)
