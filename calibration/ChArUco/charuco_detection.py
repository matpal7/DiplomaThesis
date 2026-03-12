import os
from pathlib import Path
import cv2
import numpy as np

from calibration.image import get_undistort_functions
from utils import get_l_r_image_fnames, load_dict, save_dict

ARUCO_DICT = cv2.aruco.DICT_4X4_250
SQUARES_HORIZONTALLY = 5
SQUARES_VERTICALLY = 7
SQUARE_LENGTH = 0.054
MARKER_LENGTH = 0.037
TRESHOLD_CORNERS = 12

def create_charuco_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    return aruco_dict, board, aruco_detector


def detect_charuco_in_image(img_path, board, aruco_detector, undistored_l=None):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None, None, None
    if undistored_l is not None:
        img = undistored_l(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, gray.shape[::-1]

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board
    )

    if retval is None or retval < TRESHOLD_CORNERS or charuco_ids is None:
        return None, None, gray.shape[::-1]

    return charuco_corners, charuco_ids, gray.shape[::-1]


def collect_charuco_detections(image_paths, board, aruco_detector, undistored_l=None):
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    for img_path in image_paths:
        charuco_corners, charuco_ids, current_image_size = detect_charuco_in_image(
            img_path, board, aruco_detector, undistored_l=undistored_l
        )

        if current_image_size is not None and image_size is None:
            image_size = current_image_size

        if charuco_corners is None or charuco_ids is None:
            print(f"{img_path.name}: rejected")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        print(f"{img_path.name}: accepted, corners={len(charuco_ids)}")

    return all_charuco_corners, all_charuco_ids, image_size


def calibrate_camera_from_images(image_paths, out_dir, file_name="calib_data.npy", undistored_l = None):
    _, board, aruco_detector = create_charuco_board()

    all_charuco_corners, all_charuco_ids, image_size = collect_charuco_detections(
        image_paths, board, aruco_detector, undistored_l=undistored_l
    )

    if image_size is None:
        raise RuntimeError("No valid images found.")

    if len(all_charuco_corners) < 5:
        raise RuntimeError("Not enough valid calibration images.")

    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("\n=== Calibration finished ===")
    print("Reprojection error:", ret)
    print("K:\n", K)
    print("dist:\n", dist)

    calib_dict = {
        "K": K,
        "dist": dist,
        "image_size": image_size,
        "reprojection_error": ret
    }
    # save_dict(calib_dict, out_dir, file_name)

    return ret, K, dist, rvecs, tvecs


def main():
    parent_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = parent_dir / "dataset_11032026"
    calib_imgs_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    out_dir = parent_dir / "NICO" / "out_2"

    out_dir = parent_dir / "NICO" / "out_1103"
    calib_dirc_left_path = out_dir / "calib_data.npy"

    calib_dict = load_dict(calib_dirc_left_path)

    undistored_l, _ = get_undistort_functions(calib_dict, correct_horizon=False)

    left_images = sorted(calib_imgs_dir.glob("*_left.png"))
    # right_images = sorted(calib_imgs_dir.glob("*_right.png"))

    print("=== Calibrating LEFT camera ===")
    calibrate_camera_from_images(
        left_images,
        out_dir,file_name="left_calib.npy", undistored_l=undistored_l
    )

    # print("\n=== Calibrating RIGHT camera ===")
    # calibrate_camera_from_images(
    #     right_images,
    #     out_dir,
    #     file_name="right_calib.npy"
    # )


if __name__ == "__main__":
    main()