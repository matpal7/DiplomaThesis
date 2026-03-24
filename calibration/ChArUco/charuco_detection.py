import os
from pathlib import Path
import cv2
import numpy as np
from sympy.codegen.ast import continue_

from calibration.image import get_undistort_functions
from utils import get_l_r_image_fnames, load_dict, save_dict

ARUCO_DICT = cv2.aruco.DICT_4X4_250
SQUARES_HORIZONTALLY = 5
SQUARES_VERTICALLY = 7
SQUARE_LENGTH = 0.054
MARKER_LENGTH = 0.037
TRESHOLD_CORNERS = 20

def create_charuco_board(squares_horizontally=6, squares_vertically=8, squares_length=32.0, marker_length=22.0):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (squares_horizontally, squares_vertically),
        squares_length,
        marker_length,
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

def detect_charuco_in_image_live(img, board, aruco_detector, undistort_fn=None, min_corners=TRESHOLD_CORNERS):
    """
    Detect ChArUco board in an already loaded image.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image
    board : cv2.aruco.CharucoBoard
        ChArUco board object
    aruco_detector : cv2.aruco.ArucoDetector
        Configured ArUco detector
    undistort_fn : callable | None
        Optional image undistortion function
    min_corners : int
        Minimum number of interpolated ChArUco corners required

    Returns
    -------
    charuco_corners : np.ndarray | None
    charuco_ids : np.ndarray | None
    vis_img : np.ndarray
        Visualization with detected markers/corners drawn
    """

    if img is None:
        raise ValueError("Input image is None")

    if undistort_fn is not None:
        img = undistort_fn(img)

    vis_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    detected_corner_count = 0


    if marker_ids is not None and len(marker_ids) > 0:
        # cv2.aruco.drawDetectedMarkers(vis_img, marker_corners, marker_ids)

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )

        if retval is not None and charuco_ids is not None:
            detected_corner_count = len(charuco_ids)

        if retval is not None and retval >= min_corners and charuco_ids is not None:
            # vis_img = cv2.aruco.drawDetectedCornersCharuco(
            #     vis_img,
            #     charuco_corners,
            #     None, #charuco_ids,
            #     cornerColor=(0, 0, 255)
            # )
            for pt in charuco_corners.reshape(-1, 2):
                x, y = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(vis_img, (x, y), 8, (0, 0, 255), 6)  # fill
                # cv2.circle(vis_img, (x, y), 29, (255, 255, 255), 2)

        text = f"Corners: {detected_corner_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        margin = 15
        x = vis_img.shape[1] - text_w - margin
        y = margin + text_h

        cv2.rectangle(
            vis_img,
            (x - 8, y - text_h - 8),
            (x + text_w + 8, y + baseline + 8),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis_img,
            text,
            (x, y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return vis_img

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
    out_dir = parent_dir / "out" / "cameras_parameters"
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