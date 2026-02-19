import os
import re
import time

import numpy as np
import pyrealsense2 as rs
import cv2
from tqdm import tqdm

from utils import get_undistort_functions, load_calib_data

# width = 3840
# height = 2160
width = 1280
height = 720
frame_size = (680, 480)


def getCalibrationPhoto(camara_index_left, camera_index_right, save_dir, frame_size=(width, height), chessboard_size=(7,6)):
    # cap_L = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #
    # cap_R = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap_l = opencv_open_camera(camara_index_left, frame_size)
    cap_r = opencv_open_camera(camera_index_right, frame_size)

    calib_dir = os.path.join(save_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    num = findMaxNumber(calib_dir)
    print("Started", num)
    frame_size_small = (680, 480)

    while cap_r.isOpened():
        success_L, image_L = cap_l.read()
        success_R, image_R = cap_r.read()
        if not success_L or not success_R:
            print("Failed to capture images from one or both cameras")
            continue  # Skip the rest of the loop if image capture failed

        grayL = cv2.cvtColor(image_L, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(image_R, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)
        imgL = image_L.copy()
        imgR = image_R.copy()

        if retL and retR == True:
            imgL = cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
            imgR = cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)

        resized_frame_L = cv2.resize(imgL, frame_size_small)
        resized_frame_R = cv2.resize(imgR, frame_size_small)
        img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('Img', img_concat_h)

        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            # cv2.imwrite(calib_dir + "/left/" + f"{num}_left.png", image_L)
            # cv2.imwrite(calib_dir + "/right/" f"{num}_right.png", image_R)
            cv2.imwrite(calib_dir + f"/{num}_left.png", image_L)
            cv2.imwrite(calib_dir + f"/{num}_right.png", image_R)
            print("{} saved".format(num))
            num += 1

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def findMaxNumber(directory):
    max_number = -1
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    pattern = r'\d+'

    for filename in files:
        numbers = re.findall(pattern, filename)
        if numbers:  # Check if any numbers were found
            current_max = max(map(int, numbers))
            if current_max > max_number:
                max_number = current_max

    return max_number + 1


def show_real_time(calib_dict, dirc, calib_dir):
    cap_L = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap_R = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    num = findMaxNumber(dirc)
    print("Max number: ", num)
    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=False)

    while cap_L.isOpened():
        succes, image_L = cap_L.read()
        succes, image_R = cap_R.read()

        img_l = image_L.copy()
        img_r = image_R.copy()

        img_l = undistort_l(img_l)
        img_r = undistort_r(img_r)

        resized_frame_L = cv2.resize(img_l, frame_size)
        resized_frame_R = cv2.resize(img_r, frame_size)

        img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('Unidistored', img_concat_h)
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(dirc + "" + str(num) + "_l.png", image_L)
            cv2.imwrite(dirc + "" + str(num) + "_r.png", image_R)
            print("{} saved".format(num))
            num += 1

def opencv_camera_capture(cap):
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from one or more cameras")
        return None
    return frame

def opencv_open_camera(camera_index, frame_size):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    return camera

def save_cameras_on_click(camara_index_left, camera_index_right, calib_dict, frame_size_RGB=(1280, 720), frame_size_RealSense=(1280, 720), save_dir="dataset_default"):
    """
    Captures and displays frames from multiple cameras and a RealSense device.
    Press SPACE to view frames, S to save frames & depth, ESC to exit.
    Saves RGB images to /rgb and depth data to /depth.
    """
    # Create directories
    frame_size_display = (680, 480)
    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Open regular cameras
    cap_l = opencv_open_camera(camara_index_left, frame_size_RGB)
    cap_r = opencv_open_camera(camera_index_right, frame_size_RGB)

    # Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, frame_size_RealSense[0], frame_size_RealSense[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, frame_size_RealSense[0], frame_size_RealSense[1], rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    undistort_l, undistort_r = get_undistort_functions(calib_dict, get_wide=False)

    print("Press SPACE to capture and display frames, S to save, ESC to exit.")
    num = findMaxNumber(rgb_dir)
    print("Max number: ", num)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

        # Capture frames
        img_l = opencv_camera_capture(cap_l)
        img_r = opencv_camera_capture(cap_r)
        img_l_origin = img_l.copy()
        img_r_origin = img_r.copy()
        # img_l = undistort_l(img_l)
        # img_r = undistort_r(img_r)
        if img_l is None or img_r is None:
            continue

        # RealSense capture
        rs_frames = pipeline.wait_for_frames()
        rs_frames = align.process(rs_frames)
        color_frame = rs_frames.get_color_frame()
        depth_frame = rs_frames.get_depth_frame()

        if color_frame and depth_frame:
            img_RealSense = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colored = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        else:
            print("Failed to capture RealSense frames")
            continue

        # Show concatenated view
        img_concat = cv2.hconcat([
            cv2.resize(img_l, frame_size_display),
            cv2.resize(img_r, frame_size_display),
            cv2.resize(img_RealSense, frame_size_display)
        ])
        cv2.imshow('Captured Frames', img_concat)

        # Save if 's' pressed
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Save RGB
            cv2.imwrite(os.path.join(rgb_dir, f"{num}_left.png"), img_l_origin)
            cv2.imwrite(os.path.join(rgb_dir, f"{num}_right.png"), img_r_origin)
            cv2.imwrite(os.path.join(rgb_dir, f"{num}_realsense.png"), img_RealSense)
            np.save(os.path.join(depth_dir, f"{num}_depth.npy"), depth_image)
            print(f"Saved captures number: {num} to {save_dir}")
            num += 1

    # Release resources
    cap_l.release()
    cap_r.release()
    pipeline.stop()
    cv2.destroyAllWindows()


def create_dataset_directory(name="dataset"):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(parent_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

if __name__ == '__main__':
    chessboard_size = (8,6)
    save_dir = create_dataset_directory("dataset")
    # save_cameras_on_click(camera_index_right=0, camara_index_left=3, save_dir=save_dir)

    # getCalibrationPhoto(1, 3, save_dir, chessboard_size=chessboard_size)

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(parent_dir, 'out')
    print(out_dir + "/calib_data.npy")
    calib_dir = load_calib_data(out_dir + "/calib_data.npy")
    dataset_dir = os.path.join(parent_dir, 'dataset')
    depth_dir = os.path.join(dataset_dir, 'depth')
    #show_undistored(depth_dir + "/rgb", calib_dir)
    save_cameras_on_click(3, 1, calib_dir, save_dir=depth_dir)
