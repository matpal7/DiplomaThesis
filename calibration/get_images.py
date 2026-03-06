import os
import re
import time

import numpy as np
import pyrealsense2 as rs
import cv2
import pyzed.sl as sl

from utils import load_dict
from calibration.image import get_undistort_functions

width_4K = 3840
height_4K = 2160
width = 1280
height = 720
frame_size = (680, 480)
frame_size_HD = (height, width)
frame_size_4K = (width_4K, height_4K)

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

def save_cameras_on_click(camara_index_left, camera_index_right, calib_dict=None, frame_size_RealSense=(1280, 720), save_dir="dataset_default"):
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
    cap_l = opencv_open_camera(camara_index_left, frame_size_4K)
    cap_r = opencv_open_camera(camera_index_right, frame_size_4K)

    # Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.color, frame_size_RealSense[0], frame_size_RealSense[1], rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, frame_size_RealSense[0], frame_size_RealSense[1], rs.format.z16, 30)
    # pipeline.start(config)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Enable left and right infrared cameras (Stereo)
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    # Enable depth stream
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    for _ in range(10):
        pipeline.wait_for_frames()

    align = rs.align(rs.stream.color)

    if calib_dict is not None:
        undistort_l, undistort_r = get_undistort_functions(calib_dict, get_wide=False)
    frame_size_small = (680, 480)

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
        if calib_dict is not None:
            img_l = undistort_l(img_l)
            img_r = undistort_r(img_r)
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

def save_realsense_calibration(filename="realsense_calibration.yaml"):

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()

    pipeline.stop()

    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])

    D = np.array(intr.coeffs)

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("D", D)
    fs.release()

    print("Calibration saved to", filename)

def read_zed_image():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera open failed:", status)
        return None

    image = sl.Mat()

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        zed.close()
        return frame

    zed.close()
    return None

def show_zed_image():
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        img = read_zed_image()
        if img is not None:
            cv2.imshow("ZED Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No image captured")

if __name__ == '__main__':
    chessboard_size = (8,6)
    # parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # out_dir = os.path.join(parent_dir, 'out')
    # print(out_dir + "/calib_data.npy")
    #
    # # calib_dir = load_dict(out_dir + "/calib_data.npy")
    # dataset_dir = os.path.join(parent_dir, 'dataset_05032026')
    # depth_dir = os.path.join(dataset_dir, 'depth')
    # save_cameras_on_click(4, 3, save_dir=depth_dir)

    # #show_undistored(depth_dir + "/rgb", calib_dir)

    show_zed_image()
