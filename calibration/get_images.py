import os
import re
import time
from pathlib import Path

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

def _get_zed_resolution(width_px, height_px):
    """Return the closest ZED SDK resolution enum for requested frame size."""
    if width_px >= 2208 and height_px >= 1242:
        return sl.RESOLUTION.HD2K
    if width_px >= 1920 and height_px >= 1080:
        return sl.RESOLUTION.HD1080
    if width_px >= 1280 and height_px >= 720:
        return sl.RESOLUTION.HD720
    return sl.RESOLUTION.VGA


def _open_zed_camera(frame_size_zed=(1280, 720), zed_depth_mode=sl.DEPTH_MODE.QUALITY):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = _get_zed_resolution(*frame_size_zed)
    init_params.depth_mode = zed_depth_mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("ZED camera open failed:", status)
        return None
    return zed

def _make_labeled_tile(image, label, frame_size_display):
    if image is None:
        tile = np.zeros((frame_size_display[1], frame_size_display[0], 3), dtype=np.uint8)
    else:
        tile = cv2.resize(image, frame_size_display)

    cv2.rectangle(tile, (0, 0), (220, 28), (0, 0, 0), -1)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return tile

def _get_omni_undistort_functions(calib_dict):
    """Return omnidirectional undistortion functions when calibration data is available."""
    if calib_dict is None:
        return None, None

    try:
        return get_undistort_functions(calib_dict, correct_horizon=False)
    except Exception as exc:
        print(f"Failed to initialize omnidirectional undistortion: {exc}")
        return None, None




def save_cameras_on_click(
        camara_index_left,
        camera_index_right,
        calib_dict=None,
        frame_size_stereo=(1280, 720),
        frame_size_realsense_rgb=(1280, 720),
        frame_size_realsense_depth=(1280, 720),
        frame_size_zed=(1280, 720),
        save_dir="dataset_default",
        use_realsense = True,
        use_zed = True):

    """
    Captures and displays frames from stereo cameras and optionally RealSense / ZED M.

    SPACE – capture frame
    S     – save frame
    ESC   – exit

    Defaults are tuned for Intel RealSense D435i and ZED M (HD720 at 30 FPS).
    Depth is saved for every enabled depth camera.
    """


    frame_size_display = (680, 480)

    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    if use_realsense or use_zed:
        os.makedirs(depth_dir, exist_ok=True)

    # ---------------------------
    # Open normal cameras
    # ---------------------------
    cap_l = opencv_open_camera(camara_index_left, frame_size_stereo)
    cap_r = opencv_open_camera(camera_index_right, frame_size_stereo)

    # ---------------------------
    # Setup RealSense (optional)
    # ---------------------------
    if use_realsense:
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(
            rs.stream.color,
            frame_size_realsense_rgb[0],
            frame_size_realsense_rgb[1],
            rs.format.bgr8,
            30,
        )
        config.enable_stream(
            rs.stream.depth,
            frame_size_realsense_depth[0],
            frame_size_realsense_depth[1],
            rs.format.z16,
            30,
        )

        pipeline.start(config)

        for _ in range(10):
            pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)

    # ---------------------------
    # Setup ZED (optional)
    # ---------------------------
    if use_zed:
        zed = _open_zed_camera(frame_size_zed=frame_size_zed, zed_depth_mode=sl.DEPTH_MODE.QUALITY)
        if zed is None:
            use_zed = False
            print("Continuing without ZED stream.")

    # ---------------------------
    # Undistortion
    # ---------------------------
    undistort_l, undistort_r = _get_omni_undistort_functions(calib_dict)

    print("Press SPACE to capture, S to save, ESC to exit.")

    num = findMaxNumber(rgb_dir)
    print("Max number:", num)

    while True:

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

        # ---------------------------
        # Capture left/right cameras
        # ---------------------------
        img_l = opencv_camera_capture(cap_l)
        img_r = opencv_camera_capture(cap_r)

        img_l_origin = img_l.copy()
        img_r_origin = img_r.copy()

        if undistort_l is not None and undistort_r is not None:
            img_l = undistort_l(img_l)
            img_r = undistort_r(img_r)

        if img_l is None or img_r is None:
            continue
        img_realsense = None
        depth_image = None
        img_zed = None
        depth_zed = None

        # ---------------------------
        # Capture RealSense (optional)
        # ---------------------------
        if use_realsense:
            rs_frames = pipeline.wait_for_frames()
            rs_frames = align.process(rs_frames)

            color_frame = rs_frames.get_color_frame()
            depth_frame = rs_frames.get_depth_frame()

            if color_frame and depth_frame:
                img_realsense = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                depth_colored = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
            else:
                print("Failed to capture RealSense frames")
                continue
                # ---------------------------
                # Capture ZED (optional)
                # ---------------------------
        if use_zed:
            zed_img_mat = sl.Mat()
            zed_depth_mat = sl.Mat()

            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(zed_img_mat, sl.VIEW.LEFT)
                zed.retrieve_measure(zed_depth_mat, sl.MEASURE.DEPTH)

                img_zed = zed_img_mat.get_data()
                img_zed = cv2.cvtColor(img_zed, cv2.COLOR_BGRA2BGR)

                depth_zed = zed_depth_mat.get_data().copy()

                depth_zed_display = np.nan_to_num(depth_zed, nan=0.0, posinf=0.0, neginf=0.0)
                depth_zed_colored = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_zed_display, alpha=0.02),
                    cv2.COLORMAP_TURBO,
                )
            else:
                print("Failed to capture ZED frames")
                continue
        # ---------------------------
        # Show preview
        # ---------------------------
        views = [
            cv2.resize(img_l, frame_size_display),
            cv2.resize(img_r, frame_size_display)
        ]

        if use_realsense:
            views.append(cv2.resize(img_realsense, frame_size_display))
        if use_zed:
            views.append(cv2.resize(img_zed, frame_size_display))

        # ---------------------------
        # Show 2x2 preview
        # ---------------------------
        top_left = _make_labeled_tile(img_l, "Stereo Left", frame_size_display)
        top_right = _make_labeled_tile(img_r, "Stereo Right", frame_size_display)
        bottom_left = _make_labeled_tile(img_realsense, "RealSense RGB", frame_size_display)
        bottom_right = _make_labeled_tile(img_zed, "ZED RGB", frame_size_display)

        top_row = cv2.hconcat([top_left, top_right])
        bottom_row = cv2.hconcat([bottom_left, bottom_right])
        preview = cv2.vconcat([top_row, bottom_row])
        cv2.imshow("Captured Frames", preview)

        # ---------------------------
        # Save
        # ---------------------------
        if key == ord('s'):

            cv2.imwrite(os.path.join(rgb_dir, f"{num}_left.png"), img_l_origin)
            cv2.imwrite(os.path.join(rgb_dir, f"{num}_right.png"), img_r_origin)

            if use_realsense:
                cv2.imwrite(os.path.join(rgb_dir, f"{num}_realsense.png"), img_realsense)
                np.save(os.path.join(depth_dir, f"{num}_realsense_depth.npy"), depth_image)
            if use_zed:
                cv2.imwrite(os.path.join(rgb_dir, f"{num}_zed.png"), img_zed)
                np.save(os.path.join(depth_dir, f"{num}_zed_depth.npy"), depth_zed)

            print(f"Saved capture {num}")
            num += 1

        elif key == ord("q"):
            break

    # ---------------------------
    # Release resources
    # ---------------------------
    cap_l.release()
    cap_r.release()

    if use_realsense:
        pipeline.stop()
    if use_zed:
        zed.close()

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
    parent_dir = Path(__file__).resolve().parent.parent

    # calib_dir = load_dict(out_dir + "/calib_data.npy")
    dataset_dir = os.path.join(parent_dir, 'dataset_11032026')
    depth_dir = os.path.join(dataset_dir, 'stereo_4k_depth')

    out_dir = parent_dir / "NICO" / "out_1103"
    calib_dict = load_dict(out_dir / "calib_data.npy")

    save_cameras_on_click(4,2, frame_size_stereo=frame_size_4K, save_dir=depth_dir, use_realsense=True, use_zed=True, calib_dict=calib_dict)

    # #show_undistored(depth_dir + "/rgb", calib_dir)

    # show_zed_image()
