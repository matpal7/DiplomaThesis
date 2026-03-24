import os
import re
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import cv2
import pyzed.sl as sl

from calibration.ChArUco.charuco_detection import create_charuco_board, detect_charuco_in_image_live
from utils import load_dict
from calibration.image import get_undistort_functions
from visualize_depth import colorize_depth

width_4K = 3840
height_4K = 2160
width = 1280
height = 720
frame_size = (680, 480)
frame_size_HD = (width, height)
frame_size_4K = (width_4K, height_4K)

FRAME_SIZE_REALSENSE = (1280, 720)
FRAME_SIZE_ZED = (1280, 720)
FRAME_SIZE_STEREO = frame_size_4K

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


def _open_zed_camera(frame_size_zed=(1280, 720), zed_depth_mode=sl.DEPTH_MODE.ULTRA):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = _get_zed_resolution(*frame_size_zed)
    init_params.depth_mode = zed_depth_mode
    init_params.coordinate_units = sl.UNIT.METER

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
        return get_undistort_functions(calib_dict, stereo=True)
    except Exception as exc:
        print(f"Failed to initialize omnidirectional undistortion: {exc}")
        return None, None




def save_cameras_on_click_old(
        camara_index_left,
        camera_index_right,
        calib_dict=None,
        frame_size_stereo=FRAME_SIZE_STEREO,
        frame_size_realsense_rgb=FRAME_SIZE_REALSENSE,
        frame_size_realsense_depth=FRAME_SIZE_REALSENSE,
        frame_size_zed=FRAME_SIZE_ZED,
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


    frame_size_display = (480, 270)

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

        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        print("Depth scale:", depth_scale)

        for _ in range(10):
            pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)

    # ---------------------------
    # Setup ZED (optional)
    # ---------------------------
    if use_zed:
        zed = _open_zed_camera(frame_size_zed=frame_size_zed, zed_depth_mode=sl.DEPTH_MODE.ULTRA)
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
        depth_meters_realsense = None
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
                depth_meters_realsense = depth_image.astype(np.float32) * depth_scale

                depth_realsense_colored = colorize_depth(depth_meters_realsense, cv2.COLORMAP_TURBO)
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

                depth_zed = zed_depth_mat.get_data().copy().astype(np.float32)
                depth_zed_colored = colorize_depth(depth_zed, cv2.COLORMAP_TURBO)

            else:
                print("Failed to capture ZED frames")
                continue
        # ---------------------------
        # Show 3x2 preview
        # ---------------------------
        top_left = _make_labeled_tile(img_l, "Stereo Left", frame_size_display)
        top_right = _make_labeled_tile(img_r, "Stereo Right", frame_size_display)

        mid_left = _make_labeled_tile(img_realsense, "RealSense RGB", frame_size_display)
        mid_right = _make_labeled_tile(
            depth_realsense_colored if use_realsense else None,
            "RealSense Depth",
            frame_size_display
        )

        bottom_left = _make_labeled_tile(img_zed, "ZED RGB", frame_size_display)
        bottom_right = _make_labeled_tile(
            depth_zed_colored if use_zed else None,
            "ZED Depth",
            frame_size_display
        )

        row1 = cv2.hconcat([top_left, top_right])
        row2 = cv2.hconcat([mid_left, mid_right])
        row3 = cv2.hconcat([bottom_left, bottom_right])

        concat_view = [row1]
        if use_realsense:
            concat_view.append(row2)
        if use_zed:
            concat_view.append(row3)
        preview = cv2.vconcat(concat_view)
        cv2.imshow("Captured Frames", preview)

        # ---------------------------
        # Save
        # ---------------------------
        if key == ord('s'):

            cv2.imwrite(os.path.join(rgb_dir, f"{num}_left.png"), img_l_origin)
            cv2.imwrite(os.path.join(rgb_dir, f"{num}_right.png"), img_r_origin)

            if use_realsense:
                cv2.imwrite(os.path.join(rgb_dir, f"{num}_realsense.png"), img_realsense)
                np.save(os.path.join(depth_dir, f"{num}_realsense_depth.npy"), depth_meters_realsense)
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



def read_zed_image():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

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

def read_realsense_frame(
    frame_size_realsense_rgb=(1280, 720),
    frame_size_realsense_depth=(1280, 720),
):
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

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale:", depth_scale)

    align = rs.align(rs.stream.color)

    try:
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, depth_scale

        img_color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * depth_scale

        return img_color, depth_m, depth_scale

    finally:
        pipeline.stop()
def show_realsense_image_live(
    frame_size_realsense_rgb=(1280, 720),
    frame_size_realsense_depth=(1280, 720),
    show_depth=False,
):
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

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale:", depth_scale)

    align = rs.align(rs.stream.color)

    try:
        for _ in range(10):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Failed to capture RealSense frame")
                continue

            img_color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            cv2.imshow("RealSense RGB", img_color)

            if show_depth:
                depth_vis = colorize_depth(depth_m, cv2.COLORMAP_TURBO)
                cv2.imshow("RealSense Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def save_cameras_on_click(
        camara_index_left,
        camera_index_right,
        calib_dict=None,
        frame_size_stereo=FRAME_SIZE_STEREO,
        frame_size_realsense_rgb=FRAME_SIZE_REALSENSE,
        frame_size_realsense_depth=FRAME_SIZE_REALSENSE,
        frame_size_zed=FRAME_SIZE_ZED,
        save_dir="dataset_default",
        use_realsense=True,
        use_zed=True,
        squares_horizontally=6,
        squares_vertically=8,
        squares_length=32.0,
        marker_length=22.0
):

    """
    SPACE – capture new frame and show it
    S     – save currently displayed frame
    ESC/Q – exit
    """

    _, board, detector = create_charuco_board(squares_horizontally=squares_horizontally,
                                              squares_vertically=squares_vertically, squares_length=squares_length,
                                              marker_length=marker_length)

    frame_size_display = (480, 270)

    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    if use_realsense or use_zed:
        os.makedirs(depth_dir, exist_ok=True)

    cap_l = opencv_open_camera(camara_index_left, frame_size_stereo)
    cap_r = opencv_open_camera(camera_index_right, frame_size_stereo)

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

        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth scale:", depth_scale)

        for _ in range(10):
            pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)

    if use_zed:
        zed = _open_zed_camera(frame_size_zed=frame_size_zed, zed_depth_mode=sl.DEPTH_MODE.ULTRA)
        if zed is None:
            use_zed = False
            print("Continuing without ZED stream.")

    undistort_l, undistort_r = _get_omni_undistort_functions(calib_dict)

    print("Press SPACE to capture, S to save displayed frame, ESC/Q to exit.")

    num = findMaxNumber(rgb_dir)
    print("Max number:", num)

    # last displayed/captured data
    last = {
        "img_l_raw": None,
        "img_r_raw": None,
        "img_l_disp": None,
        "img_r_disp": None,
        "img_realsense": None,
        "depth_realsense": None,
        "depth_realsense_colored": None,
        "img_zed": None,
        "depth_zed": None,
        "depth_zed_colored": None,
    }

    def capture_current_frames():
        img_l = opencv_camera_capture(cap_l)
        img_r = opencv_camera_capture(cap_r)

        if img_l is None or img_r is None:
            return False

        img_l_raw = img_l.copy()
        img_r_raw = img_r.copy()

        if undistort_l is not None and undistort_r is not None:
            img_l = undistort_l(img_l)
            img_r = undistort_r(img_r)

        img_realsense = None
        img_realsense_vis = None
        depth_meters_realsense = None
        depth_realsense_colored = None

        if use_realsense:
            rs_frames = pipeline.wait_for_frames()
            rs_frames = align.process(rs_frames)

            color_frame = rs_frames.get_color_frame()
            depth_frame = rs_frames.get_depth_frame()

            if not (color_frame and depth_frame):
                print("Failed to capture RealSense frames")
                return False

            img_realsense = np.asanyarray(color_frame.get_data())
            img_realsense_vis = detect_charuco_in_image_live(img_realsense.copy(), board, detector)
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meters_realsense = depth_image.astype(np.float32) * depth_scale
            depth_realsense_colored = colorize_depth(depth_meters_realsense, cv2.COLORMAP_TURBO)

        img_zed = None
        img_zed_vis = None
        depth_zed = None
        depth_zed_colored = None

        if use_zed:
            zed_img_mat = sl.Mat()
            zed_depth_mat = sl.Mat()

            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                print("Failed to capture ZED frames")
                return False

            zed.retrieve_image(zed_img_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(zed_depth_mat, sl.MEASURE.DEPTH)

            img_zed = zed_img_mat.get_data()
            img_zed = cv2.cvtColor(img_zed, cv2.COLOR_BGRA2BGR)
            img_zed_vis = detect_charuco_in_image_live(img_zed.copy(), board, detector)

            depth_zed = zed_depth_mat.get_data().copy().astype(np.float32)
            depth_zed_colored = colorize_depth(depth_zed, cv2.COLORMAP_TURBO)

        last["img_l_raw"] = img_l_raw
        last["img_r_raw"] = img_r_raw
        last["img_l_disp"] = img_l
        last["img_r_disp"] = img_r
        last["img_realsense"] = img_realsense
        last["img_realsense_vis"] = img_realsense_vis
        last["depth_realsense"] = depth_meters_realsense
        last["depth_realsense_colored"] = depth_realsense_colored
        last["img_zed"] = img_zed
        last["img_zed_vis"] = img_zed_vis
        last["depth_zed"] = depth_zed
        last["depth_zed_colored"] = depth_zed_colored
        return True

    def show_preview():
        if last["img_l_disp"] is None or last["img_r_disp"] is None:
            blank = np.zeros((frame_size_display[1], frame_size_display[0], 3), dtype=np.uint8)
            cv2.imshow("Captured Frames", blank)
            return

        top_left = _make_labeled_tile(last["img_l_disp"], "Stereo Left", frame_size_display)
        top_right = _make_labeled_tile(last["img_r_disp"], "Stereo Right", frame_size_display)

        mid_left = _make_labeled_tile(last["img_realsense_vis"], "RealSense RGB", frame_size_display)
        mid_right = _make_labeled_tile(
            last["depth_realsense_colored"] if use_realsense else None,
            "RealSense Depth",
            frame_size_display
        )

        bottom_left = _make_labeled_tile(last["img_zed_vis"], "ZED RGB", frame_size_display)
        bottom_right = _make_labeled_tile(
            last["depth_zed_colored"] if use_zed else None,
            "ZED Depth",
            frame_size_display
        )

        row1 = cv2.hconcat([top_left, top_right])

        # concat_rows = [row1]
        concat_rows = []
        if use_realsense:
            row2 = cv2.hconcat([mid_left, mid_right])
            concat_rows.append(row2)
        if use_zed:
            row3 = cv2.hconcat([bottom_left, bottom_right])
            concat_rows.append(row3)

        preview = cv2.vconcat(concat_rows)
        cv2.imshow("Captured Frames", preview)

    # capture first frame so there is something to display/save
    capture_current_frames()
    show_preview()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord("q"):
            break

        elif key == 32:  # SPACE
            if capture_current_frames():
                show_preview()

        elif key == ord('s'):
            if last["img_l_raw"] is None or last["img_r_raw"] is None:
                print("No captured frame to save. Press SPACE first.")
                continue

            # cv2.imwrite(os.path.join(rgb_dir, f"{num}_left.png"), last["img_l_raw"])
            # cv2.imwrite(os.path.join(rgb_dir, f"{num}_right.png"), last["img_r_raw"])

            if use_realsense and last["img_realsense"] is not None and last["depth_realsense"] is not None:
                cv2.imwrite(os.path.join(rgb_dir, f"{num}_realsense.png"), last["img_realsense"])
                np.save(os.path.join(depth_dir, f"{num}_realsense_depth.npy"), last["depth_realsense"])

            if use_zed and last["img_zed"] is not None and last["depth_zed"] is not None:
                cv2.imwrite(os.path.join(rgb_dir, f"{num}_zed.png"), last["img_zed"])
                np.save(os.path.join(depth_dir, f"{num}_zed_depth.npy"), last["depth_zed"])

            print(f"Saved displayed capture {num}")
            num += 1

    cap_l.release()
    cap_r.release()

    if use_realsense:
        pipeline.stop()
    if use_zed:
        zed.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    chessboard_size = (8,5)
    parent_dir = Path(__file__).resolve().parent.parent

    # calib_dir = load_dict(out_dir + "/calib_data.npy")
    dataset_dir = os.path.join(parent_dir, 'dataset_24032026')
    depth_dir = os.path.join(dataset_dir, 'stereo_4k_relative_pose')

    out_dir = parent_dir / "out" / "cameras_parameters"
    # calib_dict = load_dict(out_dir / "calib_data.npy")

    save_cameras_on_click(3,3, frame_size_stereo=frame_size_4K, save_dir=depth_dir, use_realsense=True, use_zed=True, calib_dict=None,
                          squares_horizontally=6,
                          squares_vertically=8,
                          squares_length=44.0,
                          marker_length=30.0
                            )

    # #show_undistor   ed(depth_dir + "/rgb", calib_dir)
    # show_realsense_image_live()
    # show_zed_image()
