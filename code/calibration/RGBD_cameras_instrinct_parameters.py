from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs

from get_images import FRAME_SIZE_REALSENSE, FRAME_SIZE_ZED


def save_zed_calibration(out_dir, filename="zed_left_calibration_1280x720_factory.yaml"):
    zed = sl.Camera()

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    status = zed.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {status}")

    try:
        cam_info = zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        left = calib.left_cam

        # image size
        resolution = cam_info.camera_configuration.resolution
        image_width = int(resolution.width)
        image_height = int(resolution.height)

        K = np.array([
            [left.fx, 0, left.cx],
            [0, left.fy, left.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Stereolabs distortion: k1, k2, k3, p1, p2
        # OpenCV: k1, k2, p1, p2, k3
        disto = left.disto
        D = np.array([
            disto[0],  # k1
            disto[1],  # k2
            disto[3],  # p1
            disto[4],  # p2
            disto[2],  # k3
        ], dtype=np.float64)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
        fs.write("K", K)
        fs.write("D", D)
        fs.write("image_width", image_width)
        fs.write("image_height", image_height)
        fs.release()

        print("Calibration saved to", out_path)

    finally:
        zed.close()


def save_realsense_calibration(out_dir, filename="realsense_calibration_1280x720_factory.yaml"):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        FRAME_SIZE_REALSENSE[0],
        FRAME_SIZE_REALSENSE[1],
        rs.format.bgr8,
        30
    )

    profile = pipeline.start(config)

    try:
        stream = profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()

        image_width = int(intr.width)
        image_height = int(intr.height)

        K = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

        D = np.array(intr.coeffs, dtype=np.float64)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
        fs.write("K", K)
        fs.write("D", D)
        fs.write("image_width", image_width)
        fs.write("image_height", image_height)
        fs.release()

        print("Calibration saved to", out_path)

    finally:
        pipeline.stop()

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[3]
    date = "11042026"
    out_dir = parent_dir / "out" / f"out_{date}" / "cameras_parameters"
    print(f"out_dir: {out_dir}")

    save_zed_calibration(out_dir)
    save_realsense_calibration(out_dir)