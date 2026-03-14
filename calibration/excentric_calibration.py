from pathlib import Path
import cv2
import numpy as np
import glob


def prepare_object_points(chessboard_x, chessboard_y, square_size):
    objp = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)
    objp *= square_size
    return objp


def load_images(folder, pattern="*.png"):
    images = glob.glob(str(Path(folder) / pattern))
    print(f"Found {len(images)} images in {folder}.")
    return images


def extract_chessboard_points(images, chessboard_x, chessboard_y, square_size, show=False):
    objpoints = []
    imgpoints = []

    objp = prepare_object_points(chessboard_x, chessboard_y, square_size)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    img_shape = None

    for fname in images:

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            (chessboard_x, chessboard_y),
            None
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )
            imgpoints.append(corners2)

            if show:
                cv2.drawChessboardCorners(img,
                                          (chessboard_x, chessboard_y),
                                          corners2,
                                          ret)

                cv2.imshow("corners", img)
                key = cv2.waitKey(100)

                if key == 27:
                    break

    cv2.destroyAllWindows()

    return objpoints, imgpoints, img_shape


def calibrate_camera(objpoints, imgpoints, img_shape):

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_shape,
        None,
        None
    )

    calib = {
        "K": K,
        "D": D,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "img_dim": img_shape
    }

    return calib, ret


def show_undistored(
        img_folder,
        calib,
        pattern="*.png",
        show=False):

    images = load_images(img_folder, pattern)

    for fname in images:

        img = cv2.imread(fname)
        und = undistort_image(img, calib["K"], calib["D"])

        concat = cv2.hconcat([img, und])

        cv2.imshow("Distorted | Undistorted", concat)
        key = cv2.waitKey(0)
        if key == 27:
            break


def load_camera_calibration(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)

    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()

    fs.release()

    calib = {
        "K": K,
        "D": D
    }

    return calib

def undistort_image(img, K, D):

    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
    undistorted = cv2.undistort(img, K, D, None, newK)

    return undistorted

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parent.parent
    dataset_dir = parent_dir / "dataset_05032026"

    rgb_folder = dataset_dir / "depth" / "rgb"

    calib_file = parent_dir / "calibration" / "realsense_calibration_factory.yaml"
    calib = load_camera_calibration(calib_file)

    show_undistored(
        rgb_folder,
        calib,
        pattern="*_realsense.png")