import cv2

# ------------------------------
# BOARD PARAMETERS
ARUCO_DICT = cv2.aruco.DICT_4X4_250
SQUARES_HORIZONTALLY = 5
SQUARES_VERTICALLY = 7
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.024
CAMERA_INDEX = 0
# ------------------------------

def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )

    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

        if marker_ids is not None and len(marker_ids) > 0:
            # cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board
            )

            cv2.putText(
                vis,
                f"Markers: {len(marker_ids)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if retval is not None and retval > 0 and charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

                cv2.putText(
                    vis,
                    f"ChArUco corners: {len(charuco_ids)}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    vis,
                    "No ChArUco corners",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
        else:
            cv2.putText(
                vis,
                "No markers detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Live ChArUco Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()