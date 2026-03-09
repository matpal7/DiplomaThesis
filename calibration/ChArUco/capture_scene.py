import os
import cv2

def capture_scene_from_two_cameras(
    left_camera_index=0,
    right_camera_index=1,
    save_dir="captured_scenes",
    frame_width=720,
    frame_height=480
):
    os.makedirs(save_dir, exist_ok=True)

    cap_left = cv2.VideoCapture(left_camera_index)
    cap_right = cv2.VideoCapture(right_camera_index)

    if not cap_left.isOpened():
        raise RuntimeError(f"Cannot open left camera with index {left_camera_index}")
    if not cap_right.isOpened():
        raise RuntimeError(f"Cannot open right camera with index {right_camera_index}")

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print("Press 'S' to save images")
    print("Press 'Q' to quit")

    img_id = 0

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Failed to read from one or both cameras")
            break

        left_preview = cv2.resize(frame_left, (frame_width, frame_height))
        right_preview = cv2.resize(frame_right, (frame_width, frame_height))
        combined = cv2.hconcat([left_preview, right_preview])

        cv2.imshow("Left | Right", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            left_path = os.path.join(save_dir, f"{img_id}_left.png")
            right_path = os.path.join(save_dir, f"{img_id}_right.png")

            cv2.imwrite(left_path, frame_left)
            cv2.imwrite(right_path, frame_right)

            print(f"Saved: {left_path}, {right_path}")
            img_id += 1

        elif key == ord("q"):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_scene_from_two_cameras(left_camera_index=0, right_camera_index=1)