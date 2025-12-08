def show_cameras_on_click(device_indices, frame_size=(640, 480)):
    """
    Captures and displays frames from multiple cameras (and optionally RealSense) when a key is pressed.

    Args:
        device_indices (list[int]): List of OpenCV camera indices to capture from.
        frame_size (tuple): Size (width, height) to which each camera image will be resized.
        use_realsense (bool): Whether to also capture from an Intel RealSense camera (RGB + depth).
    """
    # Open regular cameras
    caps = []
    for idx in device_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        caps.append(cap)


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("Press SPACE to capture and display frames. ESC to exit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC to exit
            break
        else:  # SPACE to capture
            frames = []

            # Capture from regular cameras
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    print("Could not read frame from one or more cameras")
                    break
                frames.append(cv2.resize(frame, frame_size))

            # Capture from RealSense if enabled

            rs_frames = pipeline.wait_for_frames()
            rs_frames = align.process(rs_frames)
            color_frame = rs_frames.get_color_frame()
            depth_frame = rs_frames.get_depth_frame()
            if color_frame and depth_frame:
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colored = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                frames.append(cv2.resize(color_image, frame_size))
                frames.append(cv2.resize(depth_colored, frame_size))
            else:
                print("Failed to capture RealSense frames")

            if len(frames) == len(caps) + (2):
                img_concat = cv2.hconcat(frames)
                cv2.imshow('Captured Frames', img_concat)

    # Release all resources
    for cap in caps:
        cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()