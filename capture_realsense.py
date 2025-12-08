import pyrealsense2 as rs
import numpy as np
import cv2

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream (Full HD)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Enable left and right infrared cameras (Stereo)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

# Enable depth stream
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

print("Press 's' to save images, 'q' to quit.")
image_counter = 0
try:
    while True:
        # Get frames from all streams
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        left_ir_frame = frames.get_infrared_frame(1)
        right_ir_frame = frames.get_infrared_frame(2)
        depth_frame = frames.get_depth_frame()

        if not color_frame or not left_ir_frame or not right_ir_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Resize color image to match infrared (1280x720)
        color_image_resized = cv2.resize(color_image, (1280, 720))

        # Convert depth to an 8-bit image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack images side-by-side (Left IR | Right IR | Color | Depth)
        stacked = np.hstack((
            color_image_resized,
            depth_colormap
        ))

        # Display images
        cv2.imshow("Left | Right | Color | Depth", stacked)

        # Save images on 's' key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save images
            cv2.imwrite(f"color_image_{image_counter}.png", color_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(f"depth_image_{image_counter}.png", depth_colormap, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            np.save(f"depth_data_{image_counter}.npy", depth_image)  # Save raw depth data
            image_counter += 1

            print("Images saved: color_image.png, depth_image.png")
            print("Raw depth data saved as depth_data.npy")

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
