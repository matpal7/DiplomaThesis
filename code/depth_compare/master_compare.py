from pathlib import Path

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[3]
    date = "27032026"
    nn_name = "DepthAnything3_stereo"
    # NNname = "FoundationStereo"
    debug = 0
    rgbd_camera_suffix = "realsense"
