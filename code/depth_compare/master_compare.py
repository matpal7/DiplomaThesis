from pathlib import Path

from code.depth_compare.compare_depth_between_cameras import run_depth_comparison_experiment

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[3]
    date = "27032026"
    nn_name = "DepthAnything3_stereo"
    # NNname = "FoundationStereo"
    debug = 0
    rgbd_camera_suffix = "zed"
    max_imgs = 7
    run_depth_comparison_experiment(
        parent_dir,
        date=date,
        nn_name=nn_name,
        rgbd_camera_suffix=rgbd_camera_suffix,
        max_imgs=max_imgs
    )

