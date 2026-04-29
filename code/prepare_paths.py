from pathlib import Path


DEPTH_ESTIMATION_NETWORK_NAMES = [
    "UniDepth_mono",
    "MoGe_mono",
    "DepthAnything3_mono",
    "DEFOM_stereo",
    "FoundationStereo",
    "S2M2_stereo",
]
def build_paths(root_dir:Path, date: str, model_name: str) -> tuple[Path, Path, Path]:
    img_dir = root_dir / "datasets" / f"dataset_{date}" / f"stereo_4k_depth" / "rgb"
    calib_dict_file = root_dir / "out" / f"out_{date}" / "cameras_parameters" / "calib_data.npy"
    out_dir = root_dir / "out" / f"out_{date}" / f"depth_estimation" / model_name
    return img_dir, calib_dict_file, out_dir


def prepare_output_dirs(out_dir: Path, disp_dir: bool = False) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "depth": out_dir / "depth",
        "vis": out_dir / "vis",
    }

    if disp_dir:
        dirs["disp"] = out_dir / "disp"

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def prepare_relative_pose_paths(root_dir:Path, date: str, rgbd_cam_suffix: str, use_factory: bool=False):
    dataset_dir = root_dir / "datasets" / f'dataset_{date}'
    relative_pose_dir = dataset_dir / 'stereo_4k_relative_pose' / 'rgb'
    out_dir = root_dir / "out" / f"out_{date}" / "cameras_parameters"
    out_dir_save = out_dir / "relative_pose"
    calib_dict_stereo = out_dir / "calib_data.npy"
    cam_suffix = "_factory" if use_factory else ""
    calib_dict_RGBD_cam = out_dir / f"{rgbd_cam_suffix}_calibration_1280x720{cam_suffix}.yaml"
    return dataset_dir, relative_pose_dir, out_dir, out_dir_save, calib_dict_stereo, calib_dict_RGBD_cam

from pathlib import Path


def prepare_depth_comparison_paths(
    parent_dir: Path,
    date: str,
    rgbd_camera_suffix: str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    dataset_dir = parent_dir / "datasets" / f"dataset_{date}"
    out_root = parent_dir / "out" / f"out_{date}"

    camera_params_dir = out_root / "cameras_parameters"
    depth_estimation_dir = out_root / "depth_estimation"
    depth_comparison_dir = out_root / "depth_comparison" / rgbd_camera_suffix

    gt_data_dir = dataset_dir / "stereo_4k_depth"
    relative_pose_path = (
        camera_params_dir
        / "relative_pose"
        / f"relative_pose_{rgbd_camera_suffix}_to_left_v3.yaml"
    )
    calib_rgbd_path = camera_params_dir / f"{rgbd_camera_suffix}_calibration_1280x720.yaml"
    calib_stereo_path = camera_params_dir / "calib_data.npy"

    return (
        gt_data_dir,
        relative_pose_path,
        calib_rgbd_path,
        calib_stereo_path,
        depth_estimation_dir,
        depth_comparison_dir,
    )

def get_depth_estimation_network_names():
    return DEPTH_ESTIMATION_NETWORK_NAMES.copy()