from estimater import *
from datareader import *
import argparse
import numpy as np
import logging
import os
import imageio
import trimesh
import nvdiffrast.torch as dr
from pathlib import Path
from typing import List, Tuple, Optional


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_k_matrix(yaml_path: str, node: str = "K") -> np.ndarray:
    """Load a 3×3 camera intrinsic matrix from an OpenCV YAML calibration file."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {yaml_path}")
    K = fs.getNode(node).mat()
    fs.release()
    if K is None:
        raise ValueError(f"Node '{node}' not found in {yaml_path}")
    return K


def load_sorted_png_files(directory: Path, camera: str, rgb: bool = True) -> List[Path]:
    """
    Return PNG files from <directory>/{rgb|depth_vis}/*_<camera>.png
    sorted by the leading numeric index in the filename stem.
    """
    folder = "rgb" if rgb else "depth_vis"
    files = list((directory / folder).glob(f"*_{camera}.png"))
    return sorted(files, key=lambda p: int(p.stem.split("_")[0]))


# ─── Image processing ─────────────────────────────────────────────────────────

def resize_frame(
    color: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    target_size: Tuple[int, int],   # (width, height)
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Resize an RGB-D frame to a fixed resolution and adjust intrinsics.

    Args:
        color : HxWx3 uint8 RGB image
        depth : HxW float32 depth image
        K     : 3x3 camera intrinsic matrix
        target_size : (width, height)
        mask  : Optional mask

    Returns:
        resized_color, resized_depth, resized_K, resized_mask
    """

    target_w, target_h = target_size

    h, w = color.shape[:2]

    scale_x = target_w / w
    scale_y = target_h / h

    color_rs = cv2.resize(
        color,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    depth_rs = cv2.resize(
        depth,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_rs = None
    if mask is not None:
        mask_rs = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    # Adjust intrinsics
    K_rs = K.copy().astype(np.float64)

    K_rs[0, 0] *= scale_x   # fx
    K_rs[1, 1] *= scale_y   # fy
    K_rs[0, 2] *= scale_x   # cx
    K_rs[1, 2] *= scale_y   # cy

    return color_rs, depth_rs, K_rs, mask_rs


def load_frame(color_path: Path, depth_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one RGB-D pair.
    Returns:
        color : HxWx3 uint8 RGB
        depth : HxW float32 in metres
    """
    color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return color, depth

def load_sorted_files(directory: Path, folder: str,
                      pattern: str = "*.png") -> List[Path]:
    print(directory / folder)
    files = list((directory / folder).glob(pattern))
    return sorted(files, key=lambda p: int(p.stem.split("_")[0]))

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_refine_iter",   type=int,   default=5)
    parser.add_argument("--track_refine_iter", type=int,   default=2)
    parser.add_argument("--debug",             type=int,   default=0)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=324)
    parser.add_argument("--frame_step",        type=int,   default=1)
    parser.add_argument("--save_vis",        type=bool,   default=True)

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # ── Paths ──────────────────────────────────────────────────────────────────
    code_dir       = Path(__file__).resolve().parent
    base_dir       = code_dir.parent
    date           = "24042026"
    scene_dir_main = base_dir / "datasets" / f"dataset_{date}" / "downstream_task"
    depth_dir_main    = base_dir / "out" / f"out_{date}" / "pose_estimation"

    out_dir        = base_dir / "out" / f"out_{date}"

    masks_dir      = depth_dir_main / "masks"
    models_dir     = depth_dir_main / "3D_models"
    camera         = "zed"
    calib_realsense = out_dir / "cameras_parameters" / f"{camera}_calibration_1280x720.yaml"

    folder_depth  = "depth_png"


    # ── Camera intrinsics ──────────────────────────────────────────────────────
    K_full = load_k_matrix(str(calib_realsense))
    logging.info("K (full-res):\n%s", K_full)

    # ── Scenes & objects ───────────────────────────────────────────────────────
    scenes = sorted(
        [d.name for d in scene_dir_main.iterdir() if d.is_dir()],
        key=lambda x: int(x.split("_")[-1]),
    )
    objects_name = ["chips_box", "apple", "lemon", "orange", "rubiks_cube", "scissors"] #, "wood_block"]
    # objects_name = ["wood_block"]
    # ── Per-scene loop ─────────────────────────────────────────────────────────
    for scene in scenes[5:]:
        scene_number = scene.split("_")[-1]
        scene_dir   = scene_dir_main / scene
        masks_scene = masks_dir / scene

        color_paths = load_sorted_files(scene_dir, "rgb",
                                        pattern=f"*_{camera}.png")
        depth_paths = load_sorted_files(depth_dir_main / "depth_rgbd" / scene, folder_depth,
                                        pattern=f"*_{camera}.png")
        assert len(color_paths) == len(depth_paths), (
            "Colour/depth count mismatch: {} vs {}".format(len(color_paths), len(depth_paths))
        )
        logging.info("Scene '%s': %d frames", scene, len(color_paths))

        objects_current = sorted(
            [
                folder.name
                for folder in masks_scene.iterdir()
                if folder.is_dir()
            ],
        )

        # ── Per-object loop ────────────────────────────────────────────────────
        for object_name in objects_name:
            if object_name not in objects_current:
                continue
            mesh_file = models_dir / "{}_textured.obj".format(object_name)
            mask_path = masks_scene / object_name / f"000_{camera}.png"

            mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_full is None:
                raise FileNotFoundError("Mask not found: {}".format(mask_path))

            debug_dir = out_dir / "pose_estimation" / "results" / camera / scene / object_name
            os.makedirs(str(debug_dir / "track_vis"), exist_ok=True)
            os.makedirs(str(debug_dir / "ob_in_cam"), exist_ok=True)

            # ── Mesh & pose bounding box ───────────────────────────────────────
            mesh               = trimesh.load(str(mesh_file))
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox               = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

            # ── Estimator init ─────────────────────────────────────────────────
            scorer  = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx   = dr.RasterizeCudaContext()
            est     = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=str(debug_dir),
                debug=args.debug,
                glctx=glctx,
            )
            logging.info("Estimator initialised")

            # ── Frame loop ─────────────────────────────────────────────────────
            for i in range(0, len(color_paths), args.frame_step):
                logging.info("Frame %d/%d", i, len(color_paths) - 1)

                color_full, depth_full = load_frame(color_paths[i], depth_paths[i])

                color, depth, K, mask = resize_frame(
                    color_full,
                    depth_full,
                    K_full,
                    target_size=(args.width, args.height),
                    mask=mask_full if i == 0 else None,
                )

                if i == 0:
                    pose = est.register(
                        K=K, rgb=color, depth=depth,
                        ob_mask=mask, iteration=args.est_refine_iter,
                    )
                else:
                    pose = est.track_one(
                        rgb=color, depth=depth,
                        K=K, iteration=args.track_refine_iter,
                    )

                # Save pose matrix
                pose_out = debug_dir / "ob_in_cam" / "{:03d}.txt".format(i)
                np.savetxt(str(pose_out), pose.reshape(4, 4))

                # Visualisation
                if args.save_vis:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(
                        vis, ob_in_cam=center_pose,
                        scale=0.1, K=K, thickness=3,
                        transparency=0, is_input_rgb=True,
                    )

                    out_img = debug_dir / "track_vis" / "{:03d}.png".format(i)
                    imageio.imwrite(str(out_img), vis)