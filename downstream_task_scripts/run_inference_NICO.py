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

def load_camera_calibration(path: Path,
                             suffix: str = "left") -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    with open(path, "r") as f:
        calib = json.load(f)
    key   = "new_K_l" if suffix == "left" else "new_K_r"
    K     = np.asarray(calib[key], dtype=np.float64).reshape(3, 3)
    D_key = "D_l" if suffix == "left" else "D_r"
    D     = np.asarray(calib[D_key], dtype=np.float64).reshape(-1, 1) \
            if D_key in calib else np.zeros((5, 1), dtype=np.float64)
    return K, D


def load_sorted_png_files(directory: Path, camera: str, rgb: bool = True) -> List[Path]:
    """
    Return PNG files from <directory>/{rgb|depth_vis}/*_<camera>.png
    sorted by the leading numeric index in the filename stem.
    """
    folder = "rgb" if rgb else "depth_vis"
    files = list((directory / folder).glob(f"*_{camera}.png"))
    return sorted(files, key=lambda p: int(p.stem.split("_")[0]))


# ─── Image processing ─────────────────────────────────────────────────────────

def downscale_frame(
    color: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Downscale an RGB-D frame and adjust the camera intrinsic matrix.

    Nearest-neighbour interpolation is used for depth and mask to preserve
    discrete values; INTER_AREA is used for the colour image.

    Args:
        color : HxWx3 uint8 RGB image.
        depth : HxW float32 depth image (metres).
        K     : 3×3 camera intrinsic matrix for the original resolution.
        scale : Downscale factor in (0, 1].  1.0 → no change.
        mask  : Optional HxW uint8 binary mask.

    Returns:
        color_ds, depth_ds, K_ds, mask_ds  (mask_ds is None if mask is None)
    """
    if scale == 1.0:
        return color, depth, K.copy(), mask

    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    h, w = color.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    new_size = (new_w, new_h)  # cv2 uses (width, height)

    color_ds = cv2.resize(color, new_size, interpolation=cv2.INTER_AREA)
    depth_ds = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)

    mask_ds = None
    if mask is not None:
        mask_ds = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

    # Scale intrinsics: fx, fy, cx, cy all scale linearly with resolution
    K_ds = K.copy().astype(np.float64)
    K_ds[0, 0] *= scale   # fx
    K_ds[1, 1] *= scale   # fy
    K_ds[0, 2] *= scale   # cx
    K_ds[1, 2] *= scale   # cy

    return color_ds, depth_ds, K_ds, mask_ds


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

def load_sorted_files(directory: Path,
                      pattern: str = "*.png") -> List[Path]:
    print(directory)
    files = list((directory).glob(pattern))
    return sorted(files, key=lambda p: int(p.stem.split("_")[0]))

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_refine_iter",   type=int,   default=5)
    parser.add_argument("--track_refine_iter", type=int,   default=2)
    parser.add_argument("--debug",             type=int,   default=0)
    parser.add_argument("--downscale",         type=float, default=0.15)
    parser.add_argument("--frame_step",        type=int,   default=1)
    parser.add_argument("--save_vis",        type=bool,   default=True)

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # ── Paths ──────────────────────────────────────────────────────────────────
    code_dir       = Path(__file__).resolve().parent
    base_dir       = code_dir.parent
    date           = "24042026"
    out_dir = base_dir / "out" / f"out_{date}"
    out_dir_pose_estimation = out_dir / "pose_estimation"
    scene_dir_main = out_dir_pose_estimation/ "undistorted_images_NICO"
    depth_dir_main    = out_dir_pose_estimation / "depth_nn"

    masks_dir      = out_dir_pose_estimation / "masks"
    models_dir     = out_dir_pose_estimation / "3D_models"
    camera         = "left"
    calib_left = out_dir / "cameras_parameters" / "calib_data.json"



    # ── Camera intrinsics ──────────────────────────────────────────────────────
    K_full, _ = load_camera_calibration(calib_left)
    logging.info("K (full-res):\n%s", K_full)

    # ── Scenes & objects ───────────────────────────────────────────────────────
    scenes = sorted(
        [d.name for d in scene_dir_main.iterdir() if d.is_dir()],
        key=lambda x: int(x.split("_")[-1]),
    )
    objects_name = ["chips_box", "apple", "lemon", "orange", "rubiks_cube", "scissors"] #, "wood_block"]
    # objects_name = ["scissors"]

    # ── Per-scene loop ─────────────────────────────────────────────────────────
    for scene in scenes[5:]:
        scene_number = scene.split("_")[-1]
        scene_dir   = scene_dir_main / scene
        masks_scene = masks_dir / scene

        color_paths = load_sorted_files(scene_dir,
                                        pattern=f"*_{camera}.png")
        depth_paths = load_sorted_files(depth_dir_main / scene / "depth_png",
                                        pattern=f"*.png")
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

            debug_dir = out_dir_pose_estimation / "results" / "left" / scene / object_name
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

                color, depth, K, mask = downscale_frame(
                    color_full, depth_full, K_full,
                    scale=args.downscale,
                    mask=mask_full if i == 0 else None,
                )

                # print(color.shape)
                # print(depth.shape)
                # print(K.shape)
                # print(mask.shape)


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