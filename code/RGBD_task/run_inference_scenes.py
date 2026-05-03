import argparse
import numpy as np
import logging
import os
# import imageio
# import trimesh
# import nvdiffrast.torch as dr
from pathlib import Path
from typing import List, Tuple, Optional
import cv2


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


def load_sorted_png_files(directory: Path, folder: str, pattern: str = "*.png") -> List[Path]:
    """
    Return PNG files from <directory>/{rgb|depth_vis}/*_<camera>.png
    sorted by the leading numeric index in the filename stem.
    """
    files = list((directory / folder).glob(pattern))
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

    If depth resolution differs from RGB resolution,
    depth is resized to match RGB using nearest-neighbour interpolation.
    """

    # Load RGB image
    color = cv2.cvtColor(
        cv2.imread(str(color_path)),
        cv2.COLOR_BGR2RGB
    )

    # Load depth image
    depth = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED
    ).astype(np.float32) / 1000.0

    # Check resolution mismatch
    h_color, w_color = color.shape[:2]
    h_depth, w_depth = depth.shape[:2]

    if (h_color, w_color) != (h_depth, w_depth):
        logging.warning(
            "Resolution mismatch: RGB=%dx%d, Depth=%dx%d. Resizing depth.",
            w_color, h_color,
            w_depth, h_depth,
        )

        depth = cv2.resize(
            depth,
            (w_color, h_color),
            interpolation=cv2.INTER_NEAREST
        )

    return color, depth


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_refine_iter",   type=int,   default=5)
    parser.add_argument("--track_refine_iter", type=int,   default=2)
    parser.add_argument("--debug",             type=int,   default=0)
    parser.add_argument("--downscale",         type=float, default=0.5)
    parser.add_argument("--frame_step",        type=int,   default=1)
    parser.add_argument("--save_vis",        type=bool,   default=True)
    parser.add_argument("--use_depth_from_rgbd",        type=bool,   default=False)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Program started")

    # set_logging_format()
    # set_seed(0)



    # ── Paths ──────────────────────────────────────────────────────────────────
    camera         = "realsense"
    # code_dir       = Path(__file__).resolve().parent
    code_dir = Path(__file__).resolve().parents[2]
    base_dir       = code_dir.parent
    date           = "24042026"
    scene_dir_main = base_dir / "datasets" / f"dataset_{date}" / "downstream_task"
    depth_dir_main = base_dir / 'out' / f"out_{date}" / "pose_estimation_depths"
    pattern_depth = f"*_{camera}.png"
    folder_depth = "depth_vis/rgbd"
    if not args.use_depth_from_rgbd:
        pattern_depth = f"*.png"
        folder_depth = "depth_vis/nn"

    out_dir        = base_dir / "out" / f"out_{date}"
    masks_dir      = out_dir / "masks"
    models_dir     = out_dir / "3D_models"
    calib_realsense = out_dir / "cameras_parameters" / f"{camera}_calibration_1280x720.yaml"

    # ── Camera intrinsics ──────────────────────────────────────────────────────
    K_full = load_k_matrix(str(calib_realsense))
    logging.info("K (full-res):\n%s", K_full)

    # ── Scenes & objects ───────────────────────────────────────────────────────
    scenes = sorted(
        [d.name for d in scene_dir_main.iterdir() if d.is_dir()],
        key=lambda x: int(x.split("_")[-1]),
    )
    objects_name = ["chips_box", "apple", "lemon", "orange", "rubiks_cube", "scissors", "wood_block"]

    # ── Per-scene loop ─────────────────────────────────────────────────────────
    for scene in scenes:
        scene_number = scene.split("_")[-1]
        scene_dir   = scene_dir_main / scene
        masks_scene = masks_dir / scene

        color_paths = load_sorted_png_files(scene_dir, "rgb", pattern=f"*_{camera}.png")
        depth_paths = load_sorted_png_files(depth_dir_main / scene, folder_depth, pattern=pattern_depth)
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

            # debug_dir = out_dir / "pose_estimation" / scene / object_name
            # os.makedirs(str(debug_dir / "track_vis"), exist_ok=True)
            # os.makedirs(str(debug_dir / "ob_in_cam"), exist_ok=True)
            #
            # # ── Mesh & pose bounding box ───────────────────────────────────────
            # mesh               = trimesh.load(str(mesh_file))
            # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            # bbox               = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            #
            # # ── Estimator init ─────────────────────────────────────────────────
            # scorer  = ScorePredictor()
            # refiner = PoseRefinePredictor()
            # glctx   = dr.RasterizeCudaContext()
            # est     = FoundationPose(
            #     model_pts=mesh.vertices,
            #     model_normals=mesh.vertex_normals,
            #     mesh=mesh,
            #     scorer=scorer,
            #     refiner=refiner,
            #     debug_dir=str(debug_dir),
            #     debug=args.debug,
            #     glctx=glctx,
            # )
            # logging.info("Estimator initialised")

            # ── Frame loop ─────────────────────────────────────────────────────
            for i in range(0, len(color_paths), args.frame_step):
                logging.info("Frame %d/%d", i, len(color_paths) - 1)

                color_full, depth_full = load_frame(color_paths[i], depth_paths[i])

                color, depth, K, mask = downscale_frame(
                    color_full, depth_full, K_full,
                    scale=args.downscale,
                    mask=mask_full if i == 0 else None,
                )

                # ── Visualization ─────────────────────────────────────────────
                #
                # # RGB preview
                # rgb_vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                #
                # # Depth preview
                # depth_vis = depth.copy()
                #
                # # Replace invalid values
                # depth_vis[~np.isfinite(depth_vis)] = 0
                #
                # # Normalize to 0-255
                # if np.max(depth_vis) > 0:
                #     depth_vis = depth_vis / np.max(depth_vis)
                #
                # depth_vis = (depth_vis * 255).astype(np.uint8)
                #
                # # Apply colormap
                # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                #
                # # Mask preview
                # if mask is not None:
                #     mask_vis = (mask > 0).astype(np.uint8) * 255
                # else:
                #     mask_vis = np.zeros(depth.shape, dtype=np.uint8)
                #
                # # Show windows
                # cv2.imshow("RGB", rgb_vis)
                # cv2.imshow("Depth", depth_vis)
                # cv2.imshow("Mask", mask_vis)
                #
                # key = cv2.waitKey(0)


                # if i == 0:
                #     pose = est.register(
                #         K=K, rgb=color, depth=depth,
                #         ob_mask=mask, iteration=args.est_refine_iter,
                #     )
                # else:
                #     pose = est.track_one(
                #         rgb=color, depth=depth,
                #         K=K, iteration=args.track_refine_iter,
                #     )

        #         # Save pose matrix
        #         pose_out = debug_dir / "ob_in_cam" / "{:03d}.txt".format(i)
        #         np.savetxt(str(pose_out), pose.reshape(4, 4))
        #
        #         # Visualisation
        #         if args.save_vis:
        #             center_pose = pose @ np.linalg.inv(to_origin)
        #             vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #             vis = draw_xyz_axis(
        #                 vis, ob_in_cam=center_pose,
        #                 scale=0.1, K=K, thickness=3,
        #                 transparency=0, is_input_rgb=True,
        #             )
        #
        #             out_img = debug_dir / "track_vis" / "{:03d}.png".format(i)
        #             imageio.imwrite(str(out_img), vis)