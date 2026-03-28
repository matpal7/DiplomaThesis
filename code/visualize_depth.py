from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt

from code.image import load_rgb_depth_pairs


def visualize_depth(
    depth: np.ndarray,
    vmin: float = None,
    vmax: float = None,
    show: bool = True,
    save_path: str | None = None,
    cmap: str = "turbo",
):
    """
    Visualize depth map using matplotlib with colorbar.

    Parameters
    ----------
    depth : np.ndarray
        2D depth map
    vmin : float | None
    vmax : float | None
    show : bool
    save_path : str | None
    cmap : str
        matplotlib colormap (e.g. 'turbo', 'viridis', 'plasma')
    """

    if depth.ndim != 2:
        raise ValueError("Depth map must be 2D")

    depth = depth.astype(np.float32)
    depth = depth * 0.001
    valid_mask = np.isfinite(depth)

    if not np.any(valid_mask):
        raise ValueError("No valid depth values")

    if vmin is None:
        vmin = float(np.percentile(depth[valid_mask], 2))

    if vmax is None:
        vmax = float(np.percentile(depth[valid_mask], 98))

    print("min:", np.percentile(depth[valid_mask], 2))
    print("max:", np.percentile(depth[valid_mask], 98))
    print("mean:", depth[valid_mask].mean())

    # mask invalid values
    depth_vis = depth.copy()
    depth_vis[~valid_mask] = np.nan

    plt.figure(figsize=(8, 6))
    im = plt.imshow(depth_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Depth [m]")

    plt.title("Depth visualization")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def colorize_depth(depth_m: np.ndarray, color_map=cv2.COLORMAP_TURBO) -> np.ndarray:
    valid_mask = np.isfinite(depth_m) & (depth_m > 0)

    if not np.any(valid_mask):
        h, w = depth_m.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    vmin = np.percentile(depth_m[valid_mask], 2)
    vmax = np.percentile(depth_m[valid_mask], 98)

    depth_norm = (depth_m - vmin) / (vmax - vmin + 1e-8)
    depth_norm[~valid_mask] = 0
    depth_norm = np.clip(depth_norm, 0, 1)

    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_u8, color_map)
    depth_colored[~valid_mask] = 0
    return depth_colored

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents
    dataset_dir = parent_dir / "dataset_17032026"
    calib_imgs_dir = dataset_dir / "stereo_4k_calibration" / "rgb"
    relative_pose_dir = dataset_dir / "stereo_4k_depth"
    imgs, depths = load_rgb_depth_pairs(relative_pose_dir, "zed", max_imgs=1)

    for img, depth in zip(imgs, depths):
        vis_depth = visualize_depth(depth)
        vis = colorize_depth(depth)
        vis_depth = cv2.hconcat([vis])
        cv2.imshow("Depth", vis_depth)
        cv2.waitKey(0)