from pathlib import Path

import numpy as np
import cv2

from code.image import load_rgb_depth_pairs

def colorize_depth(
    depth_m: np.ndarray,
    color_map=cv2.COLORMAP_TURBO,
    bar_width: int = 100,
    show_values: bool = True,
    vmin: float = None,
    vmax: float = None,
) -> np.ndarray:
    valid_mask = np.isfinite(depth_m) & (depth_m > 0)

    if not np.any(valid_mask):
        h, w = depth_m.shape[:2]
        return np.zeros((h, w + bar_width, 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(depth_m[valid_mask], 2))
    if vmax is None:
        vmax = float(np.percentile(depth_m[valid_mask], 98))

    # Main depth visualization
    depth_norm = (depth_m - vmin) / (vmax - vmin + 1e-8)
    depth_norm[~valid_mask] = 0
    depth_norm = np.clip(depth_norm, 0, 1)

    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_u8, color_map)
    depth_colored[~valid_mask] = 0

    h, w = depth_colored.shape[:2]

    # Create vertical color bar (top=max, bottom=min)
    gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    gradient = np.repeat(gradient, 20, axis=1)
    color_bar = cv2.applyColorMap(gradient, color_map)

    # White panel for bar + text
    panel = np.full((h, bar_width, 3), 255, dtype=np.uint8)
    panel[:, 10:30] = color_bar

    if show_values:
        cv2.putText(
            panel,
            f"{vmax:.2f} m",
            (32, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{vmin:.2f} m",
            (32, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        # cv2.putText(
        #     panel,
        #     "depth",
        #     (5, 20),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.45,
        #     (0, 0, 0),
        #     1,
        #     cv2.LINE_AA,
        # )

    return cv2.hconcat([depth_colored, panel])

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[2]
    date = "24042026"

    dataset_dir = parent_dir / 'datasets' / f"dataset_{date}"
    relative_pose_dir = dataset_dir / "stereo_4k_depth"
    max_imgs = 10

    imgs_zed, depths_zed           = load_rgb_depth_pairs(relative_pose_dir, "zed",        max_imgs=max_imgs)
    imgs_realsense, depths_realsense = load_rgb_depth_pairs(relative_pose_dir, "realsense", max_imgs=max_imgs)

    DISPLAY_HEIGHT = 360

    def resize_to_height(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_AREA)

    def label(img, text):
        out = img.copy()
        # cv2.putText(out, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(out, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),       1, cv2.LINE_AA)
        return out

    for img_zed, depth_zed, img_realsense, depth_realsense in zip(
        imgs_zed, depths_zed, imgs_realsense, depths_realsense
    ):
        img_realsense = img_realsense.get_img()
        img_zed       = img_zed.get_small_img(2)

        # ── compute shared depth range from BOTH cameras ──────────────────────
        depth_zed_f   = depth_zed.astype(np.float32)
        depth_rs_f    = depth_realsense.astype(np.float32)

        valid_zed = np.isfinite(depth_zed_f)   & (depth_zed_f > 0)
        valid_rs  = np.isfinite(depth_rs_f)    & (depth_rs_f  > 0)

        all_valid_depths = np.concatenate([
            depth_zed_f[valid_zed],
            depth_rs_f[valid_rs],
        ])

        shared_vmin = float(np.percentile(all_valid_depths, 2))
        shared_vmax = float(np.percentile(all_valid_depths, 98))

        # ── colorize with shared range ────────────────────────────────────────
        vis_depth_zed       = colorize_depth(depth_zed_f,  vmin=shared_vmin, vmax=shared_vmax)
        vis_depth_realsense = colorize_depth(depth_rs_f,   vmin=shared_vmin, vmax=shared_vmax)

        # ── resize to display height ──────────────────────────────────────────
        target_h    = DISPLAY_HEIGHT
        img_zed_r   = resize_to_height(img_zed,              target_h)
        img_rs_r    = resize_to_height(img_realsense,        target_h)
        dep_zed_r   = resize_to_height(vis_depth_zed,        target_h)
        dep_rs_r    = resize_to_height(vis_depth_realsense,  target_h)

        # ── labels ────────────────────────────────────────────────────────────
        top_row    = cv2.hconcat([label(img_zed_r,  "ZED Mini - RGB"),   label(dep_zed_r, "ZED Mini - Depth")])
        bottom_row = cv2.hconcat([label(img_rs_r,   "RealSense - RGB"),  label(dep_rs_r,  "RealSense - Depth")])

        # ── combine and display ───────────────────────────────────────────────
        w = min(top_row.shape[1], bottom_row.shape[1])
        combined = cv2.vconcat([top_row[:, :w], bottom_row[:, :w]])

        cv2.imshow("ZED Mini vs RealSense - RGB & Depth", combined)
        cv2.waitKey(0)

    cv2.destroyAllWindows()