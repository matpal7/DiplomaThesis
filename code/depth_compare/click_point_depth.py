from pathlib import Path

import cv2
import numpy as np
import sys

from code.image import get_rectify_functions
from code.utils import load_dict

parent_dir = Path(__file__).resolve().parents[3]
date = "12042026"
dataset_dir = parent_dir / "datasets" / f"dataset_{date}" / "stereo_4k_depth"
camera = "left"
scene = 1
# ── Load data ────────────────────────────────────────────────────────────────
rgb_path   = dataset_dir / "rgb" / f"{scene}_{camera}.png"        # ← change to your file
depth_path = parent_dir / "out" / f"out_{date}" / "depth_estimation" / "S2M2_stereo" / "depth" / f"{scene}_depth.npy"        # ← change to your file

out_dir = parent_dir / "out" / f"out_{date}" / "cameras_parameters"
calib_dict = load_dict(out_dir / "calib_data.npy")
undistort_l, undistort_r = get_rectify_functions(calib_dict)

rgb   = cv2.imread(rgb_path)
rgb = undistort_l(rgb)
depth = np.load(depth_path)     # float32, shape (H, W)

if rgb is None:
    raise FileNotFoundError(f"Cannot load image: {rgb_path}")

# ── Resize depth to match rgb if shapes differ ────────────────────────────────
if depth.shape[:2] != rgb.shape[:2]:
    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

# ── Colorize depth for display ───────────────────────────────────────────────
valid = np.isfinite(depth) & (depth > 0)
d_min = float(depth[valid].min()) if valid.any() else 0.0
d_max = float(np.percentile(depth[valid], 99)) if valid.any() else 1.0

depth_norm = np.zeros_like(depth, dtype=np.uint8)
depth_norm[valid] = np.clip(
    (depth[valid] - d_min) / max(d_max - d_min, 1e-6) * 255, 0, 255
).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
depth_color[~valid] = (30, 30, 30)

# ── State ────────────────────────────────────────────────────────────────────
overlay_rgb   = rgb.copy()
overlay_depth = depth_color.copy()
last_point    = [None]   # (x, y)

def draw_point(img, x, y, depth_val):
    out = img.copy()
    cv2.circle(out, (x, y), 8,  (0, 0, 0),   -1)
    cv2.circle(out, (x, y), 6,  (0, 255, 0),  -1)
    cv2.circle(out, (x, y), 2,  (255, 255, 255), -1)

    if np.isfinite(depth_val) and depth_val > 0:
        label = f"{depth_val:.3f} m"
        color_txt = (0, 255, 0)
    else:
        label = "invalid"
        color_txt = (0, 0, 255)

    tx, ty = x + 12, y - 12
    # clamp label inside image
    tx = min(tx, out.shape[1] - 120)
    ty = max(ty, 20)

    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0),     3, cv2.LINE_AA)
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color_txt,     1, cv2.LINE_AA)
    return out

def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    depth_val = float(depth[y, x])
    last_point[0] = (x, y)

    # console output
    if np.isfinite(depth_val) and depth_val > 0:
        print(f"  ({x:4d}, {y:4d})  →  depth = {depth_val:.4f} m")
    else:
        print(f"  ({x:4d}, {y:4d})  →  depth = INVALID")

    # overlay on both windows
    overlay_rgb[:]   = draw_point(rgb,         x, y, depth_val)
    overlay_depth[:] = draw_point(depth_color, x, y, depth_val)

    cv2.imshow("RGB",   overlay_rgb)
    cv2.imshow("Depth", overlay_depth)

# ── Windows ──────────────────────────────────────────────────────────────────
cv2.namedWindow("RGB",   cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RGB",   960, 540)
cv2.resizeWindow("Depth", 960, 540)

cv2.imshow("RGB",   overlay_rgb)
cv2.imshow("Depth", overlay_depth)

cv2.setMouseCallback("RGB",   on_mouse)
cv2.setMouseCallback("Depth", on_mouse)   # clicking depth window also works

print("Click anywhere in either window to read depth.  Press Q or ESC to quit.")

while True:
    key = cv2.waitKey(20) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cv2.destroyAllWindows()