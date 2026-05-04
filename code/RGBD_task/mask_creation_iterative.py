"""
FoundationPose Mask Creator
Creates binary mask (white=object, black=background) from RGB image.

Interactive modes:
  - GrabCut: Auto-segment with rectangle
  - Threshold: Simple color threshold
  - Manual: Brush tool (Q to finish)
"""
import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def create_manual_mask(img_path, output_path, object_name: str):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot load {img_path}")

    if output_path.exists():
        mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        print(f"  Loaded existing mask from {output_path}")
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    drawing = False
    erasing = False  # toggled by pressing R
    radius = 45
    cursor = [-1, -1]

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, radius

        # Always track cursor for the live circle preview
        if event == cv2.EVENT_MOUSEMOVE:
            cursor[0], cursor[1] = x, y

        # Left button: paint FG or erase depending on current mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), radius, 0 if erasing else 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), radius, 0 if erasing else 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

        # Scroll wheel: adjust brush size
        elif event == cv2.EVENT_MOUSEWHEEL:
            radius = max(1, radius + (3 if flags > 0 else -3))
            print(f"  Brush radius: {radius}px")

    safe_name = object_name.replace("/", "_").replace("\\", "_")
    win_title = f"MaskEditor_{safe_name}"

    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1280, 720)

    if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 0:
        raise RuntimeError(
            f"Failed to create OpenCV window '{win_title}'. "
            "Check that a display is available (DISPLAY env var) and that "
            "the Qt backend is properly installed."
        )

    cv2.setMouseCallback(win_title, mouse_callback)

    print(f"\n  Object: {object_name}")
    print(f"  Image:  {img_path}")
    print(f"  Output: {output_path}")
    print("  Controls: Left drag=paint | R=toggle erase mode | Scroll=brush size | C=clear all | S=save | Q=done | ESC=skip")

    while True:
        overlay = img.copy()
        overlay[mask == 255] = (0, 255, 0)
        display = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        # Live brush cursor — red in erase mode, green in paint mode
        cx, cy = cursor
        if cx >= 0 and cy >= 0:
            color = (0, 0, 255) if erasing else (0, 255, 0)
            cv2.circle(display, (cx, cy), radius, color, 2)
            cv2.circle(display, (cx, cy), max(1, radius // 6), color, -1)

        # Mode label in top-left corner
        label = "ERASE" if erasing else "PAINT"
        label_color = (0, 0, 255) if erasing else (0, 255, 0)
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2, cv2.LINE_AA)

        cv2.imshow(win_title, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):                          # R — toggle erase mode
            erasing = not erasing
            print(f"  Mode: {'ERASE' if erasing else 'PAINT'}")
        elif key == ord('c'):
            mask[:] = 0
            print("  Mask cleared.")
        elif key == ord('s'):
            cv2.imwrite(str(output_path), mask)
            print(f"  Saved: {output_path}")
        elif key == ord('q'):
            cv2.destroyWindow(win_title)
            return mask
        elif key == 27:                              # ESC — skip without saving
            print(f"  Skipped: {object_name}")
            cv2.destroyWindow(win_title)
            return None

    cv2.destroyWindow(win_title)
    return mask


def visualize_mask(rgb_path, mask_path):
    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), 0)

    overlay = rgb.copy()
    overlay[mask == 255] = [0, 255, 0]

    cv2.imshow('RGB', rgb)
    cv2.imshow('Mask', mask)
    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_first_image(scene_rgb_dir: Path, suffix: str = "_realsense.png") -> Path | None:
    """Return the first image (by name) in the scene RGB directory."""
    candidates = sorted(scene_rgb_dir.glob(f"*{suffix}"))
    return candidates[0] if candidates else None


def run(args):
    parent_dir = Path(__file__).resolve().parents[3]
    date = args.date

    dataset_dir = parent_dir / "out" / f"out_{date}" / "pose_estimation" / "undistorted_images_NICO"
    out_base    = parent_dir / "out" / f"out_{date}" / "pose_estimation"/ "masks"

    scenes = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")])
    if not scenes:
        print(f"No scenes found in {dataset_dir}")
        return

    print(f"Found {len(scenes)} scene(s): {[s.name for s in scenes]}")
    print(f"Objects to annotate: {args.objects}\n")

    cancelled = False  # set to True when user presses ESC

    for scene_dir in scenes:
        if cancelled:
            break

        scene_name = scene_dir.name
        rgb_dir = scene_dir

        if not rgb_dir.exists():
            print(f"⚠ No rgb/ dir in {scene_dir}, skipping.")
            continue

        first_img = get_first_image(rgb_dir, suffix=args.suffix)
        if first_img is None:
            print(f"⚠ No images with suffix '{args.suffix}' in {rgb_dir}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Scene: {scene_name} | Image: {first_img.name}")
        print(f"{'='*60}")

        for obj in args.objects:
            if cancelled:
                break

            out_dir  = out_base / scene_name / obj
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / first_img.name

            if out_path.exists() and not args.overwrite:
                print(f"  ✓ Already exists, skipping ({obj}): {out_path}")
                continue

            result = create_manual_mask(first_img, out_path, object_name=f"{scene_name}/{obj}")

            if result is None:               # ESC was pressed — abort everything
                print("\n  ✗ Annotation cancelled by user. Stopping all remaining scenes.")
                cancelled = True
                break

            if args.vis and out_path.exists():
                visualize_mask(first_img, out_path)

    if cancelled:
        print("\nAnnotation session ended early (ESC).")
    else:
        print("\nAll scenes annotated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoundationPose Mask Creator — multi-scene, multi-object")

    parser.add_argument("--date",     type=str, default="24042026",
                        help="Dataset date string (e.g. 09042026)")
    parser.add_argument("--objects",  type=str, nargs="+", default=["apple", "orange", "lemon", "rubiks_cube", "scissors",  "chips_box", "wood_block"],
                        help="apple")
    parser.add_argument("--suffix",   type=str, default="_left.png",
                        help="Image filename suffix to look for")

    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if mask already exists")
    parser.add_argument("--vis",      action="store_true", default=False,
                        help="Visualize each mask after saving")

    args = parser.parse_args()
    run(args)