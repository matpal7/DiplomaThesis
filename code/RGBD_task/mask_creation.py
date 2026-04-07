#!/usr/bin/env python3
"""
FoundationPose Mask Creator
Creates binary mask (white=object, black=background) from RGB image.

Usage:
  python create_mask.py input.png output_mask.png

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


def create_grabcut_mask(img_path, output_path):
    """Automatic GrabCut segmentation"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot load {img_path}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Default rectangle (adjust these 4 numbers to your object)
    rect = (w // 4, h // 4, w // 2, h // 2)
    print(f"Using rect: {rect} (x,y,w,h)")
    print("Edit rect values above and re-run for better results")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_final = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), mask_final)
    print(f"✓ GrabCut mask saved: {output_path}")
    return mask_final


def create_threshold_mask(img_path, output_path, bgr_lower=(0, 100, 100), bgr_upper=(50, 255, 255)):
    """Simple color threshold (good for red apple)"""
    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV range for red apple (adjust as needed)
    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cv2.imwrite(str(output_path), mask)
    print(f"✓ Threshold mask saved: {output_path}")
    return mask


def create_manual_mask(img_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot load {img_path}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    drawing = False
    mode = 1   # 1 = foreground (white), 0 = background (black)
    radius = 7

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, mode, radius
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), radius, 255 if mode == 1 else 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), radius, 255 if mode == 1 else 0, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            mode = 1 - mode
            print("Mode:", "FOREGROUND" if mode == 1 else "BACKGROUND")

    print("Left drag = paint")
    print("Right click = toggle foreground/background")
    print("s = save, q = save and quit")

    cv2.namedWindow("Mask Editor")
    cv2.setMouseCallback("Mask Editor", mouse_callback)

    while True:
        overlay = img.copy()
        overlay[mask == 255] = (0, 255, 0)
        display = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        cv2.imshow("Mask Editor", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(str(output_path), mask)
            print(f"Mask saved: {output_path}")
        elif key == ord('q'):
            cv2.imwrite(str(output_path), mask)
            print(f"Mask saved: {output_path}")
            break

    cv2.destroyAllWindows()
    return mask


def visualize_mask(rgb_path, mask_path):
    """Show original + mask overlay"""
    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), 0)

    overlay = rgb.copy()
    overlay[mask == 255] = [0, 255, 0]  # Green overlay

    cv2.imshow('RGB', rgb)
    cv2.imshow('Mask', mask)
    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoundationPose Mask Creator")
    # parser.add_argument('input', help="Input RGB image (0000.png)")
    # parser.add_argument('output', help="Output mask (demo_data/apple/mask/0000.png)")
    parser.add_argument('--method', choices=['grabcut', 'threshold', 'manual'],
                        default='manual', help="Mask creation method")
    parser.add_argument('--vis', default=True, action='store_true', help="Visualize result")

    args = parser.parse_args()
    parent_dir = Path(__file__).resolve().parents[3]
    date = "07042026"
    img_number = "000"
    scene = "009"
    args.input = parent_dir / "datasets" / f'dataset_{date}' / "cameras_downstream_task" / f"scene_{scene}" / "rgb" / f"{img_number}_zed.png"

    args.output = parent_dir / "out" / f"out_{date}" / "masks"  / f"scene_{scene}" / f"{img_number}_zed.png"


    Path(args.output).parent.mkdir(parents=True, exist_ok=True)




    if args.method == 'grabcut':
        mask = create_grabcut_mask(args.input, args.output)
    elif args.method == 'threshold':
        mask = create_threshold_mask(args.input, args.output)
    elif args.method == 'manual':
        mask = create_manual_mask(args.input, args.output)

    if args.vis:
        visualize_mask(args.input, args.output)

    print(f"✅ Ready for FoundationPose: python run_demo.py")