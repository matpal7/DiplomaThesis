import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from calibration.image import load_l_r_images_rectified
from src.depth_anything_3.api import DepthAnything3
from utils import load_dict
from visualize_depth import colorize_depth


def get_K_baseline(calib_dict):
    K = calib_dict['K_l']
    tvec = calib_dict['tvec'].reshape(-1)
    baseline = float(np.linalg.norm(tvec))
    baseline = baseline / 1000.0
    return K, baseline

def get_stereo_parameters(calib_dict, scale=1):
    # Use rectified intrinsics if images are undistorted
    K_l = calib_dict['new_K_l'].copy()
    K_r = calib_dict['new_K_r'].copy()

    if scale != 1:
        K_l[0, :] /= scale
        K_l[1, :] /= scale
        K_r[0, :] /= scale
        K_r[1, :] /= scale

    rvec = calib_dict['rvec']
    tvec = calib_dict['tvec'].reshape(-1)

    R_lr, _ = cv2.Rodrigues(rvec)

    t_lr = tvec / 1000.0

    T_left = np.eye(4)

    T_right = np.eye(4)
    T_right[:3, :3] = R_lr
    T_right[:3, 3] = t_lr

    intrinsics = [K_l, K_r]
    extrinsics = [T_left, T_right]

    return intrinsics, extrinsics

def load_model(model="depth-anything/da3-large", device="cuda"):
    device = torch.device(device)
    # Load the model from huggingface hub (or load from local).
    model = DepthAnything3.from_pretrained(model).to(device)
    return model


def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Run DepthAnything_v3 on folders of left/right images")
    parser.add_argument('--model', default=f'depth-anything/da3-large', type=str,
                        help='pretrained model path')
    parser.add_argument("--max_imgs", type=int, default=5)
    parser.add_argument("--monocular", type=bool, default=False)
    args = parser.parse_args()

    NN_name = 'DepthAnything3'
    if args.monocular:
        NN_name += "_mono"
    else:
        NN_name += "_stereo"

    torch.autograd.set_grad_enabled(False)

    root_dir = Path(__file__).resolve().parent
    img_dir = root_dir / "dataset_11032026" / "stereo_4k_depth" / "rgb"
    calib_dict_file = root_dir / "out" / "cameras_parameters" / "calib_data.npy"
    out_dir = root_dir.parent / "out_estimation" / "stereo" / NN_name

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = Path(out_dir) / "vis"
    depth_dir = Path(out_dir) / "depth"
    vis_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, extrinsics = None, None

    calib_dict = load_dict(calib_dict_file)
    if not args.monocular:
        intrinsics, extrinsics = get_stereo_parameters(calib_dict, scale=args.scale)

    model = load_model()
    smaller_resolution = (960, 540)

    imgs_l, imgs_r = load_l_r_images_rectified(
        calib_dict, img_dir, max_imgs=args.max_imgs
    )
    if len(imgs_l) == 0:
        raise RuntimeError("No images found from the provided input arguments")

    logging.info("Found %d images", len(imgs_l))


    for img_l, img_r in tqdm(zip(imgs_l, imgs_r), desc="inference"):
        img_number = img_l.number
        img_l = img_l.get_resized_img(smaller_resolution)

        if args.monocular:
            prediction = model.inference([img_l])
            depth = prediction.depth[0]

        else:
            img_r = img_r.get_resized_img(smaller_resolution)

            prediction = model.inference(
                [img_l, img_r],
                intrinsics=intrinsics,
                extrinsics=None,
                align_to_input_ext_scale=False,
            )
            depth = prediction.depth[0]

        vis = colorize_depth(depth)


        cv2.imwrite(vis_dir / f"{img_number}_vis.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{img_number}_depth.npy", depth)


if __name__ == "__main__":
    main()
