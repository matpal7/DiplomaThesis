# Estimating relative pose between stereo-left and RGBD camera using ChArUco

This workflow estimates a rigid transform between two cameras that see the same ChArUco board.

## Distorted vs. undistorted images

For omnidirectional cameras, use the **original distorted images** and set the model to `omni`.
The script internally undistorts 2D detections with `cv2.omnidir.undistortPoints(...)` before PnP and computes reprojection error with `cv2.omnidir.projectPoints(...)`.

So you do **not** need to pre-undistort images yourself.

## 1) Required inputs

- Intrinsics for **left stereo camera** in `.npy` dict.
  - For pinhole: `K` + `dist` (or `D`).
  - For omni: `K` + distortion (`dist`/`D`) + `xi`.
- Intrinsics for **RGB stream of RGBD camera** in `.yaml`/`.yml` (with `K`, `D`) or in `.npy` dict.
- Synchronized image pairs in one folder, for example:
  - `000_left.png`
  - `000_realsense.png`
  - `001_left.png`
  - `001_realsense.png`

## 2) Run pose estimation

From repository root:

```bash
python calibration/ChArUco/charuco_relative_pose_pnp.py \
  --image-dir dataset_09032026/depth/rgb \
  --cam1-calib NICO/out_2/left_calib.npy \
  --cam2-calib calibration/realsense_calibration.yaml \
  --cam1-suffix _left.png \
  --cam2-suffix _realsense.png \
  --cam1-model omni \
  --cam2-model pinhole \
  --output NICO/out_2/T_realsense_from_left.npy
```

The script outputs:

- `T_cam2_cam1` (4x4 matrix): transform from camera 1 frame to camera 2 frame.
- `rvec_cam2_cam1`, `tvec_cam2_cam1`.
- Per-pair reprojection error and aggregate consistency metrics.

## 3) Transform direction

If you use:
- `cam1 = left stereo`,
- `cam2 = RGBD RGB`,

then result `T_cam2_cam1` equals:

- **`T_rgbd_left`** (RGBD <- left).

If you need the opposite direction:

```python
T_left_rgbd = np.linalg.inv(T_rgbd_left)
```

## 4) Practical advice

- Use at least 15–20 image pairs with varied board orientations/distances.
- Reject runs where average reprojection error is high (typically > 1.0–1.5 px for good images).
- Ensure both cameras use fixed focus/exposure during capture.
- Keep ChArUco board parameters identical to the physical board used during capture.
