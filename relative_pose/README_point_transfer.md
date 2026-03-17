# Point transfer between calibrated cameras (single + multi-target)

Script: `relative_pose/visualize_point_transfer.py`

## What it can do

- **Mode `cam1_to_cam2`**: click in source camera (`cam1`) and project point into target camera (`cam2`) using source depth.
- **Mode `cam2_to_cam1_via_cam1_depth`**: click in target (`cam2`) and back-query source (`cam1`) pixel using lookup from `cam1` depth.
- **Mode `multi_target_from_cam1`**: click once in `cam1` and simultaneously draw the projected point in **multiple target cameras** (e.g. RealSense + ZCam), when you have relative poses from `cam1` to each target.

## Example: click in left RGBD and project to RealSense + ZCam

```bash
python relative_pose/visualize_point_transfer.py \
  --mode multi_target_from_cam1 \
  --image-cam1 dataset/.../0_left.png \
  --cam1-calib out/cameras_parameters/left_calibration.yaml \
  --depth-map-cam1 dataset/.../0_left_depth.npy \
  --target-images dataset/.../0_realsense.png dataset/.../0_zcam.png \
  --target-calibs out/cameras_parameters/realsense_calibration.yaml out/cameras_parameters/zcam_calibration.yaml \
  --target-relative-poses out/cameras_parameters/relative_pose/relative_pose_left_to_realsense.yaml out/cameras_parameters/relative_pose/relative_pose_left_to_zcam.yaml \
  --target-names realsense zcam \
  --depth-scale 0.001
```

> If your pose files store opposite direction, add `--invert-relative-pose`.

## Controls

- **Left click**: add point transfer
- **c**: clear points
- **s**: save current view (if `--save` is provided)
- **q / ESC**: quit
