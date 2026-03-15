# RGBD → 3D → ľavá referenčná kamera (aj bez hĺbky v ľavej kamere)

Skript `relative_pose/visualize_point_transfer.py` má dva režimy:

- **Mode A (`cam1_to_cam2`)**: klikáš v `cam1` (napr. RealSense) a bod sa premietne do `cam2` (ľavá stereo).
- **Mode B (`cam2_to_cam1_via_cam1_depth`)**: klikáš v `cam2` (ľavá stereo) a skript nájde zodpovedajúci pixel v `cam1` (RealSense),
  **aj keď ľavá kamera nemá depth**. Lookup sa vyráta iba z RealSense hĺbky.

To rieši presne situáciu: máš depth iba z Intel RealSense a chceš zarovnať XY súradnice voči ľavej stereo.

## Matematika pipeline

1. Vezmeme pixel z RealSense `(u_rs, v_rs)` + `Z_rs` z depth mapy.
2. Back-projection do 3D v RealSense kamere: `X_rs`.
3. Transformácia do ľavej kamery: `X_left = T_left_rs * X_rs`.
4. Projekcia do ľavého obrazu: `(u_left, v_left)`.

Pri režime B sa tento krok spraví pre veľa RealSense pixelov a vyrobí sa lookup `left_pixel -> realsense_pixel`.

## A) RealSense -> Left (klikám v RealSense)

```bash
python relative_pose/visualize_point_transfer.py \
  --mode cam1_to_cam2 \
  --image-cam1 dataset_11032026/stereo_4k_relative_pose/rgb/0_realsense.png \
  --image-cam2 dataset_11032026/stereo_4k_relative_pose/rgb/0_left.png \
  --cam1-calib out/cameras_parameters/realsense_calibration.yaml \
  --cam2-calib out/cameras_parameters/left_NICO.yaml \
  --relative-pose out/cameras_parameters/relative_pose/relative_pose_realsense_to_left.yaml \
  --depth-map-cam1 dataset_11032026/stereo_4k_relative_pose/depth/0_realsense_depth.npy \
  --source-name realsense --target-name left
```

## B) Left -> RealSense bez left depth (klikám v ľavej kamere)

```bash
python relative_pose/visualize_point_transfer.py \
  --mode cam2_to_cam1_via_cam1_depth \
  --image-cam1 dataset_11032026/stereo_4k_relative_pose/rgb/0_realsense.png \
  --image-cam2 dataset_11032026/stereo_4k_relative_pose/rgb/0_left.png \
  --cam1-calib out/cameras_parameters/realsense_calibration.yaml \
  --cam2-calib out/cameras_parameters/left_NICO.yaml \
  --relative-pose out/cameras_parameters/relative_pose/relative_pose_realsense_to_left.yaml \
  --depth-map-cam1 dataset_11032026/stereo_4k_relative_pose/depth/0_realsense_depth.npy \
  --source-name realsense --target-name left
```

V tomto režime klikáš do ľavého obrazu a vpravo sa zobrazí zodpovedajúci bod v RealSense RGB.

> Ak je transformácia uložená opačne, pridaj `--invert-relative-pose`.

## Poznámka k obmedzeniam

Bez hĺbky v ľavej kamere vieš mapovať len body/scény, ktoré vidí RealSense depth.
Ak je pixel vľavo mimo pokrytia RealSense (alebo v oklúzii), lookup preň nebude existovať.

## Ovládanie

- **ľavý klik**: pridá korešpondenciu bodu
- **c**: vymaže body
- **s**: uloží vizualizáciu
- **q / ESC**: koniec
