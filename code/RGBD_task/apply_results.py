import numpy as np
import cv2
from pathlib import Path
import trimesh


def load_K(path: str | Path) -> np.ndarray:
    K = np.loadtxt(path, dtype=np.float64)
    assert K.shape == (3, 3), f"Expected 3×3 matrix, got {K.shape}"
    return K


def load_mesh(mesh_path: str | Path):
    mesh = trimesh.load(str(mesh_path))
    return mesh


def get_bbox_corners(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = mesh.bounds
    min_xyz, max_xyz = bounds[0], bounds[1]
    return np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
    ], dtype=np.float64)


def project_points(points_3d: np.ndarray, T: np.ndarray, K: np.ndarray,
                   dist: np.ndarray | None = None) -> np.ndarray:
    R       = T[:3, :3]
    t       = T[:3,  3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    dist    = np.zeros(5) if dist is None else dist
    pts_2d, _ = cv2.projectPoints(points_3d.astype(np.float64),
                                   rvec, t, K, dist)
    return pts_2d.reshape(-1, 2).astype(int)


def draw_bbox_3d(img: np.ndarray, pts_2d: np.ndarray,
                 color=(0, 255, 0), thickness=2) -> np.ndarray:
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
    return img


def draw_axes(img: np.ndarray, T: np.ndarray, K: np.ndarray,
              axis_length: float = 0.05,
              origin_3d: np.ndarray | None = None,
              dist: np.ndarray | None = None,
              thickness: int = 2) -> np.ndarray:
    if origin_3d is None:
        origin_3d = np.zeros(3)

    axes_3d = np.array([
        origin_3d,
        origin_3d + np.array([axis_length, 0.0,         0.0        ]),
        origin_3d + np.array([0.0,         axis_length, 0.0        ]),
        origin_3d + np.array([0.0,         0.0,         axis_length]),
    ], dtype=np.float64)

    R       = T[:3, :3]
    t       = T[:3,  3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    dist    = np.zeros(5) if dist is None else dist

    pts_2d, _ = cv2.projectPoints(axes_3d, rvec, t, K, dist)
    pts_2d    = pts_2d.reshape(-1, 2).astype(int)

    origin = tuple(pts_2d[0])
    cv2.arrowedLine(img, origin, tuple(pts_2d[1]), (0,   0,   255), thickness, tipLength=0.2)
    cv2.arrowedLine(img, origin, tuple(pts_2d[2]), (0,   255,   0), thickness, tipLength=0.2)
    cv2.arrowedLine(img, origin, tuple(pts_2d[3]), (255,   0,   0), thickness, tipLength=0.2)

    font, font_scale, off = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 6
    cv2.putText(img, "X", (pts_2d[1][0]+off, pts_2d[1][1]),     font, font_scale, (0,   0,   255), thickness)
    cv2.putText(img, "Y", (pts_2d[2][0]+off, pts_2d[2][1]),     font, font_scale, (0,   255,   0), thickness)
    cv2.putText(img, "Z", (pts_2d[3][0]+off, pts_2d[3][1]+off), font, font_scale, (255,   0,   0), thickness)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent
    chips_dir  = parent_dir / "demo_data" / "wood_block"

    # K = load_K(chips_dir / "cam_K.txt")

    # ← KEY CHANGE: load mesh with same normalization as FoundationPose
    mesh = trimesh.load(str(chips_dir / "textured.obj"))
    corners_3d = get_bbox_corners(mesh)
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0  # stred pôvodného meshe
    bbox_size = mesh.bounds[1] - mesh.bounds[0]
    axis_length = float(bbox_size.max() * 0.3)


    print(f"Bbox size (m): {bbox_size}  |  Axis length: {axis_length:.4f} m")

    # results_dir = chips_dir / "results"
    # pose_files  = sorted(results_dir.glob("*_zed.txt"))
    #
    # for pose_path in pose_files:
    #     frame_id = pose_path.stem.split("_")[0]
    #     rgb_path = chips_dir / "rgb" / f"{frame_id}_zed.png"
    #
    #     if not rgb_path.exists():
    #         print(f"[SKIP] {rgb_path}")
    #         continue
    #
    #     T   = np.loadtxt(pose_path, dtype=np.float64)
    #     img = cv2.imread(str(rgb_path))
    #     if img is None:
    #         continue
    #
    #     pts_2d = project_points(corners_3d, T, K)
    #     img    = draw_bbox_3d(img, pts_2d, color=(0, 255, 0))
    #     img    = draw_axes(img, T, K, axis_length=axis_length, origin_3d=bbox_center)
    #
    #     x1, y1 = pts_2d[:, 0].min(), pts_2d[:, 1].min()
    #     x2, y2 = pts_2d[:, 0].max(), pts_2d[:, 1].max()
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #     cv2.putText(img, f"Frame {frame_id}", (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    #
    #     print(f"[{frame_id}] t = {T[:3, 3].round(4)}")
    #     cv2.imshow("FoundationPose — bbox", img)
    #
    #     key = cv2.waitKey(0)
    #     if key in (ord('q'), 27):
    #         break
    #     elif key == ord('s'):
    #         out = results_dir / f"{frame_id}_bbox.png"
    #         cv2.imwrite(str(out), img)
    #         print(f"  Saved → {out}")
    #
    # cv2.destroyAllWindows()