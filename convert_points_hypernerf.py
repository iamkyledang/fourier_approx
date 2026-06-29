import os
import numpy as np
from plyfile import PlyData, PlyElement

HYPERNERF_INTERP = "hypernerf_interp"

# Each outer folder; the actual dataset lives one level deeper (auto-detected).
SCENES = [
    "interp_aleks-teapot",
    "interp_chickchicken",
    "interp_cut-lemon",
    "interp_hand",
    "interp_slice-banana",
    "interp_torchocolate",
]

for scene in SCENES:
    outer_path = os.path.join(HYPERNERF_INTERP, scene)
    inner_dirs = [
        d for d in os.listdir(outer_path)
        if os.path.isdir(os.path.join(outer_path, d))
    ]
    assert len(inner_dirs) == 1, (
        f"Expected exactly one dataset subfolder in {outer_path}, got: {inner_dirs}"
    )
    dataset_path = os.path.join(outer_path, inner_dirs[0])
    npy_path = os.path.join(dataset_path, "points.npy")
    ply_path = os.path.join(dataset_path, "points3d.ply")

    if os.path.exists(ply_path):
        print(f"[{scene}] already has points3d.ply — skipping.")
        continue

    if not os.path.exists(npy_path):
        print(f"[{scene}] no points.npy found — skipping.")
        continue

    pts = np.load(npy_path)
    xyz = pts[:, :3]
    # Neutral grey: avoids random SH initialization noise while still allowing
    # the model to learn colors freely from the first iteration.
    rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex = np.zeros(len(xyz), dtype=dtype)
    vertex['x'], vertex['y'], vertex['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex['red'], vertex['green'], vertex['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    PlyData([PlyElement.describe(vertex, 'vertex')]).write(ply_path)
    print(f"[{scene}] converted {len(xyz)} points → {ply_path}")

print("\nDone.")
