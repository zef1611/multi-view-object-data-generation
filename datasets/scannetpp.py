"""ScanNet++ adapter (STUB).

ScanNet++ ships DSLR + iPhone captures plus an aligned 3D mesh, but the
DSLR images do NOT have per-pixel depth. To wire it in, this subclass
must:

1. `load_frame` reads RGB + camera intrinsics + pose from
   `<scene>/dslr/colmap/cameras.txt` (COLMAP) or `<scene>/iphone/`
   (with depth) and returns a `Frame` whose `.depth` is None for DSLR.

2. `reproject` overrides the default (depth-based) version with mesh
   raycasting via trimesh: for each src pixel, cast a ray through the
   mesh, take the first hit as `X_world`, then project into target.
"""

from .base import BaseSceneAdapter


class ScanNetPPAdapter(BaseSceneAdapter):
    source_name = "scannet++"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ScanNetPPAdapter is a stub. Implement list_frames/load_frame "
            "and override reproject() with trimesh raycast for DSLR frames."
        )
