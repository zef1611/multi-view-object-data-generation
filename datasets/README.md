# Adding a new dataset adapter

The pipeline is dataset-agnostic. To support a new dataset, subclass
`BaseSceneAdapter` and implement two methods. Drop the file in this
directory and register it in `generate_correspondences.py:make_adapter`.

## Required methods

```python
from crosspoint_gen.adapters.base import BaseSceneAdapter, Frame

class MyAdapter(BaseSceneAdapter):
    def __init__(self, scene_dir: Path):
        self.scene_dir = scene_dir
        self.scene_id = scene_dir.name
        # Load any once-per-scene metadata here (intrinsics, mesh, etc.)

    def list_frames(self) -> list[str]:
        """Return all frame ids in capture order. Strings, e.g. ['0','1','2',...]."""

    def load_frame(self, frame_id: str) -> Frame:
        """Build a Frame for one frame. See base.py for the field contract."""
```

That's it. The core (`pairs`, `match`, `project`, `dedup`, `emit`) and
all models (`gdino`, `sam21`, `qwen_filter`) work unchanged.

## Optional overrides (for performance or adapters without depth)

```python
def image_path(self, frame_id: str) -> Path:
    """Cheap RGB-path lookup, avoids loading depth+pose. Override if your
    load_frame() does heavy I/O — used by Qwen2.5-VL quality filter and
    by visualizers."""

def load_pose(self, frame_id: str) -> np.ndarray:
    """Cheap pose-only load. Override if pose is in a separate file —
    used by adaptive frame sampling, which needs pose for thousands of
    frames but not depth."""

def reproject(self, src: Frame, p_src, tgt: Frame) -> Optional[Reprojection]:
    """Override when your dataset has no per-pixel depth (ScanNet++
    DSLR, Matterport panoramas). Implement via mesh raycast (trimesh)
    or precomputed SfM tracks. Default uses Frame.depth."""

def qc_instance_mask(self, frame_id: str) -> Optional[tuple[np.ndarray, dict]]:
    """Optional: return (HxW int objectId mask, {id: label}) so the QC
    script can compare emitted points against ground-truth instance
    labels. Skip if your dataset has no per-frame instance masks."""
```

## Camera convention

**OpenCV** (+X right, +Y down, +Z forward). If your dataset uses
OpenGL/Blender (+Y up, -Z forward) or some custom rig, convert in
`load_frame` before returning the `Frame`. The pipeline assumes OpenCV
in `core/pairs._optical_axis` and `core/project.reproject_mask` — a
mismatch will silently corrupt every correspondence.

Quick conversion: `pose_opencv = pose_opengl @ diag([1, -1, -1, 1])`.

## Verification: stand up an adapter without any real data

`tests/test_mock_adapter.py` shows a totally synthetic adapter built
from in-memory numpy arrays — no files on disk, no models, just the
core pipeline. Use it as a template and a regression check that your
new adapter wires up correctly.

## Datasets shipped here

| Adapter | File | Status |
|---|---|---|
| ScanNet | `scannet.py` | implemented |
| ScanNet++ | `scannetpp.py` | stub (DSLR needs mesh raycast in `reproject`) |
| Matterport3D / HM3D | `matterport.py` | stub (panorama → perspective tile unwrap) |
