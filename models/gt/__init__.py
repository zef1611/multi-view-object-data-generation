"""Ground-truth-driven detectors (currently ScanNet-only).

`base.py` holds the shared GT-instance extraction logic; `scannet.py`
emits GT bboxes directly, `scannet_gdino.py` re-grounds GT labels
through GDino.
"""

from .scannet import ScanNetGTDetector
from .scannet_gdino import ScanNetGTLabelGDinoDetector

__all__ = ["ScanNetGTDetector", "ScanNetGTLabelGDinoDetector"]
