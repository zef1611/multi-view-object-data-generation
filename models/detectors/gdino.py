"""Grounding-DINO closed-vocabulary detector wrapper.

Loads `IDEA-Research/grounding-dino-base` via HuggingFace Transformers.

Closed-vocab detection done in three steps:

1. **Chunk** the class list into groups small enough that the dotted
   prompt fits in GDino's text encoder (~256 tokens). The default
   `chunk_size_classes=50` keeps a safe margin for multi-token class
   names. For 177 ScanNet200 classes this is 4 forward passes — much
   faster than per-class batching (177× through both encoders).

2. **Substring-canonicalize** the per-detection label. GDino's
   `text_labels` is the concatenation of every class phrase that crossed
   `text_threshold` (e.g. `"chair office chair armchair"` for a real
   office chair). We map that span back to a single class string by
   picking the **longest class from our list that is a substring** of
   the span — `"office chair"` beats `"chair"` and `"armchair"`.

3. **Cross-chunk NMS** to dedupe boxes that fired in multiple chunks
   (e.g. a chair box surviving in both the chunk containing "chair" and
   the chunk containing "office chair").

Structural classes (wall, floor, ceiling, doorframe, ...) should be
omitted from the class list so they never appear in detections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ..base import Detection, Detector

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SCANNET200_CLASSES_FILE = CONFIG_DIR / "scannet200_general_objects.txt"


# Indoor object vocabulary (~89 classes), whole-objects only. Sub-parts
# (headboard, drawer-as-component, screen, handle, knob, switch) are
# intentionally excluded so labels stay at the parent-object level.
# Structural classes (wall, floor, ceiling, door, window) are also
# intentionally excluded.
DEFAULT_CLASSES: list[str] = [
    # Furniture
    "chair", "office chair", "armchair", "dining chair", "stool", "bench",
    "table", "dining table", "coffee table", "desk", "nightstand",
    "sofa", "couch", "bed", "mattress",
    "bookshelf", "shelf", "cabinet", "dresser", "wardrobe",
    # Appliances / electronics
    "monitor", "tv", "laptop", "computer", "keyboard", "mouse", "printer",
    "refrigerator", "oven", "microwave", "stove", "toaster", "dishwasher",
    "washer", "dryer", "speaker", "lamp", "desk lamp", "floor lamp", "ceiling fan",
    # Kitchen / bathroom
    "sink", "toilet", "bathtub", "shower", "mirror", "towel", "soap dispenser",
    "cup", "bowl", "plate", "bottle", "pan", "pot", "kettle", "knife block",
    # Soft furnishings / decor
    "pillow", "cushion", "blanket", "rug", "carpet", "curtain", "blinds",
    "painting", "picture frame", "poster", "clock", "vase", "plant", "flower pot",
    # Containers / personal items
    "trash can", "box", "basket", "bag", "backpack", "suitcase", "laundry basket",
    "book", "shoe", "hat", "mug", "phone", "remote control",
    # Misc
    "fan", "heater", "radiator", "exit sign", "fire extinguisher",
    "railing", "staircase",
]


def load_classes_from_file(path: Path,
                           max_classes: Optional[int] = None) -> list[str]:
    """Read a one-class-per-line vocabulary file.

    Lines starting with `#` and blank lines are skipped. Inline comments
    (`class_name  # note`) are stripped. If `max_classes` is set, only
    the first N kept classes are returned.
    """
    classes: list[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.split("#", 1)[0].strip()
        if s:
            classes.append(s)
    if max_classes is not None:
        classes = classes[:max_classes]
    return classes


def _canonicalize_label(raw: str, classes_lower: list[str]) -> str:
    """Map GDino's raw label string to a single canonical class.

    GDino has two failure modes that the canonicalizer fixes:

    * **Compound spans**: e.g. `"chair office chair armchair"` is
      returned when multiple class phrases all crossed `text_threshold`
      for the same box. We pick the longest class that is a substring
      of the raw span (`"office chair"` beats `"chair"` and `"armchair"`).

    * **Partial token firing**: e.g. `"trash"` is returned when only
      the first BERT token of `"trash can"` crossed threshold. We pick
      the longest class that *contains* the raw span as a substring
      (`"trash can"` beats `"trash"`).

    Direction 1 (class ⊂ raw) is tried first since it's the strict
    "exact phrase fired" case. Direction 2 (raw ⊂ class) is the fallback
    for partial firings. If neither matches, the raw label is kept.
    """
    s = (raw or "").lower().strip()
    if not s:
        return raw
    # Direction 1: any classes that appear as substrings of raw → pick longest.
    contained = [c for c in classes_lower if c and c in s]
    if contained:
        return max(contained, key=len)
    # Direction 2: any classes that *contain* raw → pick longest.
    containing = [c for c in classes_lower if c and s in c]
    if containing:
        return max(containing, key=len)
    return raw


class GDinoDetector(Detector):
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        classes: Optional[list[str]] = None,
        box_threshold: float = 0.20,
        text_threshold: float = 0.40,
        nms_iou: float = 0.7,
        topk: int = 30,
        chunk_size_classes: int = 50,
        device: Optional[str] = None,
    ):
        self.classes: list[str] = list(classes) if classes else list(DEFAULT_CLASSES)
        # Lowercased copy used for substring canonicalization. Stored once
        # so we don't lower-case on every detection.
        self._classes_lower: list[str] = [c.lower() for c in self.classes]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.topk = topk
        self.chunk_size_classes = chunk_size_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n_chunks = (len(self.classes) + chunk_size_classes - 1) // max(chunk_size_classes, 1)
        logger.info("Loading Grounding-DINO %s on %s (%d classes, %d chunks)",
                    model_id, self.device, len(self.classes), n_chunks)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device).eval()

    def config(self) -> dict:
        """Reported into the perception cache key so threshold/class-list
        changes automatically invalidate cached masks."""
        return {
            "model": self.model_id,
            "classes": list(self.classes),
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "nms_iou": self.nms_iou,
            "topk": self.topk,
            "chunk_size_classes": self.chunk_size_classes,
        }

    @torch.inference_mode()
    def detect(self, frame) -> list[Detection]:
        """Closed-vocab detection against `self.classes`.

        Runs ceil(len(classes) / chunk_size_classes) forward passes, each
        on a sub-list of classes that fits the text-encoder token budget.
        Per-chunk outputs are canonicalized to a single class string,
        merged, and cross-chunk NMS resolves duplicate boxes that fired
        in more than one chunk.
        """
        if not self.classes:
            return []
        image_path = frame.image_path
        image = Image.open(image_path).convert("RGB")
        target_hw = image.size[::-1]

        # Per-chunk detections.
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        for start in range(0, len(self.classes), self.chunk_size_classes):
            chunk = self.classes[start: start + self.chunk_size_classes]
            chunk_lower = self._classes_lower[start: start + self.chunk_size_classes]
            boxes, scores, labels = self._detect_chunk(
                image, target_hw, chunk, chunk_lower,
            )
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_labels.extend(labels)

        if not all_boxes:
            return []

        boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        # Cross-chunk NMS: drops near-duplicate boxes that fired in
        # multiple chunks. Scores select the survivor.
        keep = nms(boxes_t, scores_t, self.nms_iou)[: self.topk].tolist()
        out: list[Detection] = []
        for k in keep:
            x0, y0, x1, y1 = boxes_t[k].tolist()
            out.append(Detection(
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                score=float(scores_t[k]),
                label=all_labels[k],
            ))
        return out

    def _detect_chunk(
        self,
        image: Image.Image,
        target_hw: tuple[int, int],
        classes: list[str],
        classes_lower: list[str],
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """One forward pass over one chunk of classes.

        Returns (boxes, scores, canonical_labels) parallel lists. NMS is
        applied within the chunk; cross-chunk NMS happens in the caller.
        """
        # The HF processor accepts text=[classes] (list-of-lists) and
        # internally merges to "class1. class2. ..." — fine, the dotted
        # string is what GDino was trained on.
        inputs = self.processor(
            images=image, text=[classes],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[target_hw],
        )[0]
        boxes = results["boxes"].cpu()
        scores = results["scores"].cpu()
        raw_labels = results.get("text_labels", results.get("labels"))
        if boxes.numel() == 0:
            return [], [], []

        # Within-chunk NMS first so canonicalization runs on fewer items.
        keep = nms(boxes, scores, self.nms_iou)[: self.topk].tolist()
        out_b: list[list[float]] = []
        out_s: list[float] = []
        out_l: list[str] = []
        for k in keep:
            canon = _canonicalize_label(str(raw_labels[k]), classes_lower)
            out_b.append(boxes[k].tolist())
            out_s.append(float(scores[k]))
            out_l.append(canon)
        return out_b, out_s, out_l

    @torch.inference_mode()
    def detect_batched_prompts(
        self,
        image_path: Path,
        prompts: list[str],
        chunk_size: int = 8,
    ) -> list[list[Detection]]:
        """Run N single-label prompts over ONE image in a batched forward pass.

        Returns a list of length `len(prompts)` — `result[i]` contains the
        detections for `prompts[i]`. Each prompt is evaluated independently
        (separate text encoders for each replica), so no cross-phrase
        attention fusion can occur. Per-prompt NMS and top-k caps are still
        applied inside the loop.

        Used by `gemini+gdino` where the per-frame label list is short
        (~10–15) and per-class encoding is cheap. For long fixed
        vocabularies (ScanNet200) prefer `detect()` instead.

        `chunk_size` caps the micro-batch dimension to keep peak memory
        bounded when many prompts are supplied.
        """
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not prompts:
            return []
        image = Image.open(image_path).convert("RGB")
        target_hw = image.size[::-1]  # (H, W)

        out_per_prompt: list[list[Detection]] = []
        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start: start + chunk_size]
            n = len(chunk)
            inputs = self.processor(
                images=[image] * n,
                text=chunk,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            results_list = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[target_hw] * n,
            )
            for results in results_list:
                boxes = results["boxes"].cpu()
                scores = results["scores"].cpu()
                labels = results.get("text_labels", results.get("labels"))
                if boxes.numel() == 0:
                    out_per_prompt.append([])
                    continue
                keep = nms(boxes, scores, self.nms_iou)[: self.topk].tolist()
                dets = []
                for k in keep:
                    x0, y0, x1, y1 = boxes[k].tolist()
                    dets.append(Detection(
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        score=float(scores[k]),
                        label=str(labels[k]),
                    ))
                out_per_prompt.append(dets)
        return out_per_prompt

    @torch.inference_mode()
    def detect_multi_frame(
        self,
        items: list[tuple[Path, list[str]]],
        micro_batch: int = 4,
    ) -> list[list[list[Detection]]]:
        """Run (frame_i, prompt_j) rows across multiple frames in one HF call.

        ``items`` is a list of ``(image_path, prompts)`` pairs. Returns
        ``out`` such that ``out[i][j]`` is the detection list for
        ``items[i][1][j]`` on ``items[i][0]``. Mirrors
        :meth:`detect_batched_prompts` exactly when ``items`` has length 1.

        ``micro_batch`` caps the flat (frame, prompt) batch dimension
        per HF forward to keep peak GPU memory bounded.
        """
        if not items:
            return []

        # Flatten (image, prompt) rows. Cache PIL.Image per unique path so
        # we open each frame at most once even if it appears in multiple
        # micro-batches.
        rows: list[tuple[int, int, Image.Image, tuple[int, int], str]] = []
        image_cache: dict[Path, tuple[Image.Image, tuple[int, int]]] = {}
        out: list[list[list[Detection]]] = []
        for i, (image_path, prompts) in enumerate(items):
            cleaned = [p.strip() for p in prompts if p and p.strip()]
            out.append([[] for _ in cleaned])
            if not cleaned:
                continue
            if image_path not in image_cache:
                img = Image.open(image_path).convert("RGB")
                image_cache[image_path] = (img, img.size[::-1])  # (H, W)
            img, target_hw = image_cache[image_path]
            for j, p in enumerate(cleaned):
                rows.append((i, j, img, target_hw, p))

        if not rows:
            return out

        for start in range(0, len(rows), micro_batch):
            chunk = rows[start: start + micro_batch]
            images = [r[2] for r in chunk]
            target_sizes = [r[3] for r in chunk]
            texts = [r[4] for r in chunk]
            inputs = self.processor(
                images=images, text=texts,
                return_tensors="pt", padding=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            results_list = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
            for (frame_idx, prompt_idx, _img, _hw, _p), results in zip(
                    chunk, results_list):
                boxes = results["boxes"].cpu()
                scores = results["scores"].cpu()
                labels = results.get("text_labels", results.get("labels"))
                if boxes.numel() == 0:
                    continue
                keep = nms(boxes, scores, self.nms_iou)[: self.topk].tolist()
                dets: list[Detection] = []
                for k in keep:
                    x0, y0, x1, y1 = boxes[k].tolist()
                    dets.append(Detection(
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        score=float(scores[k]),
                        label=str(labels[k]),
                    ))
                out[frame_idx][prompt_idx] = dets
        return out
