"""SAM 3 segmenter wrapper.

Loads `facebook/sam3` via the HuggingFace transformers `Sam3Processor` /
`Sam3Model`. Each Detection's bbox becomes a positive box prompt; the
returned per-box mask is kept iff it covers `>= min_area_frac` of the
image and overlaps the prompt bbox by `>= min_iou_with_bbox`.
Mask-level NMS de-duplicates overlapping segmentations from different
prompt boxes (e.g. two GDino boxes on the same chair).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..base import Detection, ObjectMask, Segmenter

logger = logging.getLogger(__name__)


class SAM3Segmenter(Segmenter):
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        min_area_frac: float = 0.005,
        min_iou_with_bbox: float = 0.5,
        mask_nms_iou: float = 0.4,
        device: Optional[str] = None,
    ):
        from transformers import Sam3Model, Sam3Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        logger.info("Loading SAM 3 %s on %s", model_id, self.device)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.min_area_frac = min_area_frac
        self.min_iou_with_bbox = min_iou_with_bbox
        self.mask_nms_iou = mask_nms_iou

    def config(self) -> dict:
        return {
            "model": self.model_id,
            "min_area_frac": self.min_area_frac,
            "min_iou_with_bbox": self.min_iou_with_bbox,
            "mask_nms_iou": self.mask_nms_iou,
        }

    @staticmethod
    def _bbox_iou(mask: np.ndarray, bbox: tuple[float, ...]) -> float:
        x0, y0, x1, y1 = (int(round(v)) for v in bbox)
        H, W = mask.shape
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        bbox_area = (x1 - x0) * (y1 - y0)
        inter = int(mask[y0:y1, x0:x1].sum())
        union = int(mask.sum()) + bbox_area - inter
        return inter / union if union > 0 else 0.0

    def _mask_nms(self, masks: list[ObjectMask]) -> list[ObjectMask]:
        if self.mask_nms_iou >= 1.0 or len(masks) <= 1:
            return masks
        ordered = sorted(enumerate(masks), key=lambda t: -t[1].score)
        kept: list[ObjectMask] = []
        kept_arrays: list[np.ndarray] = []
        for _, m in ordered:
            redundant = False
            for k in kept_arrays:
                inter = int(np.logical_and(m.mask, k).sum())
                if inter == 0:
                    continue
                union = int(np.logical_or(m.mask, k).sum())
                if union > 0 and inter / union >= self.mask_nms_iou:
                    redundant = True
                    break
            if not redundant:
                kept.append(m)
                kept_arrays.append(m.mask)
        return kept

    @staticmethod
    def _bbox_xyxy_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
        iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    @torch.inference_mode()
    def segment(self, image_path: Path,
                detections: list[Detection]) -> list[ObjectMask]:
        if not detections:
            return []
        image = Image.open(image_path).convert("RGB")
        boxes_xyxy = [[float(c) for c in d.bbox] for d in detections]
        input_boxes = [boxes_xyxy]
        input_boxes_labels = [[1] * len(detections)]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        masks_t = results["masks"]
        boxes_t = results["boxes"]
        scores_t = results["scores"]
        if len(masks_t) == 0:
            return []

        masks_np = masks_t.cpu().numpy() if hasattr(masks_t, "cpu") else np.asarray(masks_t)
        boxes_np = boxes_t.cpu().numpy() if hasattr(boxes_t, "cpu") else np.asarray(boxes_t)
        scores_np = scores_t.cpu().numpy() if hasattr(scores_t, "cpu") else np.asarray(scores_t)
        if masks_np.dtype != bool:
            masks_np = masks_np.astype(bool)

        H, W = np.array(image).shape[:2]
        min_area = int(self.min_area_frac * H * W)

        # SAM3 returns up to ~N output queries above threshold, NOT one-per-prompt
        # and not in prompt order. Match each input detection to the output
        # query whose predicted box has the highest IoU with the prompt;
        # carry the prompt's label forward.
        out: list[ObjectMask] = []
        used_outputs: set[int] = set()
        for det in detections:
            best_iou = 0.0
            best_idx = -1
            for j, pb in enumerate(boxes_np):
                if j in used_outputs:
                    continue
                iou = self._bbox_xyxy_iou(tuple(det.bbox), tuple(pb))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx < 0 or best_iou < self.min_iou_with_bbox:
                continue
            mask = masks_np[best_idx]
            if mask.shape != (H, W):
                continue
            area = int(mask.sum())
            if area < min_area:
                continue
            used_outputs.add(best_idx)
            ys, xs = np.where(mask)
            cx = float(xs.mean()); cy = float(ys.mean())
            out.append(ObjectMask(
                mask=mask, bbox=det.bbox, score=float(scores_np[best_idx]),
                label=det.label, centroid=(cx, cy), area=area,
            ))
        return self._mask_nms(out)
