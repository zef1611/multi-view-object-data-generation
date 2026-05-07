"""SAM 2.1 segmenter wrapper.

Loads `facebook/sam2.1-hiera-large` via the official `sam2` package's
`SAM2ImagePredictor.from_pretrained`. Each Detection's bbox becomes a
box prompt; the highest-scoring mask per box is returned.
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


class SAM21Segmenter(Segmenter):
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        min_area_frac: float = 0.005,
        min_iou_with_bbox: float = 0.5,
        mask_nms_iou: float = 0.4,
        device: Optional[str] = None,
        compile_image_encoder: bool = False,
    ):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        logger.info("Loading SAM 2.1 %s on %s", model_id, self.device)
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id)
        # SAM2 reads device from its underlying model.
        self.predictor.model.to(self.device)
        self.min_area_frac = min_area_frac
        self.min_iou_with_bbox = min_iou_with_bbox
        # Mask-level NMS catches cases where two GDino boxes have
        # bbox-IoU < dedup_iou but the segmented masks land on the
        # same physical object (e.g. "black folding chair" + "dark
        # brown backrest pillow" both segmenting one chair).
        self.mask_nms_iou = mask_nms_iou
        self.compile_image_encoder = compile_image_encoder
        if compile_image_encoder:
            # Wrap only the dense ViT image encoder. Mask decoder has
            # variable box-count input → recompile thrash if compiled.
            # mode="default" runs inductor without CUDA graphs (graphs
            # would invalidate on every new image size).
            try:
                enc = self.predictor.model.image_encoder
                self.predictor.model.image_encoder = torch.compile(
                    enc, mode="default", fullgraph=False, dynamic=False,
                )
                logger.info("torch.compile enabled on SAM image_encoder "
                            "(mode=default, dynamic=False)")
            except Exception as e:  # compile is best-effort; never break the run
                logger.warning("torch.compile of SAM image_encoder failed: %s; "
                               "continuing without compile", e)

    def config(self) -> dict:
        return {
            "model": self.model_id,
            "min_area_frac": self.min_area_frac,
            "min_iou_with_bbox": self.min_iou_with_bbox,
            "mask_nms_iou": self.mask_nms_iou,
            "compile_image_encoder": self.compile_image_encoder,
        }

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

    def _postprocess_frame_masks(
        self,
        detections: list[Detection],
        masks_np: np.ndarray,
        scores_np: np.ndarray,
        H: int,
        W: int,
    ) -> list[ObjectMask]:
        """Shared tail for single- and multi-frame paths.

        ``masks_np`` shape is normalized to (N, H, W) and ``scores_np``
        to (N,) by the caller. Applies area-fraction floor, bbox-IoU
        floor, and finally mask-NMS.
        """
        min_area = int(self.min_area_frac * H * W)
        out: list[ObjectMask] = []
        for det, m_raw, s in zip(detections, masks_np, scores_np):
            mask = (m_raw > 0.5)
            area = int(mask.sum())
            if area < min_area:
                continue
            if self._bbox_iou(mask, det.bbox) < self.min_iou_with_bbox:
                continue
            ys, xs = np.where(mask)
            cx = float(xs.mean()); cy = float(ys.mean())
            out.append(ObjectMask(
                mask=mask, bbox=det.bbox, score=float(s),
                label=det.label, centroid=(cx, cy), area=area,
            ))
        return self._mask_nms(out)

    @staticmethod
    def _normalize_predict_shapes(masks_np: np.ndarray, scores_np: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """Match SAM2's per-image output shape to (N, H, W) + (N,).

        SAM2 sometimes squeezes when N==1 → (H, W); with multimask_output=False
        and a batch box it returns (N, 1, H, W). Handle both.
        """
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        if scores_np.ndim == 2:
            scores_np = scores_np[:, 0]
        elif scores_np.ndim == 0:
            scores_np = scores_np[None]
        return masks_np, scores_np

    @torch.inference_mode()
    def segment(self, image_path: Path,
                detections: list[Detection]) -> list[ObjectMask]:
        if not detections:
            return []
        image = np.array(Image.open(image_path).convert("RGB"))
        H, W = image.shape[:2]
        self.predictor.set_image(image)

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        masks_np, scores_np, _ = self.predictor.predict(
            box=boxes, multimask_output=False,
        )
        masks_np, scores_np = self._normalize_predict_shapes(masks_np, scores_np)
        return self._postprocess_frame_masks(detections, masks_np, scores_np, H, W)

    @torch.inference_mode()
    def segment_multi_frame(
        self,
        items: list[tuple[Path, list[Detection]]],
    ) -> list[list[ObjectMask]]:
        """Multi-frame analogue of :meth:`segment`.

        Calls SAM2's ``set_image_batch`` + ``predict_batch`` once over
        every non-empty frame, then applies the same per-frame
        postprocess tail. Frames whose detection list is empty are
        skipped entirely (no SAM call) and return ``[]``.
        """
        out: list[list[ObjectMask]] = [[] for _ in items]
        active_idx = [i for i, (_, dets) in enumerate(items) if dets]
        if not active_idx:
            return out

        images: list[np.ndarray] = []
        for i in active_idx:
            path, _ = items[i]
            images.append(np.array(Image.open(path).convert("RGB")))

        self.predictor.set_image_batch(images)
        box_batch = [
            np.array([d.bbox for d in items[i][1]], dtype=np.float32)
            for i in active_idx
        ]
        masks_list, scores_list, _ = self.predictor.predict_batch(
            box_batch=box_batch, multimask_output=False,
        )
        for k, i in enumerate(active_idx):
            _path, dets = items[i]
            img = images[k]
            H, W = img.shape[:2]
            masks_np, scores_np = self._normalize_predict_shapes(
                masks_list[k], scores_list[k],
            )
            out[i] = self._postprocess_frame_masks(
                dets, masks_np, scores_np, H, W,
            )
        return out
