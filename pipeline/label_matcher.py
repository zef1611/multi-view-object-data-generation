"""Embedding-based label similarity for cross-frame object matching.

Open-vocab labelers (Gemini, Qwen-VL) describe the same physical object
with paraphrases across frames — `"brown wooden chair"` vs `"wooden
armchair"`, `"silver laptop computer"` vs `"open silver laptop"`. String
equality misses these.

`LabelMatcher` resolves them by computing CLIP-text embeddings and
scoring with cosine similarity. Two labels match iff their cosine ≥
`threshold` (default 0.85, calibrated on the prompt vocabulary used by
the Gemini labeler).

The encoder is loaded lazily on first use, so importing this module is
free. Embeddings are cached in-memory by lowercased string — typical
scenes have 30-100 unique labels, so the cache is small.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LabelMatcher:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.85,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.threshold = threshold
        self.device = device
        self._tokenizer = None
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def config(self) -> dict:
        return {"model": self.model_id, "threshold": self.threshold}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPTextModel, CLIPTokenizerFast
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP-text %s on %s", self.model_id, self.device)
        self._tokenizer = CLIPTokenizerFast.from_pretrained(self.model_id)
        self._model = (
            CLIPTextModel.from_pretrained(self.model_id)
            .to(self.device).eval()
        )

    def _embed_one(self, label: str) -> np.ndarray:
        key = (label or "").strip().lower()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self._ensure_loaded()
        import torch
        with torch.inference_mode():
            inputs = self._tokenizer(
                [key], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            out = self._model(**inputs)
            # `pooler_output`: pooled CLIP text embedding (`[1, hidden]`).
            v = out.pooler_output[0].cpu().numpy().astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        self._cache[key] = v
        return v

    def embed(self, labels: list[str]) -> np.ndarray:
        """Return an `(N, D)` row-normalized embedding matrix. Uses the
        per-label cache so repeated labels across frames are free."""
        return np.stack([self._embed_one(l) for l in labels], axis=0)

    def cosine(self, a: str, b: str) -> float:
        ea = self._embed_one(a)
        eb = self._embed_one(b)
        return float(np.dot(ea, eb))

    def match(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a.strip().lower() == b.strip().lower():
            return True
        return self.cosine(a, b) >= self.threshold

    def cluster(self, labels: list[str]) -> dict[str, str]:
        """Cluster a label list into canonical groups (greedy: each label
        joins the cluster of the first existing canonical it matches).

        Returns a `label -> canonical` dict for downstream remapping.
        """
        canonicals: list[str] = []
        canon_embs: list[np.ndarray] = []
        out: dict[str, str] = {}
        for lbl in labels:
            key = (lbl or "").strip().lower()
            if not key:
                continue
            if key in out:
                continue
            e = self._embed_one(key)
            best_idx = -1
            best_sim = self.threshold
            for i, ce in enumerate(canon_embs):
                sim = float(np.dot(e, ce))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx == -1:
                canonicals.append(key)
                canon_embs.append(e)
                out[key] = key
            else:
                out[key] = canonicals[best_idx]
        return out
