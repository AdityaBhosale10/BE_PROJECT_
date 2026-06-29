from __future__ import annotations

import io
import logging
from typing import List

import numpy as np
import requests

logger = logging.getLogger(__name__)


class ClipEmbeddingsService:  # pragma: no cover
    """
    CLIP embeddings (image + text).

    This is intentionally isolated from EmbeddingsService to keep the text-only pipeline light.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:  # pragma: no cover
        try:
            import torch  # type: ignore
            import open_clip  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "CLIP dependencies missing. Install `open-clip-torch` and `pillow`."
            ) from e

        self._torch = torch
        self._Image = Image
        self._open_clip = open_clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_text(self, text: str) -> List[float]:  # pragma: no cover
        torch = self._torch
        tokens = self.tokenizer([text])
        with torch.no_grad():
            feats = self.model.encode_text(tokens.to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().float().numpy().tolist()

    def embed_image_url(self, url: str, timeout_s: int = 20) -> List[float]:  # pragma: no cover
        torch = self._torch
        Image = self._Image

        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(image_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().float().numpy().tolist()

    def embed_image_bytes(self, content: bytes) -> List[float]:  # pragma: no cover
        """Embed image from raw bytes."""
        torch = self._torch
        Image = self._Image

        img = Image.open(io.BytesIO(content)).convert("RGB")
        image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(image_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().float().numpy().tolist()

