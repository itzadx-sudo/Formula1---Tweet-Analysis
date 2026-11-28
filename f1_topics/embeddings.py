"""Batched transformer tweet embedder with attention-mask mean pooling and L2 normalization."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from bertopic.backend._base import BaseEmbedder
except Exception:
    class BaseEmbedder:  # type: ignore[no-redef]
        pass


def resolve_device(preferred: str | None = None) -> str:
    """Select optimal PyTorch accelerator ('mps', 'cuda', or 'cpu')."""
    if preferred:
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TransformerTweetEmbedder(BaseEmbedder):
    """HuggingFace transformer embedder compatible with BERTopic and standalone batch encoding."""

    def __init__(
        self,
        model_name: str = "vinai/bertweet-base",
        batch_size: int = 64,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_name = resolve_device(device)
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer_kwargs: dict[str, object] = {"use_fast": True}
        if "bertweet" in self.model_name.lower():
            tokenizer_kwargs["normalization"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        self._model = AutoModel.from_pretrained(self.model_name).to(torch.device(self.device_name))
        self._model.eval()

    def embed(self, documents: Sequence[str], verbose: bool = False) -> np.ndarray:
        """Encode tweets into L2-normalized dense embeddings via attention-masked mean pooling."""
        if len(documents) == 0:
            return np.empty((0, 768), dtype=np.float32)

        self._ensure_loaded()
        import torch
        import torch.nn.functional as F

        assert self._tokenizer is not None and self._model is not None
        device = torch.device(self.device_name)
        batches: list[np.ndarray] = []

        for start in range(0, len(documents), self.batch_size):
            chunk = list(documents[start : start + self.batch_size])
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = F.normalize(summed / counts, p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(batches)
