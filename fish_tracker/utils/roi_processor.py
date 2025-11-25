# utils/roi_processor.py
from logging import Logger
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

from dataclasses import dataclass
from numpy.typing import NDArray
from PIL import Image
from typing import Any, Callable, Literal, Optional

from fish_tracker.config import ROIProcessorConfig
from fish_tracker.utils.logger import get_logger


class BaseEmbeddingModel:
    def __init__(self, device: str = "cpu") -> None:
        self.device: str = device
        self.model: Optional[nn.Module] = None
        self.preprocess: Optional[Callable[[Image.Image], torch.Tensor]] = None

    def build_model(self) -> None:
        pass

    def preprocess_crops(self, crops: list[NDArray[Any]]) -> list[torch.Tensor]:
        return [torch.empty(0) for _ in crops]

    def compute_embeddings(
        self, processed_crops: list[torch.Tensor]
    ) -> list[list[float]]:
        return [[] for _ in processed_crops]

    def process(self, crops: list[NDArray[Any]]) -> list[list[float]]:
        processed_crops: list[torch.Tensor] = self.preprocess_crops(crops)
        if not processed_crops:
            return []
        embeddings: list[list[float]] = self.compute_embeddings(processed_crops)
        assert len(embeddings) == len(crops)
        return embeddings


class Mobilenetv3smallEmbedding(BaseEmbeddingModel):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.logger: Logger = get_logger("Mobilenetv3smallEmbedding")
        self.logger.info("Mobilenetv3smallEmbedding Initialization")
        self.build_model()

    def build_model(self) -> None:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_model: models.MobileNetV3 = models.mobilenet_v3_small(weights=weights)
        self.preprocess = weights.transforms()
        self.model = (
            nn.Sequential(
                base_model.features,  # convolutions
                nn.AdaptiveAvgPool2d((1, 1)),  # global pooling
                nn.Flatten(),  # 576-d vector
            )
            .to(self.device)
            .eval()
        )

    def preprocess_crops(self, crops: list[NDArray[Any]]) -> list[torch.Tensor]:
        assert self.preprocess is not None, "preprocess() was not initialized"
        try:
            return [self.preprocess(Image.fromarray(crop)) for crop in crops]
        except Exception as e:
            self.logger.error(f"Crop preprocessing failed: {e}")
            return [torch.empty(0) for _ in crops]

    def compute_embeddings(
        self, processed_crops: list[torch.Tensor]
    ) -> list[list[float]]:
        assert self.model is not None, "model was not initialized"
        try:
            batch = torch.stack(processed_crops).to(self.device)
            with torch.no_grad():
                return self.model(batch).cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Embedding computation failed: {e}")
            return []


EMBEDDING_REGISTRY = {
    "mobilenetv3small": Mobilenetv3smallEmbedding,
}


@dataclass
class ROIResult:
    bbox: tuple[int, int, int, int]
    embedding: list[float]
    crop: NDArray[Any]


class ROIProcessor:
    def __init__(self, config: ROIProcessorConfig) -> None:
        self.logger: Logger = get_logger("ROIProcessor")
        self.logger.info("ROIProcessor initialized")
        self.embedding_model_name: Literal[None, "mobilenetv3small"] = (
            config.embedding_model_name
        )
        self.pad_ratio: float = config.pad_ratio
        self.target_size: tuple[int, int] | None = config.target_size
        self.device: str = config.device
        self.embedding_model: BaseEmbeddingModel = self._build_embedding_model()

    def _build_embedding_model(self) -> BaseEmbeddingModel:
        name: str = (self.embedding_model_name or "").lower()
        EmbeddingCls = EMBEDDING_REGISTRY.get(name)
        if EmbeddingCls is None:
            self.logger.warning("No embedding model specified — embeddings disabled.")
            return BaseEmbeddingModel(device=self.device)
        return EmbeddingCls(device=self.device)

    def _extract_crop(
        self, frame: NDArray[Any], bbox: tuple[int, int, int, int]
    ) -> NDArray[Any] | None:
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        if self.pad_ratio:
            pad = int(max(w, h) * self.pad_ratio)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if self.target_size:
            crop = cv2.resize(crop, self.target_size)
        return crop

    def process(
        self, frame: NDArray[Any], bboxes: list[tuple[int, int, int, int]]
    ) -> list[ROIResult]:
        if not bboxes:
            return []

        crops: list[NDArray[Any]] = []
        valid_bboxes: list[tuple[int, int, int, int]] = []
        for bbox in bboxes:
            crop_np = self._extract_crop(frame, bbox)
            if crop_np is None:
                continue
            crops.append(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
            x1, y1, w, h = map(int, bbox)
            valid_bboxes.append((x1, y1, w, h))
        embeddings = self.embedding_model.process(crops)

        # Création des ROIResult
        results: list[ROIResult] = [
            ROIResult(bbox=bbox, embedding=embedding, crop=crop)
            for bbox, embedding, crop in zip(valid_bboxes, embeddings, crops)
        ]
        return results
