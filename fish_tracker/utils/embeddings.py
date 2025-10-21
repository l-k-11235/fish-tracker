import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image


class EmbeddingGenerator:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.embedding_model = None
        self.preprocess = None
        self.crops_embeddings = None

    def build_model(self):
        raise NotImplementedError

    def get_crop(self, bbox, frame):
        x1, y1, w, h = map(int, bbox)
        x2 = x1 + w
        y2 = y1 + h
        crop = Image.fromarray(frame[y1:y2, x1:x2])
        return crop

    def get_crops_embeddings(self, frame, bboxes):
        raise NotImplementedError


class Mobilenetv3small(EmbeddingGenerator):
    def build_model(self):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet_v3_small = models.mobilenet_v3_small(weights=weights)
        mobilenet_v3_small.eval()

        # Preprocessing
        preprocess = weights.transforms()

        # Emmbedding model
        embedding_mobilenet_v3_small = nn.Sequential(
            mobilenet_v3_small.features,  # convolutions
            nn.AdaptiveAvgPool2d((1, 1)),  # global pooling
            nn.Flatten(),  # 576-d vector
        )
        embedding_mobilenet_v3_small.eval()

        self.embedding_model = embedding_mobilenet_v3_small
        self.preprocess = preprocess

    def get_crops_embeddings(self, frame, bboxes):
        if len(bboxes) == 0:
            self.crops_embeddings = []
            return
        crops = []
        for bbox in bboxes:
            crop = self.get_crop(bbox, frame)
            tensor = self.preprocess(crop)
            crops.append(tensor)

        batch = torch.stack(crops)  # shape (N, C, H, W)

        with torch.no_grad():
            self.crops_embeddings = self.embedding_model(batch).numpy().tolist()


def build_embedding_model(name: str) -> EmbeddingGenerator:
    name = name.lower()
    if name == "mobilenetv3small":
        model = Mobilenetv3small()
        model.build_model()
        return model
    else:
        raise ValueError(f"Unknown embedding model: {name}")
