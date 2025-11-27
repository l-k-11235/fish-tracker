import clip
import numpy as np
import torch

from pathlib import Path
from PIL import Image

from fish_tracker.utils.logger import get_logger

class BaseROIFilter:
    def __init__(self,
                 model_path=None,
                 output_dir_template="/app/data/outputs/crops/frame_{frame_num}"):
        self.model_path = model_path
        self.output_dir_template=output_dir_template
    

    def process(self, roi_list, frame_num):
        if not roi_list:
            self.logger.warning("Empty ROI list, returning []")
            return []
        self.output_dir = Path(self.output_dir_template.format(frame_num=frame_num))
        self.output_dir.mkdir(parents=True, exist_ok=True)


class CLIPROIFilter(BaseROIFilter):

    def __init__(self, text=None):
        super().__init__(model_path="ViT-B/32")
        self.text = text or [
            "A small low-quality underwater image region showing a living organism like a fish, even if partly hidden",
            "A small low-quality underwater image region showing only water or a non-living object"
        ]
        self.logger = get_logger("CLIPROIFilter")
        self.logger.info("CLIPROIFilter Initialization")
    
    def build_model(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise

        text_tokens = clip.tokenize(self.text)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def process(self, roi_list, frame_num):
        super().process(roi_list, frame_num)

        if not all("crop" in r and isinstance(r["crop"], Image.Image) for r in roi_list):
            raise ValueError("Each ROI must contain a 'crop' key with a PIL Image.")

        filtered = []
        images = torch.stack([self.preprocess(item["crop"]) for item in roi_list]).to(self.device)
        self.logger.debug(images.shape)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = image_features @ self.text_features.T
        self.logger.debug(similarities.shape)

        filtered = []
        for i, (item, sim) in enumerate(zip(roi_list, similarities)):
            filename = self.output_dir / f"crop_{i}.png"
            item["crop"].save(filename)
            self.debug(sim)
            label = int(np.argmax(sim))
            self.logger.debug(f"{i} label: {label}, similarity: {sim}, ROI size: {item['crop'].size}")
            if label == 0:
                filtered.append(item)
        return filtered

        # for i, item in enumerate(roi_list):
        #     filename = self.output_dir / f"crop_{i}.png"
        #     item["crop"].save(filename)
        #     image = self.preprocess(item["crop"]).unsqueeze(0)
        #     with torch.no_grad():
        #         image_features = self.model.encode_image(image)
        #         image_features /= image_features.norm(dim=-1, keepdim=True)
        #     similarity = (image_features @ self.text_features.T).squeeze(0) #.softmax(dim=-1)
        #     similarity = similarity.cpu().numpy()
        #     label = int(np.argmax(similarity))
        #     self.logger.debug(f'{i} label: {label}, similarity: {similarity},  ROI size: {item["crop"].size}')
        #     if label == 0:
        #         filtered.append(item)
        # return filtered


def build_roi_filter(name: str, **kwargs) -> BaseROIFilter:
    name = name.lower()
    if name == "clip":
        roi_filter = CLIPROIFilter(**kwargs)
        roi_filter.build_model()
        return roi_filter
    else:
        raise ValueError(f"Unknown Roi filter: {name}")
