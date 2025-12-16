import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image


class DinoV2ImageEmbedder:
    def __init__(
        self,
        model_path: str = "facebook/dinov2-base",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def encode_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token: [batch, tokens, dim] -> [batch, dim]
        feats = outputs.last_hidden_state[:, 0, :]  # [1, D]
        feats = F.normalize(feats, dim=-1)
        return feats[0].cpu().numpy()