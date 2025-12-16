from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


class VisualMemoryItem:
    # TODO 图像中主要 item 的文本 / 视觉特征
    image_path: str
    caption: str
    key_objects: List[str]
    summary: str
    text_embedding: np.ndarray
    visual_embedding: np.ndarray
    extra: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class VisualMemoryImage:
    image_path: str
    caption: str
    key_objects: List[str]
    summary: str
    text_embedding: np.ndarray
    visual_embedding: np.ndarray
    items: List[VisualMemoryItem] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

MemoryEntry = Union[VisualMemoryImage, VisualMemoryItem]


class VisualMemoryStore:
    def __init__(self, text_embedder, image_embedder) -> None:
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.images: List[VisualMemoryImage] = []
        self.items: List[VisualMemoryItem] = []


    def add_item(
        self,
        image_path: str,
        caption: str,
        key_objects: List[str],
        extra: Optional[Dict[str, Any]], # BUG here
    ):
        summary_parts = []
        if caption:
            summary_parts.append(caption.strip())
        if key_objects:
            summary_parts.append("Main Objects: " + "; ".join(key_objects))
        summary = "\n".join(summary_parts)

        text_emb = self.text_embedder.encode_documents([caption])[0]
        visual_emb = self.image_embedder.encode_image(image_path)
        
        image_mem = VisualMemoryImage(
            image_path=image_path,
            caption=caption,
            key_objects=key_objects,
            summary=summary,
            text_embedding=text_emb,
            visual_embedding=visual_emb,
            # extra=extra,
        )
        self.images.append(image_mem)
        return image_mem


    def _search_by_embedding(
        self, query_vec: np.ndarray, top_k: int = 5, use_visual: bool = False,
    ):
        """
        - use_visual=False: 用文本 embedding (item.embedding)
        - use_visual=True:  用视觉 embedding (item.visual_embedding)
        """
        if not self.images:
            return []
        sims: List[float] = []
        for image in self.images:
            emb = image.visual_embedding if use_visual else image.text_embedding
            if emb is None:
                continue
            sims.append(float(np.dot(query_vec, emb)))
        idx_scores = sorted(
            list(enumerate(sims)), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(self.images[i], score) for i, score in idx_scores]

    def search_by_text(
        self, query: str, top_k: int = 5
    ):
        """
        文本 -> 视觉记忆 检索。
        """
        if not self.images:
            return []
        q_vec = self.text_embedder.encode_queries([query])[0]
        return self._search_by_embedding(q_vec, top_k=top_k, use_visual=False)

    def search_by_summary_embedding(
        self, query_embedding: np.ndarray, top_k: int = 5
    ):
        return self._search_by_embedding(query_embedding, top_k=top_k, use_visual=False)
    
    def search_by_image(
        self, image_path: str, top_k: int = 5
    ):
        if not self.images or self.image_embedder is None:
            return []
        q_vec = self.image_embedder.encode_image(image_path)
        return self._search_by_embedding(q_vec, top_k=top_k, use_visual=True)