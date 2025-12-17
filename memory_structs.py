from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor

@dataclass
class VisualMemoryItem:
    # TODO 图像中主要 item 的文本 / 视觉特征
    # add box
    image_path: str
    caption: str
    key_objects: str
    text_embedding: np.ndarray
    visual_embedding: Optional[np.ndarray] = None
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
        # self.sam3_model = build_sam3_image_model()
        # self.processor = Sam3Processor(sam3_model)
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
        
        image_items: List[VisualMemoryItem] = []
        if key_objects:
            item_visual_embeddings = self.get_sam3_images(image_path, key_objects)
            # import pdb; pdb.set_trace()
            item_text_embeds = self.text_embedder.encode_documents(key_objects)
            for idx, item_cap in enumerate(key_objects):

                item_vis_emb: Optional[np.ndarray] = None
                if idx < len(item_visual_embeddings):
                    item_vis_emb = item_visual_embeddings[idx]
    
                item = VisualMemoryItem(
                    image_path=image_path,
                    caption=item_cap,
                    key_objects=item_cap,
                    text_embedding=item_text_embeds[idx],
                    visual_embedding=item_vis_emb,
                    extra={"index": idx},
                )

                image_items.append(item)
                self.items.append(item)

        image_mem.items = image_items
        self.images.append(image_mem)
        return image_mem

    def get_sam3_images(self, image_path, key_objects):
        if False:
            # Load an image
            image = Image.open(image_path)
            inference_state = processor.set_image(image)
            # Prompt the model with text
            output = processor.set_text_prompt(state=inference_state, prompt=key_objects)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        
        return []
    
    
    def _search_by_embedding(
        self, query_vec: np.ndarray, top_k: int = 5, use_visual: bool = False, level: str = "image",  # "image" or "item"
    ):
        """
        - use_visual=False: 用文本 embedding (item.embedding)
        - use_visual=True:  用视觉 embedding (item.visual_embedding)
        """
        if level == "image":
            candidates = self.images
        elif level == "item":
            candidates = self.items
        else:
            raise ValueError("level must be 'image' or 'item'")
        if not candidates:
            return []
        
        sims: List[float] = []
        for image in candidates:
            emb = image.visual_embedding if use_visual else image.text_embedding
            if emb is None:
                continue
            sims.append(float(np.dot(query_vec, emb)))
        idx_scores = sorted(
            list(enumerate(sims)), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(candidates[i], score) for i, score in idx_scores]


    def search_by_text(
        self, query: str, top_k: int = 5, search_level: str = "image"
    ):
        """
        文本 -> 视觉记忆 检索。
        """
        if not self.images:
            return []
        q_vec = self.text_embedder.encode_queries([query])[0]
        return self._search_by_embedding(
            q_vec, top_k=top_k, use_visual=False, level=search_level
        )
        # return self._search_by_embedding(q_vec, top_k=top_k, use_visual=False)


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