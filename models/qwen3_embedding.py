from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class QwenTextEmbedding:
    """
    使用 Qwen3-Embedding 系列模型，将文本编码为向量，用于相似度检索。
    参考 Hugging Face model card 的 Transformers 使用方式。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def last_token_pool(self, last_hidden_states, attention_mask):
        """
        与官方示例一致的 last_token_pool 实现。
        对于右 padding，取每个样本的最后一个非 padding token；
        对于左 padding，直接取最后一个位置。
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _encode(self, texts: List[str]):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def encode_queries(self, queries: List[str]):
        # wrapped = [self._build_query_text(q) for q in queries]
        # return self._encode(wrapped)
        return self._encode(queries)

    def encode_documents(self, docs: List[str]):
        return self._encode(docs)