import json
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from utils import extract_json_fragment

class QwenVLBackend:
    """
    对 Qwen3-VL 做一层简单封装，提供：
        - 通用多模态对话接口 chat()
        - 图像语义抽取 extract_image_semantics()
        - Query 改写 rewrite_query()
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ):
        """
        封装一次 Qwen3-VL 的 chat 调用。
        messages: 符合 Qwen3-VL chat_template 的消息列表。
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Qwen3-VL 不需要 token_type_ids，如果存在就去掉
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(
                inputs["input_ids"], generated_ids
            )
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # NOTE: batch = 0
        return output_texts[0]

    # ---------- 对话 ----------

    def chat(
        self,
        history: List[Dict[str, Any]],
        user_text: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ):
        """
        基于已有 history + 本轮 user 输入进行多模态对话。
        history: 形如 [{"role": "user"/"assistant", "content": [...]}, ...]
        """
        image_paths = image_paths or []
        content: List[Dict[str, Any]] = []

        for p in image_paths:
            img = Image.open(p).convert("RGB")
            content.append({"type": "image", "image": img})

        if user_text:
            content.append({"type": "text", "text": user_text})

        messages = history + [
            {
                "role": "user",
                "content": content,
            }
        ]
        return self._generate(messages)

    # ---------- 图像语义抽取 ----------

    def extract_image_semantics(
        self,
        image_path: str,
        extra_instruction: Optional[str] = None,
    ):
        """
        语义信息: caption + obj list
        Returns:
            {
                "caption": "...",
                "objects": ["女孩", "白色小狗", "草地"],
                "raw_output": "模型原始输出"
            }
        """
        img = Image.open(image_path).convert("RGB")

        base_instruction = (
            "Extract the key visual entities from the image (people, objects, scenes, text, etc.). "
            "Output the result strictly in JSON format with the following fields: "
            "caption: a one-sentence summary of the entire image; "
            "objects: a list of strings, where each string represents a key visual entity. "
            "Output JSON only. Do not include any additional text."
        )
        if extra_instruction:
            base_instruction += f" Additional Information: {extra_instruction}"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a multimodal information extraction assistant. Your task is to convert image content into structured JSON."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": base_instruction},
                ],
            },
        ]

        raw_output = self._generate(messages, max_new_tokens=1024)
        json_text = extract_json_fragment(raw_output)

        caption = ""
        objects: List[str] = []

        try:
            data = json.loads(json_text.replace("```json", '').replace("```", ''))
            caption = str(data.get("caption", "")).strip()
            raw_objs = data.get("objects", [])
            if isinstance(raw_objs, list):
                for obj in raw_objs:
                    if isinstance(obj, str):
                        objects.append(obj.strip())
                    elif isinstance(obj, dict):
                        # 尝试从 name 或 label 字段读取
                        name = obj.get("name") or obj.get("label") or ""
                        if name:
                            objects.append(str(name).strip())
        except Exception:
            # TODO: json 解析逻辑增强 or 次数增加
            print("JSON解析失败: ", raw_output)
            caption = raw_output.strip()
            objects = []

        return {
            "caption": caption,
            "objects": objects,
            "raw_output": raw_output,
        }

    # Task 5: query 改写
    def rewrite_query(
        self,
        user_query: str,
        conversation_context: str,
        for_image: bool = True, # TODO: 目前 query 只是对图像检索 query 的优化
    ):
        """
        Task 5: query 改写
        """
        system_prompt = """You are a query rewriting assistant for retrieval.
            In conversations, users may ask things like
            \"help me find the photo with a white dog from earlier.\"
            Given the provided dialogue context, rewrite the user's retrieval intent
            into a concise but as specific as possible that includes
            key visual and semantic information, suitable for retrieving historical
            image semantic memories.
            For example, if the user's query is \"find a photo of a white dog\",
            the rewritten query should be \"a white dog\".
            If the user's query is \"A close-up of a shiny green apple with its stem visible.\",
            the rewritten query should be \"a shiny green apple with its stem\".
            Output strictly in JSON format: {\"rewritten_query\": \"...\"}.
            Do not output any additional content."""
        
        # NOTE: 测试的简单场景中不需要上下文消息 
        user_prompt = f"""
            Conversation context: \n{conversation_context}\n
            Original retrieval query: {user_query}\n
        """
            
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        raw_output = self._generate(messages, max_new_tokens=128)
        json_text = extract_json_fragment(raw_output)

        try:
            data = json.loads(json_text.replace("```json", '').replace("```", ''))
            rewritten = str(data.get("rewritten_query", "")).strip()
            if rewritten:
                return rewritten
        except Exception:
            pass

        return user_query