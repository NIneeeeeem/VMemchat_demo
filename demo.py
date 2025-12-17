from curses import is_term_resized
import os
from typing import Any, Dict, List, Optional, Tuple
import argparse
from PIL import Image
import torch
from models import QwenVLBackend, QwenTextEmbedding, DinoV2ImageEmbedder
from memory_structs import VisualMemoryStore, VisualMemoryItem
debug = True
debug = False
if debug:
    import debugpy; debugpy.connect(('localhost', 5679))


class QwenVLMemoryChatbot:
    """
        - 存图像语义记忆 save_image_memory()
        - 文本检索 search_by_text()
        - 图像检索 search_by_image()
        - 普通多模态对话 chat()
    """

    def __init__(
        self,
        vlm_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        text_embed_path: str = "Qwen/Qwen3-Embedding-0.6B",
        visual_embed_path: str = "facebook/dinov2-large",
        device: Optional[str] = None,
        context_max_len: int = 8,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vlm = QwenVLBackend(model_name=vlm_path, device=device)
        self.embedder = QwenTextEmbedding(model_name=text_embed_path, device=device)
        self.visual_embedder = DinoV2ImageEmbedder(visual_embed_path, device=device)
        self.memory = VisualMemoryStore(self.embedder, self.visual_embedder)

        # 用于 query 改写的文本历史（只记录纯文本，避免图片）
        self.text_history: List[Tuple[str, str]] = []  # (role, text)
        # 多模态对话历史，供 Qwen3-VL chat_template 使用
        self.chat_history: List[Dict[str, Any]] = []
        self.max_len = context_max_len # 上下文窗口长度限制


    def _append_text_history(self, role: str, text: str) -> None:
        if not text:
            return
        self.text_history.append((role, text))
        if len(self.text_history) > self.max_len:
            self.text_history = self.text_history[-self.max_len:]

    def _build_text_context(self, max_turns: int = 5) -> str:
        """
        将最近若干轮纯文本对话拼接，用于 query 改写 [Task 5]
        """
        ctx_pairs = self.text_history[-max_turns:]
        lines = [f"{role}: {txt}" for role, txt in ctx_pairs]
        return "\n".join(lines)

    # Task 2. 语义记忆
    def save_image_memory(
        self,
        image_path: str,
        user_note: Optional[str] = None,
        extra_instruction: Optional[str] = None,
    ):
        """
        Pipeline:
            1. 调用 Qwen3-VL 抽取图像语义（caption + objects）。
            2. 合并语义为 summary，使用 Qwen3-Embedding 生成向量。
            3. 写入 VisualMemoryStore。
        """
        semantics = self.vlm.extract_image_semantics(
            image_path=image_path,
            extra_instruction=extra_instruction,
        )
        caption = semantics.get("caption", "")
        objects = semantics.get("objects", [])

        extra: Dict[str, Any] = {
            "user_note": user_note or "",
            "raw_output": semantics.get("raw_output", ""),
        }

        item = self.memory.add_item(
            image_path=image_path,
            caption=caption,
            key_objects=objects,
            extra=extra,
        )
        return item


    # Task 3. 文本检索
    def search_by_text(
        self,
        user_query: str,
        top_k: int = 5,
        use_rewrite: bool = True,
    ):
        """
        Returns: 
            {
                "final_query": "...",  # 可能是改写后的 query
                "results": [
                    {
                        "item": VisualMemoryItem,
                        "score": float
                    },
                    ...
                ]
            }
        """
        context = self._build_text_context()
        final_query = user_query
        if use_rewrite:
            final_query = self.vlm.rewrite_query(
                user_query=user_query,
                conversation_context=context,
                for_image=True,
            )
            print("Rewrite Query: ", final_query)

        image_results = self.memory.search_by_text(final_query, top_k=top_k)
        item_results = self.memory.search_by_text(final_query, top_k=top_k, search_level = "item")
        # TODO: 判定逻辑
        return {
            "final_query": final_query,
            "image_results": [
                {
                    "item": item,
                    "score": score,
                }
                for item, score in image_results
            ],
            "item_results": [
                {
                    "item": item,
                    "score": score,
                }
                for item, score in item_results
            ]
            
        }

    # Task 4. 图像检索
    def search_by_image(
        self,
        image_path: str,
        user_hint: Optional[str] = None,
        top_k: int = 5,
        use_rewrite: bool = True,
    ) -> Dict[str, Any]:
        """
        目标4：支持通过图像对存储的视觉物品进行检索。
        """
        # Branch 1 先提 caption，转为语义计算
        semantics = self.vlm.extract_image_semantics(image_path=image_path)
        caption = semantics.get("caption", "") or ""
        objects = semantics.get("objects", []) or []

        query_text_parts: List[str] = []
        if caption:
            query_text_parts.append(caption)
        if objects:
            query_text_parts.append("包含实体: " + "、".join(objects))
        if user_hint:
            query_text_parts.append("用户额外说明: " + user_hint)

        base_query = "; ".join(query_text_parts) or (user_hint or "来自图像的检索请求")

        context = self._build_text_context()
        final_query = base_query
        if use_rewrite:
            final_query = self.vlm.rewrite_query(
                user_query=base_query,
                conversation_context=context,
                for_image=True,
            )

        language_results = self.memory.search_by_text(final_query, top_k=top_k)
        
        # Branch 2 image embeds similarity
        visual_results = self.memory.search_by_image(image_path, top_k=top_k)
        # TODO: merge language_results and visual_results

        return {
            "base_query": base_query,
            "final_query": final_query,
            "image_semantics": semantics,
            "language_results": [
                {
                    "item": item,
                    "score": score,
                }
                for item, score in language_results
            ],
            "visual_results": [
                {
                    "item": item,
                    "score": score,
                }
                for item, score in visual_results
            ],
        }


    def chat(
        self,
        user_text: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        普通多模态对话（不自动做检索），主要用于：
            - 和用户自然聊天
            - 获得额外的对话上下文，给后面的 query 改写使用
        """
        image_paths = image_paths or []
        reply = self.vlm.chat(self.chat_history, user_text=user_text, image_paths=image_paths)

        # 更新多模态 history + 文本 history
        content: List[Dict[str, Any]] = []
        if user_text:
            content.append({"type": "text", "text": user_text})
            self._append_text_history("user", user_text)
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            content.insert(0, {"type": "image", "image": img})

        if content:
            self.chat_history.append({"role": "user", "content": content})

        self.chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": reply}]}
        )
        self._append_text_history("assistant", reply)

        max_turns = 12
        if len(self.chat_history) > 2 * max_turns:
            self.chat_history = self.chat_history[-2 * max_turns :]

        return reply


def build_parser():
    parser = argparse.ArgumentParser(
        description="Example Python script with argument parser"
    )
    parser.add_argument(
        "--vlm_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Input directory path"
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Output directory path"
    )
    parser.add_argument(
        "--visual_embed_path",
        type=str,
        default="facebook/dinov2-large",
        help="Output directory path"
    )
    return parser


def print_search_results(result_dict: Dict[str, Any]) -> None:
    def show_rank(results):
        for rank, hit in enumerate(results, start=1):
            item: VisualMemoryItem = hit["item"]
            score: float = hit["score"]
            print(f"Top {rank}  score={score:.4f}")
            print(f"  image    : {item.image_path}")
            print(f"  caption  : {item.caption}")
            if isinstance(item.key_objects, list):
                objects_str = ", ".join(item.key_objects)
            elif isinstance(item.key_objects, str):
                objects_str = item.key_objects
            else:
                objects_str = str(item.key_objects)

            print(f"objects: {objects_str}")
            note = item.extra.get("user_note", "")
            if note:
                print(f"  user_note: {note}")
                
    if "base_query" in result_dict:
        print(f"[base_query ] {result_dict['base_query']}")
    print(f"[final_query] {result_dict['final_query']}")
    print("-" * 60)
    image_results = result_dict.get("image_results", [])
    item_results = result_dict.get("item_results", [])
    language_results = result_dict.get("language_results", [])
    visual_results = result_dict.get("visual_results", [])
    if not item_results and not visual_results and not language_results and not image_results:
        print("未检索到任何结果")
        return
    if image_results:
        show_rank(image_results)
        print("="*30)
    if item_results:
        show_rank(item_results)
        print("="*30)
    if visual_results:
        print("Visual_results")
        show_rank(visual_results)
        print("="*30)
    if language_results:
        print("Language_results")
        show_rank(language_results)
        print("="*30)


def main(args):
    bot = QwenVLMemoryChatbot(args.vlm_path, args.embed_path, args.visual_embed_path)
    print(
        "\nCommands: \n"
        "  /save <img_path> [Note]      task 2. 保存图像语义到 mem \n"
        "  /search_text <query>         task 3. 文本检索\n"
        "  /search_image <img_path>     task 4. 图像检索\n"
        "  others                       chat \n"
        "  /exit                        exit \n"
    )
    # TODO: function calling, 自然语言决定 /save or /search_text or /search_image
    
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye～")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "exit", "/quit", "quit"}:
            print("\nBye～")
            break

        # ---------- /save ----------
        if user_input.startswith("/save "):
            parts = user_input.split(" ", 2)
            if len(parts) < 2:
                print("NOTE: /save <img_path> [Note]")
                continue
            image_path = parts[1]
            user_note = parts[2] if len(parts) >= 3 else None # 目前暂不支持 （不确定是否是噪声）

            if not os.path.exists(image_path):
                print(f"图像不存在：{image_path}")
                continue

            item = bot.save_image_memory(
                image_path=image_path,
                user_note=user_note,
                extra_instruction=None,
            )
            print("读取成功：")
            print(f"  image   : {item.image_path}")
            print(f"  caption : {item.caption}")
            print(f"  objects : {', '.join(item.key_objects)}")

        # ---------- /search_text ----------
        elif user_input.startswith("/search_text "):
            query = user_input[len("/search_text ") :].strip()
            if not query:
                print("NOTE: /search_text <query>")
                continue
            print("文本检索: ", query)
            # TODO: top_k, use_rewrite 传参进来
            results = bot.search_by_text(query, top_k=5, use_rewrite=True)
            print_search_results(results)

        # ---------- /search_image ----------
        elif user_input.startswith("/search_image "):
            parts = user_input.split(" ", 2)
            if len(parts) < 2:
                print("用法：/search_image <图片路径> [可选:说明文本]")
                continue
            image_path = parts[1]
            user_hint = parts[2] if len(parts) >= 3 else None # 目前暂不支持 （不确定是否是噪声）

            if not os.path.exists(image_path):
                print(f"文件不存在：{image_path}")
                continue

            print("正在从查询图片抽取语义并检索视觉记忆...")
            results = bot.search_by_image(
                image_path=image_path,
                user_hint=user_hint,
                top_k=5,
                use_rewrite=True,
            )
            print_search_results(results)

        else:
            reply = bot.chat(user_text=user_input)
            print(f"Assistant> {reply}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)