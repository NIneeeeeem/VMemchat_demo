def extract_json_fragment(text: str) -> str:
    """
    从模型输出中尽可能提取 JSON 片段。
    例如模型可能输出：
        ```json
        { ... }
        ```
    或者前后有自然语言说明，这里只取第一个 { 和 最后一个 } 之间的内容。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    # 如果没找到，就直接返回原文本，让上层处理异常
    return text