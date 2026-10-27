"""
llm_utils.py — 后向兼容垫片

所有功能已迁移到 llm_manager.py。
本文件仅保留 chat_with_text() 供未迁移的代码使用。

新代码请直接:
    from llm_manager import llm_call, parse_llm_json, extract_code_block
"""

import warnings

from llm_manager import llm_call


def chat_with_text(prompt: str, task_type: str = "default", **kwargs) -> str:
    """后向兼容接口，内部转发到 llm_manager.llm_call()"""
    warnings.warn(
        "chat_with_text() 已废弃，请改用 llm_manager.llm_call()",
        DeprecationWarning, stacklevel=2,
    )
    return llm_call(prompt, task_type=task_type, **kwargs)