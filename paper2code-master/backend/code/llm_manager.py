"""
llm_manager.py — 统一 LLM 调用管理

一站式接口：模型路由 + 调用 + 重试 + 缓存 + JSON 解析 + 代码提取。

用法:
    from llm_manager import llm_call, parse_llm_json, extract_code_block

    # 调用 LLM
    raw = llm_call("你好", task_type="heavy")

    # 解析 JSON 响应
    data = parse_llm_json(raw)

    # 提取代码块
    code = extract_code_block(raw)

设计原则:
    - 唯一入口：llm_call() 替代所有直接调用 chat_with_text()
    - 唯一出口：parse_llm_json() / extract_code_block() 替代各 step 的私有解析
    - llm_utils.py 降级为私有 HTTP 层，不再对外暴露
"""

import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import requests

import config as _config

logger = logging.getLogger("llm_manager")


# =========================
# 公共接口
# =========================

def llm_call(prompt: str, task_type: str = "default", *,
             system: str = "You are a helpful assistant.",
             timeout: int = None,
             max_retries: int = None) -> str:
    """
    统一 LLM 调用入口。

    Args:
        prompt: 用户 prompt
        task_type: 任务类型
            - "default" → 当前全局选中的模型
            - "heavy"   → 复杂任务（Step3 提取、Step7 代码生成）
            - "light"   → 简单任务（Step5 补全）
        system: 系统 prompt
        timeout: 超时秒数（默认用 config.TIMEOUT）
        max_retries: 最大重试次数（默认用 config.MAX_RETRIES）

    Returns:
        LLM 返回的文本字符串

    Raises:
        RuntimeError: 超过最大重试次数后抛出
    """
    model_cfg = _config.get_task_model_config(task_type)
    model_key = _resolve_model_key(task_type)

    timeout = timeout or _config.TIMEOUT
    max_retries = max_retries or _config.MAX_RETRIES

    headers = _build_headers(model_cfg, model_key=model_key)
    body = _build_body(prompt, system, model_cfg)
    url = model_cfg["chat_url"]

    return _http_post_with_retry(
        headers=headers, body=body,
        url=url, timeout=timeout,
        max_retries=max_retries,
        model_key=model_key,
        system=system,
    )


def llm_vision_call(prompt: str, image_path: str = None, *,
                    image_base64: str = None,
                    mime_type: str = "image/png",
                    system: str = "You are a helpful assistant.",
                    task_type: str = "heavy",
                    timeout: int = None,
                    max_retries: int = None) -> str:
    """
    视觉 LLM 调用入口：支持传入图片（base64）+ 文本 prompt。

    Args:
        prompt: 文本 prompt
        image_path: 图片文件路径（自动转 base64）
        image_base64: 已编码的 base64 图片字符串（与 image_path 二选一）
        mime_type: 图片 MIME 类型（image/png / image/jpeg 等）
        system: 系统 prompt
        task_type: 任务类型（默认 heavy）
        timeout: 超时秒数
        max_retries: 最大重试次数

    Returns:
        LLM 返回的文本字符串

    Raises:
        RuntimeError: 超过最大重试次数后抛出
        ValueError: 未提供图片或图片编码失败
    """
    import base64 as _b64

    # 获取 base64 图片数据
    if image_base64:
        b64_data = image_base64
    elif image_path:
        try:
            with open(image_path, "rb") as f:
                b64_data = _b64.b64encode(f.read()).decode("ascii")
        except OSError as e:
            raise ValueError(f"无法读取图片文件: {image_path}: {e}") from e
    else:
        raise ValueError("必须提供 image_path 或 image_base64")

    model_cfg = _config.get_task_model_config(task_type)
    model_key = _resolve_model_key(task_type)

    timeout = timeout or _config.TIMEOUT
    max_retries = max_retries or _config.MAX_RETRIES

    headers = _build_headers(model_cfg, model_key=model_key)
    url = model_cfg["chat_url"]

    # OpenAI 兼容的 vision 消息格式
    data_url = f"data:{mime_type};base64,{b64_data}"
    body = {
        "model": model_cfg["model_name"],
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "max_tokens": 16384,
        "temperature": 0,
        "stream": False,
    }

    return _http_post_with_retry(
        headers=headers, body=body,
        url=url, timeout=timeout,
        max_retries=max_retries,
        model_key=model_key,
        system=system,
    )


def parse_llm_json(raw: str) -> dict:
    """
    从 LLM 输出中提取 JSON 对象。

    三步尝试：
    1. 直接 json.loads
    2. 去掉 markdown 代码块后解析
    3. 括号深度扫描提取第一个完整 JSON

    Returns:
        解析成功返回 dict，失败返回 {"raw_response": ..., "error": ...}
    """
    raw = raw.strip()

    # 1. 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. 去掉 markdown 代码块
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 括号深度扫描
    start = raw.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except json.JSONDecodeError:
                        break

    logger.warning("Failed to parse LLM response as JSON")
    return {
        "raw_response": raw[:2000],
        "error": "JSON parse failed after all attempts",
    }


def extract_code_block(raw: str) -> str:
    """
    从 LLM 输出中提取代码块。

    处理：
    - markdown 代码块包裹（支持任意语言标签）
    - 多个代码块时选择最像完整文件的
    - 无代码块时直接返回原文

    Returns:
        提取到的代码字符串
    """
    raw = raw.strip()

    blocks = re.findall(r"```[a-zA-Z0-9_+.-]*\s*\n(.*?)```", raw, re.DOTALL)
    if blocks:
        if len(blocks) == 1:
            return blocks[0].strip()
        # 多个代码块时，优先选看起来像完整文件的
        def _score(block: str) -> tuple:
            has_import = bool(re.search(r"^import |^from ", block, re.MULTILINE))
            has_class = bool(re.search(r"^class ", block, re.MULTILINE))
            has_def = bool(re.search(r"^def ", block, re.MULTILINE))
            return (int(has_import) + int(has_class) + int(has_def), len(block))
        return max(blocks, key=_score).strip()

    # 处理被截断的代码块（有开头 ``` 但没有结尾 ```）
    truncated = re.search(r"```[a-zA-Z0-9_+.-]*\s*\n(.+)", raw, re.DOTALL)
    if truncated:
        content = truncated.group(1).strip()
        if content:
            return content

    return raw


# =========================
# 内部实现（HTTP 层）
# =========================

def _resolve_model_key(task_type: str) -> str:
    """解析 task_type 对应的 model key"""
    if task_type == "default":
        return _config.SELECTED_MODEL
    return _config.TASK_MODELS.get(task_type, _config.SELECTED_MODEL)


def _get_api_key_for_model(model_key: str = None) -> str:
    """按模型查找 API Key。优先级：settings.json 按模型 > 全局 Key > 环境变量"""
    if model_key:
        try:
            settings_path = _config.SETTINGS_FILE
            if settings_path.exists():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                key = settings.get("api_keys", {}).get(model_key, "").strip()
                if key:
                    return key
        except (json.JSONDecodeError, OSError):
            pass
    # 回退到全局 Key
    return _config.API_KEY


def _build_headers(model_cfg: dict, model_key: str = None) -> dict:
    """构建请求头，按模型查找对应的 API Key"""
    api_key = _get_api_key_for_model(model_key)
    if model_cfg.get("api_format") == "anthropic":
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _build_body(prompt: str, system: str, model_cfg: dict) -> dict:
    """构建请求体"""
    model_name = model_cfg["model_name"]

    if model_cfg.get("api_format") == "anthropic":
        return {
            "model": model_name,
            "max_tokens": 16384,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }

    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 16384,
        "temperature": 0,
        "stream": False,
    }


def _strip_markdown_fences(text: str) -> str:
    """移除 ```json ... ``` 或 ``` ... ``` 包裹"""
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _cache_path(prompt: str, model_key: str, system: str = "") -> Optional[Path]:
    """
    计算缓存文件路径。
    缓存 key 包含 model_key + system prompt + prompt，避免不同配置缓存串了。
    """
    content = f"{model_key}:{system}:{prompt}"
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    cache_dir = _config.PROJECT_ROOT / "data" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.txt"


def _http_post_with_retry(headers: dict, body: dict,
                          url: str, timeout: int,
                          max_retries: int,
                          model_key: str,
                          system: str = "") -> str:
    """带重试和缓存的 HTTP POST。"""
    # 检查缓存
    cache_file = _cache_path(str(body), model_key, system=system)
    if cache_file and cache_file.exists():
        try:
            cached = cache_file.read_text(encoding="utf-8")
            logger.info(f"[LLM] cache hit ({model_key})")
            return cached
        except OSError:
            pass

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"[LLM] request ({model_key}) attempt={attempt+1}")
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # 提取文本
            text = _extract_response_text(data, url)

            # 空响应视为可重试错误：不返回空串、不写缓存（否则下游把
            # 「内容为空」当最终结果落盘，如生成文件为空被校验器记为失败）
            if not text.strip():
                raise RuntimeError("LLM returned empty response")

            # 写缓存
            if cache_file:
                try:
                    cache_file.write_text(text, encoding="utf-8")
                except OSError:
                    pass

            return text

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            is_rate_limit = status == 429
            is_server_error = (status or 0) >= 500

            if is_rate_limit or is_server_error:
                retry_after = _parse_retry_after(e.response)
                delay = retry_after if retry_after > 0 else min(2 ** attempt, 60)
                logger.warning(f"[LLM] HTTP {status}, retry in {delay}s ({model_key})")
                time.sleep(delay)
                last_error = e
                continue

            raise RuntimeError(f"LLM API error {status}: {e}") from e

        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(2 ** attempt, 60)
                logger.warning(f"[LLM] error: {e}, retry in {delay}s ({model_key})")
                time.sleep(delay)
                last_error = e
                continue

            raise RuntimeError(f"LLM call failed after {max_retries} retries: {e}") from e

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def _extract_response_text(data: dict, url: str) -> str:
    """从 API 响应中提取文本，兼容 Anthropic 和 OpenAI 格式。"""
    # Anthropic Messages API
    if "content" in data and isinstance(data["content"], list):
        parts = []
        for block in data["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "\n".join(parts)

    # OpenAI-compatible API
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        pass

    raise RuntimeError(f"Unexpected LLM response format: {str(data)[:500]}")


def _parse_retry_after(response) -> float:
    """解析 Retry-After 头（秒）"""
    try:
        val = (response.headers or {}).get("Retry-After", "")
        if not val:
            return 0.0
        return max(0.0, float(val))
    except (ValueError, TypeError):
        return 0.0