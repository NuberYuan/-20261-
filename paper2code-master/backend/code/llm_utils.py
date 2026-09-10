"""
DeepSeek API 调用工具

- 文本模型 (deepseek-v4-pro)
- 自动重试 + 指数退避
- 统一错误处理
- 对「返回空内容 / 非 JSON / 结构缺失」这类 deepseek 偶发「无法返回」情况也做重试
"""

import email.utils
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import (
    API_KEY,
    CHAT_URL,
    TEXT_MODEL_NAME,
    TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    PROJECT_ROOT,
)

from logger import logger


# =================================
# 通用请求逻辑
# =================================


def _extract_content_safe(response: requests.Response):
    """从响应中安全提取文本内容。

    结构缺失（choices 为空）、非 JSON、内容不是字符串时返回 None，
    由调用方把这些情况当作可重试的「无法返回」处理。
    """
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _retry_after_seconds(response: requests.Response):
    """解析 Retry-After 头，返回建议等待秒数；无该头或无法解析返回 None。

    支持两种形式:
        - 秒数:   Retry-After: 120
        - 日期:   Retry-After: Fri, 31 Dec 1999 23:59:59 GMT
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _chat_with_retry(headers: dict, body: dict, timeout: int = TIMEOUT) -> str:
    """POST 请求 + 内容提取，带重试。

    可恢复情况（会重试，最多 MAX_RETRIES 次）:
        - 429 限流（按 Retry-After 退避后重试）
        - 5xx 服务端错误
        - 请求超时 / 连接失败 / 其它请求异常
        - 200 但内容为空 / 非 JSON / 结构缺失（deepseek 偶发「无法返回」）

    其它 4xx 客户端错误（认证失败、参数错误等）不重试，直接抛出。

    Raises:
        RuntimeError: 所有重试均失败
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        wait_after = None
        try:
            response = requests.post(
                CHAT_URL,
                headers=headers,
                json=body,
                timeout=timeout,
            )

            if response.status_code == 200:
                content = _extract_content_safe(response)
                if isinstance(content, str) and content.strip():
                    return content
                # 200 但无有效文本 —— 视为「无法返回」，重试
                logger.warning(
                    f"DeepSeek 返回空内容/无效文本，第 {attempt}/{MAX_RETRIES} 次尝试"
                )
                last_error = RuntimeError("DeepSeek 返回空内容（无法返回）")

            elif response.status_code == 429:
                # 限流：按 Retry-After 退避后重试，不视为永久错误
                wait_after = _retry_after_seconds(response)
                logger.warning(
                    f"DeepSeek 限流 (429)，第 {attempt}/{MAX_RETRIES} 次尝试"
                    + (f"，Retry-After={wait_after:.1f}s" if wait_after else "")
                )
                last_error = RuntimeError(
                    f"DeepSeek 限流 (429): {response.text[:200]}"
                )

            elif 400 <= response.status_code < 500:
                # 其它 4xx（认证失败、参数错误等）不重试
                raise RuntimeError(
                    f"DeepSeek API 客户端错误 ({response.status_code}): {response.text}"
                )

            else:
                # 5xx 及其它服务端错误重试
                wait_after = _retry_after_seconds(response)
                logger.warning(
                    f"DeepSeek API 服务端错误 ({response.status_code})，"
                    f"第 {attempt}/{MAX_RETRIES} 次尝试"
                )
                last_error = RuntimeError(
                    f"DeepSeek 错误 ({response.status_code}): {response.text}"
                )

        except requests.exceptions.Timeout:
            logger.warning(
                f"DeepSeek API 超时，第 {attempt}/{MAX_RETRIES} 次尝试"
            )
            last_error = TimeoutError(f"DeepSeek API 请求超时 ({timeout}s)")

        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"DeepSeek API 连接失败，第 {attempt}/{MAX_RETRIES} 次尝试: {e}"
            )
            last_error = e

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"DeepSeek API 请求异常，第 {attempt}/{MAX_RETRIES} 次尝试: {e}"
            )
            last_error = e

        # 最后一次不 sleep
        if attempt < MAX_RETRIES:
            if wait_after is not None:
                # 服务端明确告知的等待时间（限流等），上限 60s 防止过度等待
                time.sleep(min(wait_after, 60.0))
            else:
                time.sleep(RETRY_BASE_DELAY ** attempt)

    raise RuntimeError(
        f"DeepSeek API 调用失败，已重试 {MAX_RETRIES} 次"
    ) from last_error


# =================================
# 文本模型
# =================================


# =================================
# 响应缓存
# =================================

# 环境变量 PAPER2CODE_LLM_CACHE=0 可关闭缓存
CACHE_ENABLED = os.getenv("PAPER2CODE_LLM_CACHE", "1") == "1"
_CACHE_DIR = PROJECT_ROOT / "data" / ".llm_cache"
_CACHE_MAX_FILES = 500  # 缓存文件数上限


def _cache_path(prompt: str) -> Path:
    """按 (模型 + prompt) 哈希得到缓存文件路径。

    temperature=0 下相同 prompt 结果确定，可安全复用；
    系统提示词是常量，因此只对 prompt 做哈希即可。
    """
    key = hashlib.sha256((TEXT_MODEL_NAME + "\n" + prompt).encode("utf-8")).hexdigest()
    return _CACHE_DIR / (key + ".txt")


def _evict_cache():
    """缓存文件数超过上限时，删除最旧的文件。"""
    try:
        files = sorted(_CACHE_DIR.glob("*.txt"), key=lambda f: f.stat().st_mtime)
        if len(files) > _CACHE_MAX_FILES:
            for f in files[: len(files) - _CACHE_MAX_FILES]:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def chat_with_text(prompt: str) -> str:
    """
    调用 DeepSeek 文本模型。

    Args:
        prompt: 用户提示词

    Returns:
        模型回复文本

    Raises:
        RuntimeError: API 调用失败（含多次「无法返回」后仍失败）
    """
    cache_file = _cache_path(prompt)
    if CACHE_ENABLED and cache_file.exists():
        try:
            cached = cache_file.read_text(encoding="utf-8")
            logger.info("DeepSeek Text 命中缓存")
            return cached
        except OSError:
            pass

    logger.info("调用 DeepSeek Text")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": TEXT_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert academic researcher.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # thinking 关闭：直接输出最终 content，节省 token 和费用
        "thinking": {"type": "disabled"},
        "temperature": 0.7,
        "temperature": 0,
    }

    content = _chat_with_retry(headers, body)

    if CACHE_ENABLED:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(content, encoding="utf-8")
            _evict_cache()
        except OSError:
            logger.debug("写入 LLM 缓存失败（忽略）")

    logger.info("DeepSeek Text 完成")
    return content
