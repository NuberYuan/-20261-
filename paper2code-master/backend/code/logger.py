"""
日志模块

- 懒初始化：只有第一次 getLogger 调用时才创建 log 目录和 handler
- 避免 import 时就创建目录导致权限错误
- 支持多个不同名称的 logger，共享同一日志文件
"""

import logging
import os
from pathlib import Path

_initialized = False
_loggers = {}
_file_handler = None
_console_handler = None


def _init_handlers():
    """初始化共享的 handler（只执行一次）。"""
    global _initialized, _file_handler, _console_handler
    if _initialized:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # 日志目录 — 基于项目根目录
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(
            str(log_dir / "app.log"),
            encoding="utf-8",
        )
        _file_handler.setFormatter(formatter)
    except (OSError, PermissionError):
        # 如果无法写文件日志，只输出到控制台
        pass

    # 控制台输出
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(formatter)

    _initialized = True


def get_logger(name: str = "paper_reproduction") -> logging.Logger:
    """获取指定名称的 logger（懒初始化，共享 handler）。"""
    if name in _loggers:
        return _loggers[name]

    _init_handlers()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if _file_handler is not None:
        logger.addHandler(_file_handler)
    if _console_handler is not None:
        logger.addHandler(_console_handler)

    _loggers[name] = logger
    return logger


# 向后兼容：模块级 logger 实例
logger = get_logger()
