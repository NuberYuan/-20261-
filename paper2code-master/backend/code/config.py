"""
项目全局配置

- 支持通过前端设置 API Key（保存到 data/settings.json）
- 支持环境变量 MIMO_API_KEY / DEEPSEEK_API_KEY
- 所有路径基于 PROJECT_ROOT 计算，可移植
- 支持多论文目录结构

当前默认模型：小米 MiMo-V2.6-Pro（可在前端切换）。
"""

import json
import os
from pathlib import Path

# =========================
# 项目根目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# 多模型配置
# =========================

MODELS = {
    "deepseek": {
        "name": "DeepSeek-V4-Pro",
        "chat_url": "https://api.deepseek.com/v1/chat/completions",
        "model_name": "deepseek-v4-pro",
        "api_key_url": "https://platform.deepseek.com",
        "supports_thinking": True,
    },
    "mimo": {
        "name": "小米 MiMo-V2.6-Pro",
        "chat_url": "https://api.xiaomimimo.com/v1/chat/completions",
        "model_name": "mimo-v2.6-pro",
        "api_key_url": "https://api.xiaomimimo.com",
        "supports_thinking": False,
    },
    "qwen": {
        "name": "通义千问 Qwen",
        "chat_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model_name": "qwen-plus",
        "api_key_url": "https://dashscope.console.aliyun.com",
        "supports_thinking": False,
    },
    "claude": {
        "name": "Claude Sonnet 4",
        "chat_url": "https://api.anthropic.com/v1/messages",
        "model_name": "claude-sonnet-4-20250514",
        "api_key_url": "https://console.anthropic.com",
        "supports_thinking": True,
        "api_format": "anthropic",
    },
    "gpt": {
        "name": "GPT-4o",
        "chat_url": "https://api.openai.com/v1/chat/completions",
        "model_name": "gpt-4o",
        "api_key_url": "https://platform.openai.com",
        "supports_thinking": False,
    },
    "gemini": {
        "name": "Gemini 2.5 Pro",
        "chat_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model_name": "gemini-2.5-pro",
        "api_key_url": "https://aistudio.google.com",
        "supports_thinking": False,
    },
    "kimi": {
        "name": "Kimi K2",
        "chat_url": "https://api.moonshot.cn/v1/chat/completions",
        "model_name": "moonshot-v1-128k",
        "api_key_url": "https://platform.moonshot.cn",
        "supports_thinking": False,
    },
}

DEFAULT_MODEL = "mimo"

SETTINGS_FILE = PROJECT_ROOT / "data" / "settings.json"

def _load_api_key() -> str:
    """按优先级加载当前模型的 API Key: 环境变量 > settings.json 按模型 > settings.json 全局 > 空字符串"""
    # 1. 环境变量优先
    env_key = os.getenv("MIMO_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            # 2. 按模型存储的 Key（api_keys.mimo / api_keys.deepseek / ...）
            model_key = settings.get("api_keys", {}).get(_load_selected_model(), "").strip()
            if model_key:
                return model_key
            # 3. 全局 Key（向后兼容）
            global_key = settings.get("api_key", "").strip()
            if global_key:
                return global_key
        except (json.JSONDecodeError, OSError):
            pass
    return ""

def save_api_key(api_key: str):
    """保存当前模型的 API Key 到 settings.json（按模型存储）"""
    save_api_key_for_model(SELECTED_MODEL, api_key)


def save_api_key_for_model(model_key: str, api_key: str):
    """保存指定模型的 API Key 到 settings.json"""
    # API Key 放入 HTTP Authorization 头，必须是 latin-1 可编码的 ASCII
    if not api_key.isascii():
        raise ValueError(f"API Key 只能包含 ASCII 字符，收到: {api_key[:20]}...")
    settings = {}
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if "api_keys" not in settings:
        settings["api_keys"] = {}
    settings["api_keys"][model_key] = api_key
    # 清理旧的全局字段（向后兼容迁移）
    settings.pop("api_key", None)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_selected_model() -> str:
    """从 settings.json 读取已选模型，未设置则返回 DEFAULT_MODEL"""
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            model_key = settings.get("selected_model", "").strip()
            if model_key and model_key in MODELS:
                return model_key
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_MODEL

def save_selected_model(model_key: str):
    """保存选中的模型到 settings.json"""
    settings = {}
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    settings["selected_model"] = model_key
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================
# 任务模型路由
# =========================

# 任务类型 → 默认模型 key
TASK_MODEL_DEFAULTS = {
    "heavy": DEFAULT_MODEL,   # Step3 信息提取、Step7 代码生成/修复（需要强模型）
    "light": DEFAULT_MODEL,   # Step5 子步骤补全（轻量模型即可）
}

def _load_task_models() -> dict:
    """从 settings.json 读取任务模型配置，未设置则用默认值"""
    result = dict(TASK_MODEL_DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            saved = settings.get("task_models", {})
            for task_type in result:
                key = saved.get(task_type, "").strip()
                if key and key in MODELS:
                    result[task_type] = key
        except (json.JSONDecodeError, OSError):
            pass
    return result

def save_task_model(task_type: str, model_key: str):
    """保存单个任务类型的模型到 settings.json"""
    if task_type not in TASK_MODEL_DEFAULTS:
        raise ValueError(f"未知任务类型: {task_type}")
    if model_key not in MODELS:
        raise ValueError(f"不支持的模型: {model_key}")
    settings = {}
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if "task_models" not in settings:
        settings["task_models"] = {}
    settings["task_models"][task_type] = model_key
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

def get_task_model_config(task_type: str) -> dict:
    """返回指定任务类型的模型完整配置。

    task_type:
        "heavy" → 复杂任务（Step3 提取、Step7 代码生成）
        "light" → 简单任务（Step5 补全）
        "default" → 当前全局选中的模型
    """
    if task_type == "default":
        return MODELS[SELECTED_MODEL]
    model_key = TASK_MODELS.get(task_type, SELECTED_MODEL)
    return MODELS[model_key]

def get_current_model_config() -> dict:
    """返回当前选中模型的完整配置（chat_url, model_name 等）"""
    return MODELS[SELECTED_MODEL]

API_KEY = _load_api_key()
SELECTED_MODEL = _load_selected_model()
TASK_MODELS = _load_task_models()

# 兼容别名：指向当前选中模型的值（供已有代码使用）
_model_cfg = MODELS[SELECTED_MODEL]
CHAT_URL = _model_cfg["chat_url"]
TEXT_MODEL_NAME = _model_cfg["model_name"]

# 请求超时 (秒)。thinking 开启后单次生成更长，过短会整次重试。
# 实测单次代码生成最长约 12 分钟，900s 既不误杀慢调用，也能在挂起时较快失败重试。
TIMEOUT = 900

# LLM 调用最大重试次数
MAX_RETRIES = 3

# 重试间隔基数 (秒)
RETRY_BASE_DELAY = 2

# =========================
# 目录结构
# =========================

DATA_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
LOG_DIR = PROJECT_ROOT / "logs"


def get_paper_output_dir(paper_name: str) -> Path:
    """按论文名称获取输出目录"""
    return DATA_OUTPUT_DIR / paper_name


def get_step5_output_dir(paper_name: str) -> Path:
    """按论文名称获取 Step5 输出目录"""
    return get_paper_output_dir(paper_name) / "step5"


def get_step1_output_dir(paper_name: str) -> Path:
    """按论文名称获取 Step1 输出目录"""
    return get_paper_output_dir(paper_name) / "step1"


def get_step2_output_dir(paper_name: str) -> Path:
    """按论文名称获取 Step2 输出目录"""
    return get_paper_output_dir(paper_name) / "step2"


def get_step3_output_dir(paper_name: str) -> Path:
    """按论文名称获取 Step3 输出目录"""
    return get_paper_output_dir(paper_name) / "step3"


# =========================
# 默认论文名
# =========================

# 默认论文目录 = data/output/plant/
DEFAULT_PAPER = "plant"

# =========================
# 自动检测 PDF
# =========================

def _find_default_pdf() -> str:
    """自动查找 data/input/ 下的第一个 PDF"""
    if DATA_INPUT_DIR.exists():
        pdfs = list(DATA_INPUT_DIR.glob("*.pdf"))
        if pdfs:
            return str(pdfs[0])
    return str(DATA_INPUT_DIR / "put_in.pdf")


# =========================
# Step1 输入/输出
# =========================

# 输入 PDF（自动检测 data/input/ 下的 PDF）
INPUT_PDF = _find_default_pdf()

# 默认输出到 data/output/{DEFAULT_PAPER}/
_paper_out = DATA_OUTPUT_DIR / DEFAULT_PAPER

# Step1 输出
OUTPUT_JSON = str(_paper_out / "step1" / "raw_paper.json")

# 图片提取目录
IMAGE_OUTPUT_DIR = str(_paper_out / "step1" / "images")

# 表格提取目录
TABLE_OUTPUT_DIR = str(_paper_out / "step1" / "tables")

# =========================
# Step2 输出
# =========================

# Step2.1
CLEANED_JSON = str(_paper_out / "step2" / "cleaned_paper.json")

# Step2.2
REPAIRED_JSON = str(_paper_out / "step2" / "repaired_paper.json")

# Step2.3
STRUCTURE_DETECT = str(_paper_out / "step2" / "structure_detect_paper.json")

# =========================
# Step3 输出
# =========================

KEY_INFO_JSON = str(_paper_out / "step3" / "key_information.json")

# =========================
# 便捷函数：按论文名获取路径
# =========================


def get_paper_paths(paper_name: str) -> dict:
    """
    按论文名返回所有相关路径。

    用于支持 data/output/{paper_name}/ 的多论文目录结构。

    Returns:
        dict 包含 input_pdf 外的所有 Step1-3 输出路径
    """
    step1 = get_step1_output_dir(paper_name)
    step2 = get_step2_output_dir(paper_name)
    step3 = get_step3_output_dir(paper_name)
    return {
        "raw_json": str(step1 / "raw_paper.json"),
        "cleaned_json": str(step2 / "cleaned_paper.json"),
        "repaired_json": str(step2 / "repaired_paper.json"),
        "structure_detect": str(step2 / "structure_detect_paper.json"),
        "key_information": str(step3 / "key_information.json"),
        "images_dir": str(step1 / "images"),
        "tables_dir": str(step1 / "tables"),
    }
