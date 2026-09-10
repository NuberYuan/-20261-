"""
项目全局配置

- 支持通过前端设置 API Key（保存到 data/settings.json）
- 支持环境变量 DEEPSEEK_API_KEY
- 所有路径基于 PROJECT_ROOT 计算，可移植
- 支持多论文目录结构

本项目使用 DeepSeek-V4-Pro (deepseek-v4-pro) 进行代码生成。
"""

import json
import os
from pathlib import Path

# =========================
# 项目根目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# DeepSeek 配置
# =========================

# DeepSeek API 使用的模型：deepseek-v4-pro (即 DeepSeek-V4-Pro-0813)
# 本项目使用 DeepSeek-V4-Pro 进行论文代码复现的代码生成

SETTINGS_FILE = PROJECT_ROOT / "data" / "settings.json"

def _load_api_key() -> str:
    """按优先级加载 API Key: 环境变量 > settings.json > 空字符串"""
    # 1. 环境变量优先
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    # 2. settings.json
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            key = settings.get("api_key", "").strip()
            if key:
                return key
        except (json.JSONDecodeError, OSError):
            pass
    # 3. 返回空字符串，允许后端启动（API Key 在调用时检查）
    return ""

def save_api_key(api_key: str):
    """保存 API Key 到 settings.json"""
    settings = {}
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    settings["api_key"] = api_key
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

API_KEY = _load_api_key()

# Chat 接口（V4 仍走同一 OpenAI 兼容地址，只换 model 名即可）
CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

# 文本模型 (DeepSeek-V4-Pro)
TEXT_MODEL_NAME = "deepseek-v4-pro"

# 请求超时 (秒)。thinking 开启后单次生成更长，过短会整次重试。
TIMEOUT = 1800

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
