"""MiMo-V2.6-Pro 测试配置"""

# ── API 配置 ──
API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL_NAME = "mimo-v2.6-pro"
API_KEY = "sk-c2pyvify8cjtbe3njia9veg8yf77iivucjxjjf25e1mjt094"

# ── 图片转换 ──
MAX_BASE64_SIZE = 50 * 1024 * 1024   # 50MB（base64 字符串上限）
DEFAULT_DPI = 200                      # PDF 渲染分辨率
DEFAULT_QUALITY = 80                   # JPEG 压缩质量 (1-100)
DEFAULT_MAX_WIDTH = 1600               # 最大宽度（像素），超出等比缩放

# ── 测试用 PDF ──
# 改成你自己的论文路径
SAMPLE_PDF = r"d:\X-code\2026-1-科研训练计划\code\示例论文"