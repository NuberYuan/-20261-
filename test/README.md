# MiMo-V2.6-Pro 图片识别性能测试

测试小米 MiMo-V2.6-Pro 对学术论文 PDF 的视觉理解能力。

## 测试项

| 任务 | 说明 |
|------|------|
| `ocr` | 文字识别 — 提取页面全部文字 |
| `figure` | 图表理解 — 分析图表类型、趋势、结论 |
| `table` | 表格提取 — 转为 Markdown 表格 |
| `summary` | 页面摘要 — 总结核心内容 |
| `structure` | 结构化提取 — 输出 JSON 格式关键信息 |

## 安装依赖

```bash
pip install PyMuPDF Pillow requests
```

## 使用方法

```bash
# 测试单页（默认第1页，全部任务）
python test_mimo_vision.py "d:\X-code\2026-1-科研训练计划\code\示例论文\xxx.pdf"

# 指定页码和任务
python test_mimo_vision.py paper.pdf --page 2 --task ocr

# 多页摘要测试
python test_mimo_vision.py paper.pdf --multi 0 1 2 3 4

# 单独转换 PDF 为 base64（调试用）
python pdf_to_base64.py paper.pdf --page 0 --out page0.b64
```

## 输出

- 控制台实时显示识别结果
- 自动生成 `test_report.md` 测试报告

## 配置

编辑 `config.py` 修改 API Key、模型、图片参数。