# Paper2Code 改进清单

> 基于全量代码审查，按优先级排列。每项标注影响范围和预估工作量。

---

## P0 — 必须修（安全 / 崩溃）

### 1. 前端 XSS 注入
- **位置**: `frontend/index.html` 的 `renderList()` 函数
- **问题**: `onclick="startProcess('${paperName}')"` 直接拼接用户可控的论文名，含引号或脚本即可注入。`addLog()` 用 `innerHTML` 未转义服务端返回内容。
- **修复**: 所有 onclick 改用 `data-` 属性 + `addEventListener`；`addLog` 转义 HTML。

### 2. 后端路径穿越
- **位置**: `api_server.py` 大量 `/<paper_name>` 路由
- **问题**: 只有 `delete_paper` 做了路径穿越校验，其余路由直接拼接 `DATA_OUTPUT_DIR / paper_name`，可读取任意文件。
- **修复**: 抽取统一的 `_validate_paper_name()` 函数，所有路由入口调用。

### 3. LLM 调用无异常处理（5 处 Critical）
- **位置**: `step3/information_extractor.py:79`、`step3/figure_code_generator.py:52`、`step7/code_generator.py:18`、`step7/code_repairer.py:39`
- **问题**: `chat_with_text()` 失败后无 try/except，直接崩掉整个 pipeline。对比 `table_analyzer.py` 有完善的 fallback。
- **修复**: 每个 LLM 调用点加 try/except，返回错误标记而非崩溃。

### 4. 贪婪正则提取 JSON 失败
- **位置**: `step3/information_extractor.py:117`
- **问题**: `r'\{.*\}'` + `re.DOTALL` 是贪婪匹配，多个 JSON 对象时会把中间的无效文本也吃进去。
- **修复**: 改用括号深度扫描或非贪婪匹配。

### 5. Step5.8 绕过三层防幻觉
- **位置**: `step5/step5_8_spec.py:135`
- **问题**: `SpecAnalyzer.analyze()` 完全覆盖基类，跳过了 `_validate()`（Layer 3）和 `_find_unknown()` / `_merge()`（Layer 2）。
- **修复**: LLM 补全后调用 `self._validate(result)`。

---

## P1 — 应该修（可靠性 / 架构）

### 6. ~~JSON 解析逻辑重复且不一致~~ ✅ 已完成
- **位置**: `information_extractor`、`table_analyzer`、`step5_8_spec`、`base_analyzer` 各自实现
- **问题**: 4 种不同的 JSON 解析策略，最健壮的（3 步尝试）只在 `information_extractor` 里。
- **修复**: 已统一到 `llm_manager.parse_llm_json()`，3 步尝试（直接解析 → 去 fences → 括号深度扫描）。所有 step 已迁移。

### 7. Express 中间层多余
- **位置**: `frontend/server.js`
- **问题**: 纯代理，每个请求转发到 Flask。两跳延迟、两个进程要管理、静态文件暴露服务端代码。Flask 已有 CORS。
- **修复**: 直接让浏览器请求 Flask（改端口 5000），或在 Flask 里 serve 静态文件。只保留中文文件名修复逻辑。

### 8. 前端 fetch 全部缺 `res.ok` 检查
- **位置**: 所有 `fetch` 调用
- **问题**: 服务端 500 返回非 JSON 时 `res.json()` 直接抛异常，用户看到空白。
- **修复**: 每个 fetch 后加 `if (!res.ok) throw`。

### 9. settings.json 读写竞态
- **位置**: `config.py` 6 个函数
- **问题**: Flask 多线程并发读写 `settings.json`，无文件锁。
- **修复**: 加 `threading.Lock()` 或用 `filelock` 库。

### 10. 子进程超时不杀进程
- **位置**: `step8/experiment_runner.py:185-213`
- **问题**: `subprocess.run(timeout=...)` 超时后只抛异常，不 kill 子进程，Linux 上成为孤儿进程。
- **修复**: 用 `Popen` + 进程组管理，超时时 `os.killpg()`。

### 11. 无认证
- **位置**: Flask 绑定 `0.0.0.0:5000`
- **问题**: 同网络任何人可删论文、改 API Key、触发昂贵 LLM 调用。
- **修复**: 绑定 `127.0.0.1`；如需远程访问，加 token 认证。

### 12. 模块级常量 import 后不更新
- **位置**: `config.py` 的 `API_KEY`、`SELECTED_MODEL` 等
- **问题**: `from config import API_KEY` 拿到的是 import 时的值快照，运行时改了 config 但 import 方是值拷贝。
- **修复**: 统一用 `import config as _config` + `_config.API_KEY` 访问，不用 `from config import X`。

---

## P2 — 值得改（质量 / 体验）

### 13. ~~LLM 调用逻辑分散~~ ✅ 已完成
- **位置**: `llm_utils.py` 管调用，`config.py` 管路由，各 step 管后处理
- **问题**: 职责不清，每个 step 各自写 JSON 解析、代码块提取、错误处理。
- **修复**: 已创建 `llm_manager.py`，提供 `llm_call()` / `parse_llm_json()` / `extract_code_block()` 一站式接口。`llm_utils.py` 降级为后向兼容垫片。

### 14. 前端按钮无 loading 状态
- **位置**: 上传、处理、删除、清空、下一步按钮
- **问题**: 可重复点击，无禁用/加载态。
- **修复**: 点击后 disabled + 显示 spinner。

### 15. 单论文处理状态
- **位置**: `frontend/index.html` 的 `processingId` 全局变量
- **问题**: 只能跟踪一篇论文的处理状态，多篇并发时状态错乱。
- **修复**: 改为 `Set` 或 Map，按 paper_name 独立跟踪。

### 16. 轮询逻辑竞态
- **位置**: `startPolling` / `startPollingForNextStep` / `loadData` 互相调用
- **问题**: 多个定时器可同时存在，`loadData` 每 30 秒自动刷新可能重复启动轮询。
- **修复**: 统一一个轮询管理器，所有状态变更通过它协调。

### 17. 上传无文件大小限制
- **位置**: `api_server.py`
- **问题**: 无 `MAX_CONTENT_LENGTH`，可上传任意大小文件。
- **修复**: 设置 `app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024`。

### 18. atexit 清空 API Key
- **位置**: `api_server.py` 的 `_clear_api_key_on_exit`
- **问题**: 每次关服务都清空 `settings.json` 里的 key，与 `persist=True` 矛盾。
- **修复**: 去掉 atexit 清理，或改为可配置。

### 19. `_dir_size` 每次全量遍历
- **位置**: `temp_file_manager.py:45-58`
- **问题**: 每次调用都递归遍历所有文件，论文多时很慢。
- **修复**: 缓存结果（TTL 30 秒）或惰性计算。

### 20. 日志读取全量加载
- **位置**: `api_server.py` 的 `get_logs`
- **问题**: `f.readlines()` 读整个日志文件，长时间运行后几百 MB。
- **修复**: 只读最后 N 行（`collections.deque(f, maxlen=1000)`）。

---

## P3 — 锦上添花

### 21. Token 用量统计
- 每次 LLM 调用返回的 `usage` 字段可累加，前端展示总消耗和费用估算。
- 可按论文、按步骤、按模型分别统计，帮助用户选择性价比最优的模型组合。

### 22. WebSocket 替代轮询
- 当前 3 秒轮询浪费资源。用 WebSocket 或 SSE 实时推送状态。
- 优势：延迟低（毫秒级）、省带宽、可推送日志流。

### 23. 多模态模型支持
- Step3 图表分析当前是 OCR → LLM 生成代码 → 沙箱执行，链路长且不稳定。用多模态 LLM（如 Claude Vision、GPT-4o）直接看图一步提取。
- 可作为 `task_type="vision"` 新增，与 heavy/light 并列。

### 24. Step9-12：验证闭环
- 当前 Step8 执行完就结束，没有验证结果是否复现了论文。需要：
  - Step9: 执行结果 vs 论文结果数值对比（提取论文中的 baseline 数字，与生成代码的输出做差值计算）
  - Step10: 基于验证报告的迭代修复（精准定位失败文件，只重生成有问题的部分）
  - Step11: 复现性评分报告（结构复现度 / 数值复现度 / 可运行性 / 完整性 四维评分）

### 25. Docker 部署
- 当前只有 Windows `start.bat`，无 Linux/macOS 脚本。加 `Dockerfile` + `docker-compose.yml`。
- 一个容器跑 Flask，一个容器跑前端（nginx），或合并在一个容器里。

### 26. 无障碍
- 无 ARIA 标签、无焦点管理、颜色对比度不足（`--text-muted` #8e95a9 在浅色底上 < 4.5:1）。
- 步骤进度条无 `role="progressbar"` / `aria-valuenow`，屏幕阅读器无法感知进度。

### 27. 前端工程化
- 单文件 1800+ 行 HTML，内联 CSS/JS。可拆分为独立文件，加构建流程。
- 外部依赖（Google Fonts、JSZip CDN）无 fallback，CDN 挂了功能直接不可用。

---

## P4 — 代码质量

### 28. ~~重复代码消除~~ ✅ 大部分已完成
- **`_extract_code`** → 已统一到 `llm_manager.extract_code_block()`，原文件中的重复实现已删除。
- **JSON 解析** → 已统一到 `llm_manager.parse_llm_json()`（见 #6）。
- **`import re`** 在 `base_analyzer.py` 方法体内重复导入。
- **修复**: `llm_manager.py` 同时解决了 JSON 解析和代码提取的重复问题。

### 29. 缺少类型提示
- 全项目无 type hint。`chat_with_text` 返回 `str`，但各 step 期望 `str | dict`。
- **修复**: 关键函数加 type hint，至少 `llm_utils.py`、`config.py`、`pipeline_controller.py`。

### 30. 日志格式不统一
- 有的用 `logger.info(f"...")`，有的用 `logger.info("... %s", var)`。
- Step 日志无结构化字段（paper_name、step_name），导致 `get_logs` 只能靠子串过滤。
- **修复**: 统一用 f-string，关键日志加 `extra={"paper": paper_name, "step": step_name}`。

### 31. Prompt 语言混杂
- Step5/Step7 的 prompt 是中文，Step3 table_analyzer 是英文，`llm_utils.py` 系统消息是英文。
- 混杂语言可能导致 LLM 在 JSON 输出中夹杂非预期语言。
- **修复**: 统一 prompt 语言（建议中文，因为论文多为中文），或让系统消息跟随 prompt 语言。

### 32. `IMPORT_TO_REQ` 映射不完整
- **位置**: `step7/code_validator.py:77-90`
- 缺少 `np` → `numpy`、`plt` → `matplotlib`、`cv2` → `opencv-python` 等常见别名。
- **修复**: 补充映射表，或自动检测（import 名 ≠ pip 名时查表）。

### 33. 缓存 key 未包含 system prompt
- **位置**: `llm_utils.py` 的 `_cache_path`
- 缓存 key 只哈希了 model_name + prompt，不含 system prompt。如果 system prompt 变了，旧缓存会返回错误结果。
- **修复**: 把 system prompt 也纳入哈希，或加缓存版本号。

### 34. 指数退避无上限
- **位置**: `llm_utils.py:165`
- `time.sleep(2 ** attempt)` 无上限。如果 `MAX_RETRIES` 调大，等待时间指数爆炸。
- **修复**: 加 `min(2 ** attempt, 60)` 上限。

---

## P5 — 测试

### 35. 零测试覆盖
- 项目无任何测试代码（`step3/tests/` 目录为空的 `__init__.py`）。
- **优先写**:
  - `test_llm_utils.py`: mock LLM 返回，测试重试、缓存、JSON 解析
  - `test_pipeline_controller.py`: 测试步骤推进、断点续跑、并发锁
  - `test_config.py`: 测试 settings.json 读写、模型切换
  - `test_api.py`: 测试所有 API 端点的正常/异常路径

### 36. 缺少集成测试
- 各 step 之间没有端到端测试。改了 step3 可能悄悄破坏 step7。
- **修复**: 用一个小型测试论文跑完整 pipeline，对比各 step 输出是否符合预期。

### 37. LLM 输出质量无回归测试
- 每次改 prompt 都不知道输出是变好了还是变差了。
- **修复**: 收集 3-5 个论文样本的期望输出，作为 golden test，改 prompt 后自动对比。

---

## P6 — 可观测性

### 38. 无结构化日志
- 当前日志是纯文本，`get_logs` 靠子串过滤论文名。
- **修复**: 用 JSON 格式日志，每条带 `paper_name`、`step`、`level`、`duration_ms` 字段。

### 39. 无 LLM 调用指标
- 不知道每个 step 调了几次 LLM、花了多少钱、用了多少 token。
- **修复**: 在 `chat_with_text` 返回值中附带 `usage` 统计，累加到全局计数器。

### 40. 无 pipeline 执行耗时统计
- 不知道哪个 step 最慢、总耗时多少。
- **修复**: 每个 step 记录开始/结束时间，写入 status 文件，前端展示瀑布图。

### 41. 无错误追踪
- LLM 调用失败、子进程崩溃等错误只写日志，无告警机制。
- **修复**: 关键错误通过 webhook 通知（可选），或在前端弹出持久化错误通知。

---

## P7 — 功能扩展

### 42. 论文对比功能
- 用户上传多篇论文，系统自动对比它们的方法、数据集、结果。
- 适合做文献综述或方法选型。

### 43. 批量处理
- 当前一次只能处理一篇论文。支持文件夹批量上传，排队依次处理。
- 前端加队列视图，显示等待中/处理中/已完成数量。

### 44. 导出格式扩展
- 当前只导出 ZIP。可加：
  - 导出为 Jupyter Notebook（可交互运行）
  - 导出为 LaTeX 实验报告
  - 导出为 Markdown README

### 45. 人工干预点
- 在 Step7 代码生成前暂停，让用户审查/修改 `project_brief.yaml`。
- 在 Step8 执行前暂停，让用户确认实验配置。
- 前端加"继续"/"跳过"/"修改"按钮。

### 46. 模型 A/B 对比
- 同一篇论文、同一个 step，用两个不同模型跑，对比输出质量。
- 帮助用户选择最优模型。

### 47. 历史版本管理
- 每次 pipeline 运行保留完整快照，支持回滚到任意版本。
- 当前 `temp_file_manager` 的 keep/discard 是二进制的，没有版本概念。

### 48. 论文模板系统
- 常见论文类型（图像分类、目标检测、NLP 分类、生成模型）预置模板。
- Step5 可直接套用模板，减少 LLM 调用，提高一致性。

---

## P8 — 配置与运维

### 49. 多环境配置
- 当前所有配置硬编码在 `config.py`。支持 dev/staging/prod 环境变量覆盖。
- `FLASK_ENV`、`LOG_LEVEL`、`LLM_TIMEOUT` 等通过环境变量可配。

### 50. 健康检查增强
- 当前 `/api/health` 只返回论文数量。应检查：
  - LLM API 连通性（ping 一次）
  - 磁盘空间
  - settings.json 可写性
  - 各 step 模块可 import

### 51. 优雅关闭
- `start.bat` 启动的两个进程无协调关闭机制。
- Flask 收到 SIGTERM 时应等待当前 step 完成再退出。
- Express 同理。

### 52. 日志轮转
- `logs/app.log` 无大小限制，长时间运行会占满磁盘。
- 用 `RotatingFileHandler`（maxBytes=10MB, backupCount=5）。

### 53. 输出目录清理策略
- `data/output/` 随论文增加无限增长。
- 可加自动清理：超过 N 天未访问的论文自动标记为"待清理"，用户确认后删除。

---

## 问题分布统计

| 类别 | Critical | High | Medium | Low | 已修复 |
|------|----------|------|--------|-----|--------|
| 安全 | 2 | 0 | 2 | 0 | #1 #2 |
| 后端核心 | 0 | 0 | 5 | 6 | #3 #4 #6 |
| Step 实现 | 5 | 6 | 9 | 8 | #5 #13 #28 |
| 前端 | 2 | 2 | 4 | 3 | |
| 架构 | 0 | 1 | 3 | 2 | |
| **合计** | **9** | **9** | **23** | **19** | **8 项** |

---

## 建议执行顺序

```
✅ 已完成
  → #1  XSS 修复
  → #2  路径穿越修复
  → #3  LLM 调用加异常处理（5 处）
  → #4  贪婪正则修复
  → #5  Step5.8 防幻觉补全
  → #6  JSON 解析统一（llm_manager.parse_llm_json）
  → #13 LLM 管理统一（llm_manager.py 一站式接口）
  → #28 重复代码消除（_extract_code / _parse_json 统一）

下一批（架构优化，~1.5 天）
  → #7  去掉 Express 中间层
  → #8  fetch 错误处理
  → #9  settings.json 加锁
  → #11 绑定 127.0.0.1 + token 认证
  → #12 消除 from config import X 值拷贝陷阱

第三批（体验提升，~1 天）
  → #10 子进程管理
  → #14 按钮 loading 状态
  → #15 多论文并发跟踪
  → #17 上传大小限制

第四批（质量 + 运维，持续）
  → #35 零测试覆盖 → 先写 llm_manager / config / api 单测
  → #34 指数退避加上限
  → #33 缓存 key 加 system prompt
  → #52 日志轮转
```
