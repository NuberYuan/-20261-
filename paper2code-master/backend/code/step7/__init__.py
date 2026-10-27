# Step7: 代码生成
#
# 读取 Step6 的 project_brief.yaml，按其中的项目结构和代码生成计划，
# 逐文件生成完整的可复现代码项目，并进行代码层面的检查与修复。
#
# 职责边界:
#   ✅ 读取 Step6 / 创建项目结构 / 分阶段生成代码 / 语法与结构检查 / 需求追踪 / 自动修复
#   ❌ 不运行实验、不训练模型、不分析实验结果（这些属于 Step8）
#
# 模块:
#   step7_main.py           总控制器
#   project_initializer.py  项目骨架初始化（不调 LLM）
#   code_context_builder.py 单文件生成上下文整理
#   code_generator.py       LLM 逐文件生成
#   code_validator.py       规则检查（ast / yaml / 类 / 方法 / import）
#   code_repairer.py        失败后 LLM 修复
