"""
Step8: 实验执行管理

对 Step7 生成的 generated_project 进行实验执行管理：
    - 判断每个实验能否在当前环境运行（RUN / SKIP）及原因
    - 对 decision=RUN 的实验实际执行，保存 stdout/stderr/result
    - 汇总产出 execution_plan.yaml / execution_report.yaml

Step8 只回答"实验跑了吗 / 为什么没跑 / 跑成功了吗 / 得到了什么原始结果"。
Step8 是管线终点：不做结果分析与论文对比（结果准确性、指标对比、论文 vs 复现对比均不在本管线内）。

模块划分：
    experiment_analyzer   解析 generated_project 的实验列表
    resource_checker      用程序检测真实环境（CPU/RAM/GPU/显存/磁盘/数据集/Python/依赖，不做 LLM 猜测）
    experiment_scheduler  结合需求与资源做 RUN/SKIP 决策
    experiment_runner     对 RUN 实验实际执行并保存输出
    result_collector      聚合为 execution_report
    step8_main            总入口
"""
