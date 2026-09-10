"""
Step8.1: 实验列表解析器 (Experiment Analyzer)

读取 Step7 生成的 generated_project/config/config.yaml，把 experiment.runs
解析成结构化的实验列表，并提取每个实验的资源需求（模型 / 模式 / 数据版本 /
batch_size / epochs / 是否要求 GPU 等）。

本模块是纯规则驱动的：只解析 YAML，不做 LLM 判断。
"""

from pathlib import Path

import yaml


class ExperimentAnalyzer:
    """从 generated_project 的 config.yaml 解析实验列表。"""

    def __init__(self, generated_project: Path):
        self.project = Path(generated_project)

    def find_config(self) -> Path:
        """定位 generated_project 下的 config.yaml（兼容 .yml）。"""
        for name in ("config/config.yaml", "config/config.yml"):
            p = self.project / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"未找到 generated_project 的 config: {self.project / 'config'}"
        )

    def load_config(self) -> dict:
        cfg = yaml.safe_load(self.find_config().read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg

    def analyze(self, config: dict, data_source: str = "external_dataset",
                acquisition: str = "local_path") -> list:
        """返回实验列表，每个实验附带资源需求元数据。

        data_source：数据来源类型（external_dataset/synthetic/environment/none），
        决定 Step8 是否做数据集可用性检查；非 external_dataset 时数据集检查不适用。
        acquisition：数据获取方式（local_path/auto/n/a），external_dataset 时决定
        是否检查目录存在性（auto 表示可自动下载/随库内置，不查目录）。

        这两个参数的读取优先级：experiment.runs[i] 的 per-run 值 > config 顶层
        data_source/acquisition 键 > 本参数（来自 Step5 dataset.yaml 的 paper 级兜底）。

        返回项关键字段：
            id / run_index / model / mode / dataset_version / train_ratio
            batch_size / epochs / num_classes
            data_source   (数据来源类型，透传给 ResourceChecker)
            acquisition   (数据获取方式，透传给 ResourceChecker)
            gpu_required   (bool，由 system.device / inference.device 决定)
            dataset        {status, path, reason}  数据版本状态
        """
        runs = config.get("experiment", {}).get("runs", []) or []
        data = config.get("data", {}) or {}
        training = config.get("training", {}) or {}
        model_cfg = config.get("model", {}) or {}
        system = config.get("system", {}) or {}
        inference = config.get("inference", {}) or {}

        # 数据依赖字段的读取优先级：per-run > 顶层 config 键 > 调用参数（Step5 dataset.yaml 兜底）
        top_data_source = config.get("data_source") or data_source
        top_acquisition = config.get("acquisition") or acquisition

        batch_size_map = training.get("batch_size", {})
        epochs = training.get("epochs")
        num_classes = model_cfg.get("num_classes")

        # 显式 device 配置决定 gpu_required
        device = str(system.get("device") or inference.get("device") or "").strip().lower()
        # 兼容 "cuda" / "cuda:0" / "gpu" / "gpu:0" 等写法，避免 GPU-only 实验被误判为 CPU
        gpu_required = (
            device in ("cuda", "gpu")
            or device.startswith("cuda:")
            or device.startswith("gpu:")
        )

        # 可选资源需求（来自 config 的 system 段；缺省 None=无要求，Step8 检查时跳过）
        cpu_cores = system.get("cpu_cores") or None
        ram_gb = system.get("ram_gb") or None
        disk_gb = system.get("disk_gb") or None
        python_version = system.get("python_version") or None
        dependencies = system.get("dependencies") or None

        experiments = []
        # 预统计 (model, mode, version) 出现次数，重复项需统一追加后缀去重
        key_counts = {}
        for run in runs:
            if isinstance(run, dict):
                key = (run.get("model"), run.get("training_mode"),
                       run.get("dataset_version"))
                key_counts[key] = key_counts.get(key, 0) + 1

        used_ids = set()
        for i, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            model = run.get("model")
            mode = run.get("training_mode")
            version = run.get("dataset_version")
            train_ratio = run.get("train_ratio", 0.8)

            base = f"{model}_{mode}_{version}"
            if key_counts[(model, mode, version)] > 1:
                # 重复项统一追加 train_ratio 后缀；同一 train_ratio 仍重复时用 run_index 兜底保证唯一
                try:
                    pct = int(float(train_ratio) * 100)
                except (ValueError, TypeError):
                    pct = "x"
                exp_id = f"{base}_{pct}"
                if exp_id in used_ids:
                    exp_id = f"{base}_{pct}_{i}"
            else:
                exp_id = base
            used_ids.add(exp_id)

            if isinstance(batch_size_map, dict):
                batch_size = batch_size_map.get(model)
            else:
                batch_size = batch_size_map

            version_cfg = data.get(version) if isinstance(data, dict) else {}
            if not isinstance(version_cfg, dict):
                version_cfg = {}

            experiments.append({
                "id": exp_id,
                "run_index": i,
                "model": model,
                "mode": mode,
                "dataset_version": version,
                "train_ratio": float(train_ratio),
                "batch_size": batch_size,
                "epochs": epochs,
                "num_classes": num_classes,
                "data_source": run.get("data_source") or top_data_source,
                "acquisition": run.get("acquisition") or top_acquisition,
                "gpu_required": gpu_required,
                "cpu_cores": cpu_cores,
                "ram_gb": ram_gb,
                "disk_gb": disk_gb,
                "python_version": python_version,
                "dependencies": dependencies,
                "dataset": {
                    "status": version_cfg.get("status"),
                    "path": version_cfg.get("path"),
                    "reason": version_cfg.get("reason"),
                },
            })
        return experiments
