"""Step5.8: Agent代码规范化 → optimized_project_spec.yaml

汇总5.1~5.7，用LLM细化到每个.py文件的class/function/依赖。

改进:
  - LLM prompt 中传递完整的 model_architecture 细节
  - 传递 possible_difficulties 给 Step6
  - 对 LLM 输出做基本结构校验
"""

import json
from step5.base_analyzer import BaseAnalyzer
from llm_manager import llm_call, parse_llm_json
from logger import logger


class SpecAnalyzer(BaseAnalyzer):

    def __init__(self, paper_name: str):
        super().__init__(paper_name)
        self.yamls = {}
        for name in ["reproduction_goal", "technology", "dataset",
                      "model_design", "experiment_design",
                      "project_structure", "training_config"]:
            self.yamls[name] = self.load_yaml(f"{name}.yaml")

    def extract_by_rules(self) -> dict:
        goal = self.yamls.get("reproduction_goal", {})
        struct = self.yamls.get("project_structure", {})
        models = self.yamls.get("model_design", {})
        training = self.yamls.get("training_config", {})
        tech = self.yamls.get("technology", {})

        # 项目信息
        proj = {
            "name": struct.get("project_name", self.paper_name),
            "description": goal.get("task", ""),
            "paper_type": goal.get("paper_type", "unknown"),
        }

        # 模块架构
        root = struct.get("root_structure", {})
        dir_files = root.get("files", {})
        mapping = struct.get("module_mapping", {})
        model_list = models.get("models", [])

        # 按实际生成的目录构建模块（而非固定 ML 目录清单），
        # 纯算法论文只会有 src/config 等，不会有 data/models/training。
        _DEFAULT_DESC = {
            "data": "数据模块", "models": "模型模块", "training": "训练模块",
            "evaluation": "评估模块", "utils": "工具模块", "config": "配置模块",
            "src": "算法/方法实现",
        }
        modules = {}
        for dir_name, fs in dir_files.items():
            if dir_name == "." or not fs:
                continue
            modules[dir_name] = {
                "directory": f"{dir_name}/",
                "files": [f for f in fs if f != "__init__.py"],
                "description": mapping.get(dir_name, _DEFAULT_DESC.get(dir_name, "模块")),
            }

        # 模型类（使用 sanitized code_class_name）
        if model_list and "models" in modules:
            modules["models"]["classes"] = [
                {
                    "name": m.get("code_class_name", m.get("name", "")),
                    "type": m.get("type", ""),
                    "input": m.get("input", ""),
                    "output": m.get("output", ""),
                    "pretrained": m.get("pretrained", ""),
                    "file": f"models/{m.get('code_class_name', 'model').lower()}.py",
                }
                for m in model_list
            ]

        # 文件依赖
        all_files = []
        for d, fs in dir_files.items():
            prefix = "" if d == "." else f"{d}/"
            for f in fs:
                if f != "__init__.py":
                    all_files.append(f"{prefix}{f}")
        for m in model_list:
            fn = f"models/{m.get('code_class_name', 'model').lower()}.py"
            if fn not in all_files:
                all_files.append(fn)

        # 执行流程：按实际模块拼装，纯算法论文不再假定 data/models/train
        is_trainable = goal.get("is_trainable", False)
        data_source = self.yamls.get("dataset", {}).get("data_source", "external_dataset")
        flow_steps = []
        if "config" in modules:
            flow_steps.append("加载配置 (config/config.yaml)")
        if "data" in modules:
            flow_steps.append("data/: 准备数据集和数据加载器")
        if "src" in modules:
            flow_steps.append("src/: 执行论文的算法/方法")
        if "models" in modules:
            flow_steps.append("models/: 初始化模型")
        if is_trainable and "training" in modules:
            flow_steps.append("training/train.py: 训练循环与验证评估")
        if "evaluation" in modules:
            flow_steps.append(
                "evaluation/: 测试集评估"
                + ("与对比" if goal.get("paper_type") == "empirical_comparison" else "")
            )
        flow = [f"{i}. {s}" for i, s in enumerate(flow_steps, 1)]

        # 依赖：仅 ML/训练原型才需要 torch 系
        if is_trainable or model_list:
            external = [
                "torch", "torchvision", "numpy", "pyyaml",
                "scikit-learn", "pillow", "opencv-python",
            ]
        else:
            external = ["numpy", "pyyaml"]

        # 复现注意事项
        repro_notes = training.get("reproducibility_notes", [])

        return {
            "project": proj,
            "architecture": {"modules": modules},
            "dependencies": {
                "files": sorted(set(all_files)),
                "external": external,
            },
            "execution_flow": flow,
            "reproducibility_notes": repro_notes if repro_notes else [],
            "code_requirements": {},
        }

    def analyze(self, skip_llm: bool = False) -> dict:
        """覆盖: 规则提取后再调LLM细化code_requirements"""
        result = self.extract_by_rules()

        # LLM 细化
        logger.info(f"  [5.8] LLM细化代码规格...")

        # 构建丰富的 prompt — 包含所有之前丢失的信息
        model_arch = (
            self.key_info.get("methodology", {})
            .get("model_architecture", {})
        )
        arch_details = {}
        if isinstance(model_arch, dict):
            arch_details = {
                k: v for k, v in model_arch.items()
                if isinstance(v, str) and v not in ("NOT_FOUND", "")
            }

        important_results = (
            self.key_info.get("experiment", {}).get("important_results", "")
        )

        prompt = f"""你是资深软件架构师。基于论文信息为每个.py文件设计详细代码规格。

=== 论文关键信息 ===
{self.get_ki_summary()[:6000]}

=== 模型架构详情 ===
{json.dumps(arch_details, ensure_ascii=False, indent=2)[:3000]}

=== 实验重要结果 ===
{str(important_results)[:2000]}

=== 工程分析结果 ===
{json.dumps(self.yamls, ensure_ascii=False, indent=2)[:5000]}

=== 当前规格框架 ===
{json.dumps(result, ensure_ascii=False, indent=2)[:3000]}

=== 重要：config/config.yaml 训练参数（必须严格照抄，不得修改） ===
{self._format_training_config_for_prompt()}

请为 code_requirements 字段提供每个文件的详细规格。格式:
{{
  "code_requirements": {{
    "data/dataset.py": {{
      "description": "...",
      "class": "ClassName",
      "parent_class": "torch.utils.data.Dataset",
      "methods": ["__init__", "__len__", "__getitem__"],
      "functions": ["load_data", "preprocess"],
      "pseudocode": "简要伪代码"
    }},
    "models/xxx.py": {{ ... }},
    "training/train.py": {{
      "functions": ["train_epoch", "validate", "main"],
      "uses": ["models.*", "data.dataset", "utils.metrics"],
      "pseudocode": "..."
    }}
  }}
}}

只输出JSON，不要解释。如果论文信息不足，标注 "TO_BE_DETERMINED"。

⚠️ 特别注意：config/config.yaml 中的 scheduler.step_size 必须是训练参数表中指定的确切值，不要自己估算。"""

        try:
            resp = llm_call(prompt, task_type="light").strip()
            llm = parse_llm_json(resp)
            if "error" in llm:
                raise ValueError(llm["error"])

            if "code_requirements" in llm:
                # 基本校验：确保是 dict
                cr = llm["code_requirements"]
                if isinstance(cr, dict) and len(cr) > 0:
                    result["code_requirements"] = cr
                else:
                    logger.warning("  LLM返回的 code_requirements 为空或格式错误")
        except Exception as e:
            logger.warning(f"  LLM细化失败: {e}")

        # Layer 3: 后验证（基类的三层防幻觉机制）
        self._validate(result)

        from datetime import datetime
        result["_meta"] = {
            "paper_name": self.paper_name,
            "generated_at": datetime.now().isoformat(),
        }
        return result

    def _format_training_config_for_prompt(self) -> str:
        """从 training_config.yaml 提取关键训练参数，供 LLM 照抄"""
        tc = self.yamls.get("training_config", {})
        training = tc.get("training", {})
        if not training or training.get("applies") is False:
            return "(无训练配置)"

        lines = []
        lines.append(f"  batch_size: {training.get('batch_size', '?')}")
        lines.append(f"  epochs: {training.get('epochs', '?')}")

        opt = training.get("optimizer", {})
        lines.append(f"  optimizer: {opt.get('name', 'sgd')}")
        lines.append(f"  learning_rate: {opt.get('lr', '?')}")
        lines.append(f"  momentum: {opt.get('momentum', '?')}")
        lines.append(f"  weight_decay: {opt.get('weight_decay', '?')}")

        sched = training.get("scheduler", {})
        lines.append(f"  scheduler.name: {sched.get('name', 'StepLR')}")
        lines.append(f"  scheduler.step_size: {sched.get('step_size', '?')}")
        lines.append(f"  scheduler.gamma: {sched.get('gamma', '?')}")

        return "训练参数（config/config.yaml 必须使用以下值）：\n" + "\n".join(lines)
