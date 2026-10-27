"""Step5.3: 数据集分析 → dataset.yaml"""

from step5.base_analyzer import BaseAnalyzer
from llm_manager import llm_call, parse_llm_json

# 数据来源类型（data_source）合法值。判定"这篇论文是否需要外部数据集"，
# 供 Step8 决定是否做数据集可用性检查（非 external_dataset 时数据集检查不适用）。
DATA_SOURCE_VALUES = ("external_dataset", "synthetic", "environment", "none")

# 数据获取方式（acquisition）合法值，与 data_source 正交：
#   local_path  外部数据集需用户预置本地路径（Step8 检查目录存在性）
#   auto        外部数据集可自动下载/随库内置（MNIST/sklearn digits/HF，Step8 不查目录）
#   n/a         无外部数据（synthetic/environment/none），不适用
ACQUISITION_VALUES = ("local_path", "auto", "n/a")


class DatasetAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        ds = self.key_info.get("dataset", {})

        # 数据集版本（color/grayscale/segmented）
        versions = self._extract_versions(ds)

        # 预处理细节
        pp = str(ds.get("preprocessing", ""))

        # 规则层预判 data_source：论文未命名任何外部数据集时，不可能是 external_dataset，
        # 直接预判 synthetic/none，避免 LLM「没把握就填 external_dataset」的保守回退
        # 把程序化生成的图/序列误判成"需要用户提供本地路径的外部数据集"。
        rule_source = self._rule_data_source(ds)

        return {
            "paper_name": self.paper_name,
            # 规则层先给一个起点（未命名数据集 → synthetic/none），
            # 再由 analyze() 里 LLM 层对有命名数据集的情况细判并可能改写。
            "data_source": rule_source,
            "acquisition": "local_path" if rule_source == "external_dataset" else "n/a",
            "data_source_note": "规则预判，待LLM复核",
            "datasets": [{
                "name": self._pick(ds.get("dataset_name")),
                "source": self._pick(ds.get("dataset_source")),
                "size": self._pick(ds.get("dataset_size")),
                "num_classes": self._pick(ds.get("number_of_classes")),
                "data_type": self._pick(ds.get("data_type")),
            }],
            "dataset_versions": versions if versions else ["NOT_FOUND"],
            "preprocessing": self._build_per_version_preprocessing(ds, versions),
            "split_strategy": {
                "method": self._determine_split_method(ds),
                "ratios": self._pick(ds.get("train_test_split")),
                "leaf_mapping_constraint": (
                    "同一片叶子的多张图片必须保持在同一个集合中"
                    if "same leaf" in str(ds.get("train_test_split", "")).lower()
                    else "NOT_FOUND"
                ),
            },
        }

    def analyze(self, skip_llm: bool = False) -> dict:
        """在三层防护之后，额外用 LLM 判定 data_source / acquisition（规则无法可靠判断）。"""
        result = super().analyze(skip_llm=skip_llm)
        if not skip_llm:
            result = self._determine_data_source(result)
        # 无论是否 LLM，最终都做枚举校验，非法值回退默认。
        result["data_source"] = self._coerce_data_source(result.get("data_source"))
        result["acquisition"] = self._coerce_acquisition(
            result.get("acquisition"), result["data_source"]
        )
        return result

    def _determine_data_source(self, result: dict) -> dict:
        """LLM 从论文信息判断数据来源类型与获取方式；失败/无把握回退默认。"""
        prompt = f"""你是一个论文数据依赖分析器。判断这篇论文的实验是否需要外部数据集，并归类为下面四类之一：

- external_dataset：依赖外部真实数据集（需下载或提供本地路径，例如 ImageNet/COCO/PlantVillage 等公开数据集）
- synthetic：数据在代码内生成或合成（随机张量、程序化生成、GAN/扩散生成、从分布采样、基准函数采样等）
- environment：数据来自交互式环境或仿真器（强化学习 gym/Atari/MuJoCo、物理仿真等）
- none：无任何数据依赖（纯算法/理论推导，不训练、不评估数据）

判定要点：
- 论文未明确命名外部数据集（数据集名/来源为 NOT_FOUND），实验用到的图/序列/张量等是
  程序化生成的（星图、环图、随机图、基准函数采样等）→ 属于 synthetic，不要因为
  "运行时从磁盘读取输入"就判 external_dataset。
- 只有论文明确依赖某个真实公开数据集（ImageNet/COCO/PlantVillage 等）才是 external_dataset。

若 data_source == external_dataset，再给出 acquisition：
- local_path：需要用户把数据放到本地路径（论文没有说可自动下载）
- auto：数据可由库自动下载或随库内置（torchvision MNIST/CIFAR、sklearn 内置数据、HuggingFace datasets 等）
若 data_source 不是 external_dataset，acquisition 固定为 n/a。

严格规则：
1. 只能依据下方论文信息判断，没把握就填 external_dataset + local_path（保守默认，避免漏检）
2. 只输出 JSON，格式：{{"data_source": "...", "acquisition": "...", "data_source_note": "一句话判断依据"}}
3. data_source 必须是上面四个值之一；acquisition 必须是 local_path/auto/n/a 之一

=== 论文关键信息 ===
{self.get_ki_summary()[:6000]}

=== 论文原文片段 ===
{self.get_paper_text()[:4000]}
"""
        try:
            resp = llm_call(prompt, task_type="light").strip()
            out = parse_llm_json(resp)
            if "error" in out:
                raise ValueError(out["error"])
            result["data_source"] = self._coerce_data_source(out.get("data_source"))
            result["acquisition"] = self._coerce_acquisition(
                out.get("acquisition"), result["data_source"]
            )
            note = out.get("data_source_note")
            if note:
                result["data_source_note"] = note
        except Exception:
            result["data_source"] = "external_dataset"
            result["acquisition"] = "local_path"
            result["data_source_note"] = "LLM 判定失败，回退默认 external_dataset/local_path"

        # 硬约束：论文未命名任何外部数据集（dataset_name/source 均 NOT_FOUND）时，
        # 不可能是 external_dataset。LLM 的保守回退若仍判 external_dataset，强制修正为
        # 规则层预判的 synthetic/none，杜绝"无数据集却生成数据集路径"。
        ds = self.key_info.get("dataset", {})
        name = str(ds.get("dataset_name", "NOT_FOUND"))
        source = str(ds.get("dataset_source", "NOT_FOUND"))
        if (name == "NOT_FOUND" and source == "NOT_FOUND"
                and result.get("data_source") == "external_dataset"):
            result["data_source"] = self._rule_data_source(ds)
            result["acquisition"] = "n/a"
            result["data_source_note"] = (
                f"论文未命名外部数据集，规则修正为 {result['data_source']}"
                "（数据为程序化生成/无外部依赖）"
            )
        return result

    def _rule_data_source(self, ds: dict) -> str:
        """规则层预判 data_source，不依赖 LLM。

        论文未命名任何外部数据集（dataset_name/source 均 NOT_FOUND）时，
        不可能是 external_dataset。此时按 data_type/preprocessing 是否描述
        「生成/合成/图类/随机」等输入数据，预判 synthetic 或 none：
          - synthetic：代码内程序化生成输入（随机图/序列/基准函数采样等）
          - none：无任何数据依赖（纯理论推导，无实验输入）
        """
        name = str(ds.get("dataset_name", "NOT_FOUND"))
        source = str(ds.get("dataset_source", "NOT_FOUND"))
        if name != "NOT_FOUND" or source != "NOT_FOUND":
            # 有命名数据集 → 交由 LLM 细判（可能是 external_dataset/auto 等）
            return "external_dataset"

        text = (str(ds.get("data_type", "")) + " " + str(ds.get("preprocessing", ""))).lower()
        synthetic_signals = (
            "graph", "generated", "synthetic", "sample", "随机", "random",
            "star", "cycle", "benchmark", "生成", "合成", "图", "采样", "程序化",
        )
        if any(sig in text for sig in synthetic_signals):
            return "synthetic"
        return "none"

    @staticmethod
    def _coerce_data_source(value) -> str:
        """枚举校验：非法/缺失值回退 external_dataset。"""
        v = str(value or "").strip().lower()
        return v if v in DATA_SOURCE_VALUES else "external_dataset"

    @staticmethod
    def _coerce_acquisition(value, data_source: str) -> str:
        """枚举校验：非 external_dataset 强制 n/a；否则非法/缺失值回退 local_path。"""
        if data_source != "external_dataset":
            return "n/a"
        v = str(value or "").strip().lower()
        return v if v in ACQUISITION_VALUES and v != "n/a" else "local_path"

    @staticmethod
    def _determine_split_method(ds: dict) -> str:
        """根据数据集约束推断正确的划分方法"""
        split_text = str(ds.get("train_test_split", "")).lower()

        # 有叶片级约束 → 分组划分
        if "same leaf" in split_text:
            return "leaf_based_split"

        # 分层抽样
        if "stratified" in split_text:
            return "stratified_split"

        return "random_split"

    @staticmethod
    def _extract_segmentation_algorithm(ds: dict) -> dict:
        """从论文预处理原文中恢复分割算法的方法描述（参数缺失时只把参数标 NOT_FOUND）。

        论文通常会描述分割思路（颜色空间、分析分量、color cast 校正），只是不给
        脚本阈值/参数。此前整块硬编码 NOT_FOUND，把「方法已知、参数未知」误报成
        「方法未知」，导致下游把整条分割管线当未公开处理。
        """
        import re
        pp = str(ds.get("preprocessing", ""))
        if not pp or pp == "NOT_FOUND":
            pp = str(ds.get("data_type", ""))  # 退化信号：data_type 可能复述分割思路
        pp_lower = pp.lower()

        # 颜色空间：Lab/HSB/HSV/RGB/XYZ 等命名（含 "L*a*b*" 变体）
        color_spaces = []
        for cs in ("Lab", "HSB", "HSV", "RGB", "XYZ", "L*a*b*", "YCbCr"):
            if re.search(rf"(?<![a-z]){re.escape(cs.lower())}(?![a-z])", pp_lower):
                color_spaces.append(cs)

        # 分析分量：color/lightness/saturation/hue/luminance/intensity
        components = []
        for comp in ("color", "lightness", "saturation", "hue", "luminance", "intensity"):
            if re.search(rf"\b{comp}\b", pp_lower):
                components.append(comp)

        # color cast 校正
        fixed_color_casts = bool(
            re.search(r"color\s+casts?[^.]{0,40}\b(fix|fixed|correct|adjust|remov)",
                      pp_lower)
            or re.search(r"\b(fix|fixed|correct|adjust|remov)[^.]{0,40}color\s+casts?",
                         pp_lower)
        )

        # 掩膜/分割关键词：有才认为论文描述了分割方法
        has_mask = bool(re.search(r"mask|segment", pp_lower))
        described = has_mask and (color_spaces or components or fixed_color_casts)

        if not described:
            return {
                "method": "NOT_FOUND — 论文未公开具体分割算法",
                "color_spaces": [],
                "components": [],
                "fixed_color_casts": False,
                "parameters": "NOT_FOUND — 论文未公开分割参数",
                "implementation_status": (
                    "NOT_FOUND: 论文提及分割思路但未提供完整算法/参数"
                ),
            }

        method_bits = ["按颜色/光照分量分析生成目标掩膜，去除背景、保留目标区域"]
        if color_spaces:
            method_bits.append("分析颜色空间: " + "/".join(color_spaces))
        if components:
            method_bits.append("分析分量: " + "/".join(components))
        if fixed_color_casts:
            method_bits.append("并做 color cast 校正")
        return {
            "method": "；".join(method_bits) + "（论文描述，脚本阈值/参数未公开）",
            "color_spaces": color_spaces,
            "components": components,
            "fixed_color_casts": fixed_color_casts,
            "parameters": "NOT_FOUND — 论文未公开脚本阈值/参数",
            "implementation_status": (
                "PARTIAL: 方法描述已知（见 method/color_spaces/components），"
                "脚本参数 NOT_FOUND（阈值/形态学参数等未公开）"
            ),
        }

    def _build_per_version_preprocessing(self, ds: dict,
                                          versions: list) -> dict:
        """
        按数据集版本 (color/grayscale/segmented) 分别定义预处理配置。

        不同版本需要不同的预处理管道，不能共用一个全局配置。
        """
        # 非图像数据（合成向量 / RL 状态 / 无数据）没有颜色/灰度/分割版本概念，
        # 不能套用 PlantVillage 的图像预处理模板，改为单一 default 版本。
        if not self._is_image_data(ds):
            return self._build_generic_preprocessing(ds)

        resize = self._extract_resize(ds)
        has_pretrained = (
            self.key_info.get("methodology", {})
            .get("model_architecture", {})
            .get("pretrained_model", "NOT_FOUND") != "NOT_FOUND"
        )

        # ImageNet 标准归一化（使用预训练权重时必须）
        imagenet_norm = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "note": "ImageNet 标准归一化；不使用预训练权重时可替换为 [0.5,0.5,0.5]",
        }

        # 共享步骤
        shared = {
            "resize": resize,
            "to_tensor": "transforms.ToTensor()",
            "dtype": "torch.float32",
        }

        # ---- Color ----
        color_config = dict(shared)
        color_config.update({
            "description": "彩色版本：保持原始RGB三通道",
            "channels": 3,
            "normalization": imagenet_norm if has_pretrained else {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "note": "无预训练时的简单归一化",
            },
            "augmentation": [
                {"name": "RandomHorizontalFlip", "p": 0.5},
            ],
        })

        # ---- Grayscale ----
        gray_config = dict(shared)
        gray_config.update({
            "description": "灰度版本：转换为单通道后重复为3通道",
            "channels": 3,
            "source_channels": 1,
            "channel_handling": {
                "paper_stated": "NOT_FOUND — 论文未明确说明灰度图是保持1通道还是复制为3通道",
                "inferred_approach": "repeat to 3 channels",
                "inference_basis": (
                    "预训练权重基于 ImageNet 3 通道输入；"
                    "常见框架的图像加载层默认期望 3 通道；"
                    "论文未提及修改第一层 conv1 的 in_channels。"
                    "因此推断灰度图被复制为 3 通道以适配预训练模型。"
                ),
                "alternatives": [
                    "方案A (推荐): repeat 3通道，适配预训练权重，与论文结果可比",
                    "方案B: 保持1通道，修改 conv1 的 in_channels=1，但无法使用预训练权重",
                ],
                "choice": "方案A — repeat to 3 channels",
            },
            "pipeline": [
                "transforms.Grayscale(num_output_channels=1)",
                "transforms.ToTensor()",
                "lambda x: x.repeat(3, 1, 1)",  # 1→3 通道以适配预训练模型
            ],
            "normalization": imagenet_norm if has_pretrained else {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "augmentation": [
                {"name": "RandomHorizontalFlip", "p": 0.5},
            ],
            "note": "灰度版本为常见消融变体，用于验证颜色信息对任务的贡献",
        })

        # ---- Segmented ----
        seg_config = dict(shared)
        seg_algo = self._extract_segmentation_algorithm(ds)
        seg_config.update({
            "description": (
                "分割版本：去除背景、保留目标区域；"
                "方法描述见 segmentation_algorithm.method，"
                "脚本参数未公开时需自行实现或替代。"
            ),
            "channels": 3,
            "segmentation_algorithm": {
                **seg_algo,
                "fallback": (
                    "如无法复现分割脚本，可使用 GrabCut 或 U-Net 替代，"
                    "但结果可能与论文不完全一致。"
                ),
            },
            "normalization": imagenet_norm if has_pretrained else {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "augmentation": [
                {"name": "RandomHorizontalFlip", "p": 0.5},
            ],
            "note": "分割预处理最复杂，复现难度最高（具体性能数字见 experiment.comparison_results）",
        })

        return {
            "versions": {
                "color": color_config,
                "grayscale": gray_config,
                "segmented": seg_config,
            },
            "version_selector": {
                "config_key": "dataset_version",
                "values": ["color", "grayscale", "segmented"],
                "default": "color",
                "note": "通过 config.yaml 的 dataset_version 字段切换预处理版本",
            },
        }

    def _build_generic_preprocessing(self, ds: dict) -> dict:
        """非图像数据（合成/RL/无数据）的通用预处理配置：单一 default 版本。

        不生成 color/grayscale/segmented 图像版本，避免 PlantVillage 模板泄漏到
        非图像论文。data_source 字段本身仍由 analyze() 的 LLM 层单独判定。
        """
        pp_text = self._pick(ds.get("preprocessing"))
        return {
            "versions": {
                "default": {
                    "description": "非图像数据：通用预处理（按数据语义做标准化/归一化）",
                    "preprocessing": pp_text,
                    "normalization": {
                        "note": "非图像数据无 RGB 通道概念，按数据语义做标准化（如 z-score/归一化）",
                    },
                    "augmentation": [],
                    "note": "数据集无颜色/灰度/分割版本；仅单一 default 版本",
                }
            },
            "version_selector": {
                "config_key": "dataset_version",
                "values": ["default"],
                "default": "default",
                "note": "非图像数据仅一个 default 版本",
            },
        }

    def _extract_versions(self, ds: dict) -> list:
        """提取数据集版本 (color/grayscale/segmented)；非图像数据返回 ["default"]。

        与 _build_per_version_preprocessing 保持一致：非图像数据只有一个 default 版本，
        不能把 PlantVillage 的颜色/灰度/分割版本概念强加到合成/RL/无数据论文上。
        """
        if not self._is_image_data(ds):
            return ["default"]

        versions = []
        pp = str(ds.get("preprocessing", "")).lower()
        dtype = str(ds.get("data_type", "")).lower()

        for v in ["color", "grayscale", "segmented"]:
            if v in pp or v in dtype:
                versions.append(v)

        return versions

    def _extract_resize(self, ds: dict) -> str:
        t = str(ds.get("preprocessing", ""))
        for s in ["256", "224", "512", "299", "300"]:
            if s in t:
                return f"{s}x{s}"
        return "NOT_FOUND"
