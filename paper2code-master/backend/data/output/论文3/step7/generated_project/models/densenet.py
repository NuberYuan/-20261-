# -*- coding: utf-8 -*-
"""Densenet 模型定义（FR-03）。

本文件定义 ``Densenet`` 类：直接继承 ``torchvision.models.densenet.DenseNet``
（采用 DenseNet-121 的结构参数：growth_rate=32, block_config=(6,12,24,16),
num_init_features=64, bn_size=4）。

设计要点：
- 直接继承 torchvision 的模型类，不手写自定义卷积块，保证参数名 / 形状与
  torchvision 完全一致，transfer 模式下预训练权重可 1:1 ``load_state_dict``；
- ``num_classes`` / ``in_channels`` / ``drop_rate`` 等形状参数一律从
  config/config.yaml 读取，config 中为 null（NOT_FOUND）时用论文描述推导的
  默认值兜底，并在注释中标明来源；
- 模型类只定义结构（torchvision 默认随机初始化），不写 training_mode/transfer
  分支、不在 ``__init__`` 中加载权重；权重初始化与冻结策略统一在
  training/train.py 的 ``initialize_model`` 中处理；
- 归一化、padding 等策略与 torchvision 保持一致，参数量与参考实现同量级
  （DenseNet-121 约 7M 量级，随 num_classes 变化）。

对应 Step5 设计：architecture_summary = Inspired by DenseNet architecture,
Li et al. proposed Hdenseunet for liver and liver tumor segmentation.
"""

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml

try:  # torchvision >= 0.4 的标准导入路径
    from torchvision.models.densenet import DenseNet
except ImportError:  # pragma: no cover - 兼容旧版本导入路径
    from torchvision.models import DenseNet  # type: ignore

__all__ = ["Densenet", "load_config"]


# ---------------------------------------------------------------------------
# 默认值（仅在 config.yaml 中对应字段为 null / 缺失时使用）
# ---------------------------------------------------------------------------
_DEFAULT_NUM_CLASSES = 2        # 论文设计：binary segmentation（num_classes=2）
_DEFAULT_OUT_CHANNELS = 1       # config.model.out_channels（单通道 sigmoid 输出）
_DEFAULT_IN_CHANNELS = 3        # config.data.in_channels / config.model.in_channels
_DEFAULT_DROP_RATE = 0.0        # torchvision DenseNet 默认 drop_rate=0.0
                                #（config 的 dropout_ratio 为 null，故取该默认）
_DEFAULT_GROWTH_RATE = 32       # DenseNet-121 结构参数
_DEFAULT_BLOCK_CONFIG = (6, 12, 24, 16)   # DenseNet-121 结构参数
_DEFAULT_NUM_INIT_FEATURES = 64          # DenseNet-121 结构参数
_DEFAULT_BN_SIZE = 4                     # DenseNet-121 结构参数


# ---------------------------------------------------------------------------
# config/config.yaml 读取工具
# ---------------------------------------------------------------------------
def _resolve_config_path(config_path: Optional[str] = None) -> Optional[str]:
    """定位 config/config.yaml，返回绝对路径；找不到时返回 None。"""
    candidates = []
    if config_path:
        candidates.append(config_path)
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    candidates.append(os.path.join(project_root, "config", "config.yaml"))
    candidates.append(os.path.join(os.getcwd(), "config", "config.yaml"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """读取 config/config.yaml 并返回 dict（读取失败时返回空 dict）。"""
    path = _resolve_config_path(config_path)
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _cfg_get(config: Optional[Dict[str, Any]],
             key: Union[str, Sequence[str]],
             default: Any = None) -> Any:
    """按逐级路径安全读取 config，任何一级缺失或值为 None 时返回 default。

    只允许访问 config.yaml 中真实存在的键，例如 'model.num_classes'、
    'data.in_channels'、'training.dropout_ratio'。
    """
    if not isinstance(config, dict):
        return default
    keys = key.split(".") if isinstance(key, str) else list(key)
    node: Any = config
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return default if node is None else node


def _to_int(value: Any, default: int) -> int:
    """把 config 中的数值安全转换为 int；None / 非法值回退到 default。"""
    if value is None:
        return int(default)
    try:
        result = int(value)
    except (TypeError, ValueError):
        return int(default)
    return result


def _to_float(value: Any, default: float) -> float:
    """把 config 中的数值安全转换为 float；None / 非法值回退到 default。"""
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _to_block_config(value: Any) -> Tuple[int, ...]:
    """把 block_config 归一化为 int 元组；非法值回退到 DenseNet-121 默认。"""
    if value is None:
        return tuple(_DEFAULT_BLOCK_CONFIG)
    if isinstance(value, (list, tuple)):
        try:
            converted = tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return tuple(_DEFAULT_BLOCK_CONFIG)
        if len(converted) == 0:
            return tuple(_DEFAULT_BLOCK_CONFIG)
        return converted
    return tuple(_DEFAULT_BLOCK_CONFIG)


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
class Densenet(DenseNet):
    """Densenet（DenseNet-121 结构）：接收预处理后的图像，输出预测结果。

    FR-03: 构建 Densenet 模型，能够接收预处理后的输入并输出预测结果。

    参数
    ----
    config : None | str | dict, default None
        - None   : 从 config/config.yaml 读取；
        - str    : 指定 yaml 路径读取；
        - dict   : 直接使用外部传入的配置字典。
    num_classes : int, optional
        输出类别数；不传则依次取 config['model']['num_classes']、
        config['model']['out_channels']，均为 null 时回退到 2（论文设计：
        binary segmentation, num_classes=2）。
    in_channels : int, optional
        输入通道数；不传则取 config['model']['in_channels'] / config['data']
        ['in_channels']，缺失时回退到 3（config.data.in_channels）。
    drop_rate : float, optional
        DenseNet 分类头前的 dropout 比例；不传则取 config 的
        model.dropout_ratio / training.dropout_ratio，为 null 时回退到 0.0
        （torchvision DenseNet 默认值）。
    growth_rate, block_config, num_init_features, bn_size, memory_efficient :
        DenseNet-121 结构参数，默认值与 torchvision 一致。

    说明
    ----
    - 结构与 torchvision DenseNet 完全一致（不额外添加层）；仅当 in_channels != 3
      时就地替换 features.conv0 的输入通道数，模块名与参数名保持不变，
      预训练权重仍可按名加载；
    - forward 输出形状为 (batch_size, num_classes) 的 logits（含 batch 维），
      空间尺寸由 adaptive average pooling 归一化，因此 96x96 / 224x224 /
      512x512 等任意输入尺寸均可前向；
    - 本类仅定义结构（随机初始化），不做任何权重加载 / 冻结。
    """

    def __init__(self,
                 config: Optional[Union[str, Dict[str, Any]]] = None,
                 num_classes: Optional[int] = None,
                 in_channels: Optional[int] = None,
                 drop_rate: Optional[float] = None,
                 growth_rate: int = _DEFAULT_GROWTH_RATE,
                 block_config: Sequence[int] = _DEFAULT_BLOCK_CONFIG,
                 num_init_features: int = _DEFAULT_NUM_INIT_FEATURES,
                 bn_size: int = _DEFAULT_BN_SIZE,
                 memory_efficient: bool = False) -> None:
        # ---- 1. 解析配置 -------------------------------------------------
        if config is None:
            cfg: Dict[str, Any] = load_config()
        elif isinstance(config, dict):
            cfg = config
        elif isinstance(config, str):
            cfg = load_config(config)
        else:
            cfg = {}

        # ---- 2. 形状参数（config 优先，null 用论文推导默认值兜底）--------
        # num_classes：config.model.num_classes -> config.model.out_channels -> 2
        if num_classes is not None:
            resolved_num_classes = _to_int(num_classes, _DEFAULT_NUM_CLASSES)
        else:
            cfg_num_classes = _cfg_get(cfg, "model.num_classes", None)
            if cfg_num_classes is None:
                cfg_num_classes = _cfg_get(cfg, "model.out_channels", _DEFAULT_OUT_CHANNELS)
            resolved_num_classes = _to_int(cfg_num_classes, _DEFAULT_NUM_CLASSES)
        if resolved_num_classes <= 0:
            resolved_num_classes = _DEFAULT_NUM_CLASSES

        # in_channels：config.model.in_channels -> config.data.in_channels -> 3
        if in_channels is not None:
            resolved_in_channels = _to_int(in_channels, _DEFAULT_IN_CHANNELS)
        else:
            cfg_in_channels = _cfg_get(cfg, "model.in_channels", None)
            if cfg_in_channels is None:
                cfg_in_channels = _cfg_get(cfg, "data.in_channels", _DEFAULT_IN_CHANNELS)
            resolved_in_channels = _to_int(cfg_in_channels, _DEFAULT_IN_CHANNELS)
        if resolved_in_channels <= 0:
            resolved_in_channels = _DEFAULT_IN_CHANNELS

        # drop_rate：config.model.dropout_ratio -> config.training.dropout_ratio -> 0.0
        if drop_rate is not None:
            resolved_drop_rate = _to_float(drop_rate, _DEFAULT_DROP_RATE)
        else:
            cfg_drop_rate = _cfg_get(cfg, "model.dropout_ratio", None)
            if cfg_drop_rate is None:
                cfg_drop_rate = _cfg_get(cfg, "training.dropout_ratio", _DEFAULT_DROP_RATE)
            resolved_drop_rate = _to_float(cfg_drop_rate, _DEFAULT_DROP_RATE)
        # torch.dropout 要求 0 <= p < 1，防止 config 塞入非法值导致运行期崩溃
        resolved_drop_rate = float(min(max(resolved_drop_rate, 0.0), 1.0 - 1e-6))

        # ---- 3. 结构超参（与 torchvision DenseNet-121 保持一致）---------
        resolved_growth_rate = _to_int(growth_rate, _DEFAULT_GROWTH_RATE)
        resolved_block_config = _to_block_config(block_config)
        resolved_num_init_features = _to_int(num_init_features, _DEFAULT_NUM_INIT_FEATURES)
        resolved_bn_size = _to_int(bn_size, _DEFAULT_BN_SIZE)
        if resolved_growth_rate <= 0:
            resolved_growth_rate = _DEFAULT_GROWTH_RATE
        if resolved_num_init_features <= 0:
            resolved_num_init_features = _DEFAULT_NUM_INIT_FEATURES
        if resolved_bn_size <= 0:
            resolved_bn_size = _DEFAULT_BN_SIZE

        # ---- 4. 构建 torchvision DenseNet 主干（不添加任何额外层）------
        super().__init__(growth_rate=resolved_growth_rate,
                         block_config=resolved_block_config,
                         num_init_features=resolved_num_init_features,
                         bn_size=resolved_bn_size,
                         drop_rate=resolved_drop_rate,
                         num_classes=resolved_num_classes,
                         memory_efficient=bool(memory_efficient))

        # 保存关键配置，便于上层（如 training/train.py）读取
        self.config: Dict[str, Any] = cfg
        self.in_channels: int = resolved_in_channels
        self.out_features: int = resolved_num_classes

        # ---- 5. 输入通道适配（默认 in_channels=3 时不做任何改动）--------
        if resolved_in_channels != 3:
            old_conv = self.features.conv0
            if isinstance(old_conv, nn.Conv2d) and old_conv.in_channels != resolved_in_channels:
                self.features.conv0 = nn.Conv2d(
                    in_channels=resolved_in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=(old_conv.bias is not None),
                )
                # 结构仍与 torchvision 一致（仅 conv0 输入通道数不同），
                # 参数名保持 'features.conv0.weight'，便于按名加载权重。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入预处理后的图像张量，输出预测 logits。

        参数
        ----
        x : torch.Tensor
            形状 (batch_size, in_channels, H, W)。H/W 可为 96、224、512 等
            任意尺寸（内部 adaptive average pooling 会统一到 1x1）。

        返回
        ----
        torch.Tensor
            形状 (batch_size, num_classes) 的未归一化 logits；
            二分类场景可在损失函数侧使用 BCEWithLogitsLoss / dice，
            或在推理侧自行施加 sigmoid/softmax。
        """
        return super().forward(x)


if __name__ == "__main__":  # pragma: no cover - 简易自检（FR-03 验收）
    model = Densenet()
    model.eval()
    dummy = torch.randn(2, model.in_channels, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    total_params = sum(p.numel() for p in model.parameters())
    print("output_shape:", tuple(output.shape))
    print("num_parameters:", total_params)