# -*- coding: utf-8 -*-
"""training/train.py：模型训练流程实现。

本模块实现训练流程所需的三个核心方法：

* ``initialize_model``：构建 / 加载模型，初始化优化器、调度器与损失函数
* ``train_model``：执行训练与验证循环（含早停与最优模型选择）
* ``save_model``：保存模型权重与训练元信息
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except Exception:
    yaml = None

try:
    from data.dataset import (
        ElectricityUCIElectricityLoadDiagramsTrafficUCIPEMSFTrafficRetailFavoritaGrocerySalesFromKaggleVolatilityOxfordManInstituteRealizedLibraryDataset,
    )
except Exception:
    ElectricityUCIElectricityLoadDiagramsTrafficUCIPEMSFTrafficRetailFavoritaGrocerySalesFromKaggleVolatilityOxfordManInstituteRealizedLibraryDataset = None

try:
    from models.lstm import Lstm
except Exception:
    Lstm = None

try:
    from models.transformer import Transformer
except Exception:
    Transformer = None

logger = logging.getLogger(__name__)

__all__ = [
    "Trainer",
    "initialize_model",
    "train_model",
    "save_model",
]


# ---------------------------------------------------------------------------
# 配置工具
# ---------------------------------------------------------------------------
def load_config(config: Optional[Union[Dict[str, Any], str, Path]] = None) -> Dict[str, Any]:
    """把 config（dict / yaml 路径 / None）转换为嵌套 dict。"""
    if isinstance(config, dict):
        return dict(config)
    if config is None:
        root = Path(__file__).resolve().parents[1]
        for candidate in (
            "config/config.yaml",
            "./config/config.yaml",
            str(root / "config" / "config.yaml"),
        ):
            if os.path.isfile(candidate):
                config = candidate
                break
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.is_file() and yaml is not None:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    return loaded
            except Exception as exc:
                logger.warning("读取配置失败 %s: %s", path, exc)
    return {}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """支持点号嵌套路径与扁平点号键两种写法的配置读取。"""
    if cfg is None or not key:
        return default
    if isinstance(cfg, dict):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        node = cfg
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is not None:
            return node
    return default


def _to_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        result = int(value)
    except Exception:
        result = int(default)
    return max(result, minimum)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 损失函数：分位数损失 (quantile loss)
# ---------------------------------------------------------------------------
def quantile_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Optional[Sequence[float]] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """QL(y, ŷ, q) = q(y - ŷ)+ + (1 - q)(ŷ - y)+，在所有分位数上求和。"""
    if not torch.is_tensor(preds):
        preds = torch.as_tensor(preds)
    preds = preds.float()
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)
    targets = targets.to(preds.device).float()

    qs = [float(q) for q in (quantiles or (0.5,))]
    n_q = len(qs)
    if preds.dim() == 0:
        preds = preds.view(1)
    if preds.size(-1) != n_q:
        preds = preds.unsqueeze(-1)
        qs = [0.5]
        n_q = 1

    while targets.dim() < preds.dim():
        targets = targets.unsqueeze(-1)
    if targets.size(-1) == 1 and n_q > 1:
        targets = targets.expand(*targets.shape[:-1], n_q)

    q = preds.new_tensor(qs).view(*([1] * (preds.dim() - 1)), n_q)
    diff = targets - preds
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


class QuantileLoss(nn.Module):
    """分位数损失封装。"""

    def __init__(self, quantiles: Optional[Sequence[float]] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.quantiles = [float(q) for q in (quantiles or (0.1, 0.5, 0.9))]
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return quantile_loss(preds, targets, self.quantiles, self.reduction)


# ---------------------------------------------------------------------------
# 兜底模型（外部 TFT 实现不可用时保证训练流程可运行）
# ---------------------------------------------------------------------------
class _FallbackQuantileModel(nn.Module):
    """量化回归兜底模型：编码器 + 分位数预测头。"""

    def __init__(
        self,
        config: Optional[Any] = None,
        quantiles: Optional[Sequence[float]] = None,
        input_window: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        feature_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        dropout_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = config if isinstance(config, dict) else {}
        self.quantiles = [float(q) for q in (quantiles or (0.1, 0.5, 0.9))]
        self.input_window = _to_int(
            input_window if input_window is not None else _cfg_get(cfg, "model.input_window", 24), 24
        )
        self.forecast_horizon = _to_int(
            forecast_horizon if forecast_horizon is not None else _cfg_get(cfg, "model.forecast_horizon", 24),
            24,
        )
        self.feature_dim = _to_int(
            feature_dim if feature_dim is not None else _cfg_get(cfg, "data.feature_dim", 1), 1
        )
        hidden = _to_int(
            hidden_size if hidden_size is not None else _cfg_get(cfg, "model.hidden_size", 64), 8
        )
        dropout = _to_float(
            dropout_ratio if dropout_ratio is not None else _cfg_get(cfg, "model.dropout_ratio", 0.1), 0.1
        )
        self.hidden_size = hidden
        self.proj_in = nn.Linear(self.feature_dim, hidden)
        if Transformer is not None:
            self.encoder = Transformer(
                hidden_size=hidden,
                num_layers=1,
                num_attention_heads=2,
                dropout_ratio=dropout,
                max_seq_len=max(self.input_window, 1),
            )
        else:
            self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, self.forecast_horizon * len(self.quantiles))

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        past_inputs: Optional[torch.Tensor] = None,
        future_inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if x is None:
            x = past_inputs
        if x is None:
            x = future_inputs
        if x is None:
            x = targets
        device = next(self.parameters()).device
        if x is None:
            batch_size = int(kwargs.get("batch_size", 1) or 1)
            x = torch.zeros(batch_size, self.input_window, self.feature_dim, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device).float()
        if x.dim() == 1:
            x = x.view(1, -1, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) < self.feature_dim:
            pad = torch.zeros(*x.shape[:-1], self.feature_dim - x.size(-1), device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.size(-1) > self.feature_dim:
            x = x[..., : self.feature_dim]

        h = self.proj_in(x)
        if isinstance(self.encoder, nn.GRU):
            h, _ = self.encoder(h)
        else:
            h = self.encoder(h)
        pooled = h[:, -1, :]
        out = self.head(pooled)
        return out.view(-1, self.forecast_horizon, len(self.quantiles))


# ---------------------------------------------------------------------------
# 训练流程
# ---------------------------------------------------------------------------
class Trainer:
    """模型训练流程封装：初始化模型、训练、保存。"""

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = load_config(config)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device) if isinstance(device, str) else (
            device if device is not None else self._resolve_device()
        )
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.criterion: Optional[nn.Module] = None
        self.quantiles: List[float] = self._resolve_quantiles(self.config)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss: float = math.inf
        self.best_epoch: int = -1
        self.current_epoch: int = 0
        self.checkpoint_path: Optional[str] = None
        self.best_state_dict: Optional[Dict[str, Any]] = None
        set_seed(
            _to_int(_cfg_get(self.config, "system.seed", 42), 42),
            bool(_cfg_get(self.config, "system.deterministic", True)),
        )

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _resolve_device(self) -> torch.device:
        want = (
            _cfg_get(self.config, "system.device", None)
            or _cfg_get(self.config, "inference.device", None)
            or "cuda"
        )
        want = str(want)
        if want.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(want)

    @staticmethod
    def _resolve_quantiles(cfg: Any) -> List[float]:
        raw = (
            _cfg_get(cfg, "model.quantiles", None)
            or _cfg_get(cfg, "training.quantiles", None)
            or (0.1, 0.5, 0.9)
        )
        result = [float(q) for q in raw]
        return result or [0.1, 0.5, 0.9]

    @staticmethod
    def _apply_weight_init(module: nn.Module, method: str) -> None:
        method = (method or "kaiming_normal").lower()

        def _init(layer: nn.Module) -> None:
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if method in {"kaiming_normal", "kaiming"}:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif method in {"xavier_normal", "xavier"}:
                    nn.init.xavier_normal_(layer.weight)
                elif method in {"normal", "gaussian"}:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        module.apply(_init)

    def _build_optimizer(self, cfg: Any) -> torch.optim.Optimizer:
        name = str(_cfg_get(cfg, "training.optimizer.name", None) or "adam").lower()
        lr = _to_float(
            _cfg_get(cfg, "training.optimizer.lr", None) or _cfg_get(cfg, "training.learning_rate", 1e-3),
            1e-3,
        )
        weight_decay = _to_float(_cfg_get(cfg, "training.optimizer.weight_decay", None), 0.0)
        params = self.model.parameters()
        if name == "sgd":
            momentum = _to_float(_cfg_get(cfg, "training.optimizer.momentum", None), 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self, cfg: Any) -> Optional[Any]:
        name = _cfg_get(cfg, "training.scheduler.name", None)
        if name is None or self.optimizer is None:
            return None
        name = str(name).lower()
        if name in {"steplr", "step"}:
            step_size = _to_int(_cfg_get(cfg, "training.scheduler.step_size", None), 10)
            gamma = _to_float(_cfg_get(cfg, "training.scheduler.gamma", None), 0.1)
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return None

    def _build_criterion(self, cfg: Any) -> nn.Module:
        return QuantileLoss(self.quantiles)

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------
    def _resolve_model_class(self, name: str) -> Optional[Any]:
        module_candidates = (
            "models.tft",
            "models.tft_model",
            "models.temporal_fusion_transformer",
            "models.transformer",
            "models.lstm",
        )
        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            cls = getattr(module, name, None)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls
        if Lstm is not None:
            return Lstm
        return _FallbackQuantileModel

    def _build_model(self, cfg: Any, **kwargs: Any) -> nn.Module:
        model_name = str(_cfg_get(cfg, "model.name", "tft") or "tft")
        model_cls = self._resolve_model_class(model_name)
        if model_cls is None:
            model_cls = _FallbackQuantileModel

        build_kwargs: Dict[str, Any] = {
            "config": cfg,
            "quantiles": self.quantiles,
            "input_window": _cfg_get(cfg, "model.input_window", None),
            "forecast_horizon": _cfg_get(cfg, "model.forecast_horizon", None),
            "hidden_size": _cfg_get(cfg, "model.hidden_size", None),
            "num_layers": _cfg_get(cfg, "model.num_lstm_layers", None),
            "dropout_ratio": _cfg_get(cfg, "model.dropout_ratio", None),
            "target_dim": _cfg_get(cfg, "model.output_dim", None),
        }
        build_kwargs.update(kwargs)
        attempts: List[Dict[str, Any]] = [build_kwargs, {"config": cfg}, {}]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return model_cls(**attempt)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
        if isinstance(last_error, TypeError):
            return _FallbackQuantileModel(config=cfg, quantiles=self.quantiles)
        raise RuntimeError(f"构建模型 {model_name} 失败: {last_error}")

    # ------------------------------------------------------------------
    # 核心方法 1：initialize_model
    # ------------------------------------------------------------------
    def initialize_model(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """构建 / 加载模型，并准备优化器、调度器与损失函数。"""
        if config is not None:
            self.config = load_config(config)
        cfg = self.config

        if model is None:
            model = self._build_model(cfg, **kwargs)
        self.model = model

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

        init_method = str(_cfg_get(cfg, "model.weight_init", "kaiming_normal") or "kaiming_normal")
        self._apply_weight_init(self.model, init_method)

        self.quantiles = self._resolve_quantiles(cfg)
        self.optimizer = self._build_optimizer(cfg)
        self.scheduler = self._build_scheduler(cfg)
        self.criterion = self._build_criterion(cfg)

        logger.info("模型已初始化: %s | device=%s", type(self.model).__name__, self.device)
        return self.model

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    @staticmethod
    def _split_batch(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        targets = None
        for key in ("targets", "target", "y", "labels", "label"):
            if key in batch and torch.is_tensor(batch[key]):
                targets = batch[key]
                break
        drop = {"target", "y", "labels", "label"}
        kwargs = {k: v for k, v in batch.items() if k not in drop}
        if targets is not None:
            kwargs["targets"] = targets
        return kwargs, targets

    def _forward_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            kwargs, targets = self._split_batch(batch)
            try:
                preds = self.model(**kwargs)
            except TypeError:
                preds = self.model(batch)
        elif isinstance(batch, (list, tuple)):
            tensors = [t.to(self.device) if torch.is_tensor(t) else t for t in batch]
            targets = None
            for item in tensors:
                if torch.is_tensor(item):
                    targets = item
            try:
                preds = self.model(*tensors)
            except TypeError:
                preds = self.model(tensors[0] if tensors else None)
        else:
            tensors = batch.to(self.device) if torch.is_tensor(batch) else batch
            preds = self.model(tensors)
            targets = None

        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return preds, targets

    def _build_loaders(
        self,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        train_dataset: Optional[Dataset],
        val_dataset: Optional[Dataset],
        cfg: Any,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        dataset_cls = (
            ElectricityUCIElectricityLoadDiagramsTrafficUCIPEMSFTrafficRetailFavoritaGrocerySalesFromKaggleVolatilityOxfordManInstituteRealizedLibraryDataset
        )
        batch_size = _to_int(_cfg_get(cfg, "training.batch_size", 64), 64)
        num_workers = _to_int(_cfg_get(cfg, "system.num_workers", 0), 0, minimum=0)
        pin_memory = bool(_cfg_get(cfg, "system.pin_memory", False))
        input_window = _cfg_get(cfg, "model.input_window", None) or _cfg_get(cfg, "data.input_window", None)
        forecast_horizon = _cfg_get(cfg, "model.forecast_horizon", None) or _cfg_get(
            cfg, "data.forecast_horizon", None
        )
        dataset_name = (
            _cfg_get(cfg, "experiment.dataset", None)
            or _cfg_get(cfg, "data.dataset", None)
            or _cfg_get(cfg, "data.dataset_version", None)
        )

        def _make_dataset(split: str) -> Optional[Dataset]:
            if dataset_cls is None:
                return None
            try:
                return dataset_cls(
                    config=cfg,
                    dataset_name=dataset_name,
                    split=split,
                    input_window=input_window,
                    forecast_horizon=forecast_horizon,
                )
            except TypeError:
                try:
                    return dataset_cls(config=cfg, split=split)
                except Exception as exc:
                    logger.warning("构建 %s 数据集失败: %s", split, exc)
                    return None
            except Exception as exc:
                logger.warning("构建 %s 数据集失败: %s", split, exc)
                return None

        if train_loader is None:
            if train_dataset is None:
                train_dataset = _make_dataset("train")
            if train_dataset is not None:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
        if val_loader is None:
            if val_dataset is None:
                val_dataset = _make_dataset("val")
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # 单轮训练 / 验证
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor],
        batch: Any = None,
    ) -> torch.Tensor:
        if targets is None and isinstance(batch, dict):
            for key in ("targets", "target", "y", "labels", "label"):
                if key in batch and torch.is_tensor(batch[key]):
                    targets = batch[key]
                    break
        if targets is None:
            raise RuntimeError("batch 中未找到 targets，无法计算损失。")
        targets = targets.to(preds.device).float()
        if self.criterion is not None:
            return self.criterion(preds, targets)
        return quantile_loss(preds, targets, self.quantiles)

    def _train_one_epoch(self, loader: DataLoader, epoch: int, log_interval: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        clip_norm = _to_float(_cfg_get(self.config, "training.grad_clip_norm", None), 0.0)
        for step, batch in enumerate(loader, start=1):
            if self.optimizer is None:
                raise RuntimeError("优化器未初始化，请先调用 initialize_model。")
            self.optimizer.zero_grad()
            preds, targets = self._forward_batch(batch)
            loss = self._compute_loss(preds, targets, batch)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optimizer.step()
            total += float(loss.detach().item())
            count += 1
            if log_interval > 0 and step % log_interval == 0:
                logger.info("epoch %d step %d train_loss=%.6f", epoch, step, total / max(count, 1))
        return total / max(count, 1)

    @torch.no_grad()
    def _validate(self, loader: Optional[DataLoader]) -> float:
        if loader is None:
            return math.nan
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            preds, targets = self._forward_batch(batch)
            loss = self._compute_loss(preds, targets, batch)
            total += float(loss.detach().item())
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # 核心方法 2：train_model
    # ------------------------------------------------------------------
    def train_model(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        model: Optional[nn.Module] = None,
        config: Optional[Any] = None,
        epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行训练与验证循环（含早停与最优模型选择）。"""
        if config is not None:
            self.config = load_config(config)
        cfg = self.config

        if model is not None:
            self.model = model
        if self.model is None or self.optimizer is None or self.criterion is None:
            self.initialize_model(model=self.model, config=cfg, device=self.device, **kwargs)

        self.train_loader, self.val_loader = self._build_loaders(
            train_loader if train_loader is not None else self.train_loader,
            val_loader if val_loader is not None else self.val_loader,
            None,
            None,
            cfg,
        )
        if self.train_loader is None:
            raise RuntimeError("训练数据加载器不可用：请传入 train_loader 或提供可构建的数据集。")

        total_epochs = (
            _to_int(epochs, 1) if epochs is not None
            else _to_int(_cfg_get(cfg, "training.epochs", 30), 30)
        )
        log_interval = _to_int(_cfg_get(cfg, "logging.log_interval", 100), 100, minimum=0)

        es_cfg = cfg.get("early_stopping", {}) if isinstance(cfg, dict) else {}
        es_enabled = bool(_cfg_get(cfg, "training.early_stopping.enabled", True))
        patience = _to_int(_cfg_get(cfg, "training.early_stopping.patience", 5), 5)
        min_delta = _to_float(_cfg_get(cfg, "training.early_stopping.min_delta", 0.0), 0.0)
        if isinstance(es_cfg, dict):
            if es_cfg.get("patience") is not None:
                patience = _to_int(es_cfg.get("patience"), patience)
            if es_cfg.get("min_delta") is not None:
                min_delta = _to_float(es_cfg.get("min_delta"), min_delta)

        best_mode = str(_cfg_get(cfg, "checkpoint.best_mode", "min") or "min").lower()
        save_best = bool(_cfg_get(cfg, "checkpoint.save_best", True))

        bad_epochs = 0
        for epoch in range(1, total_epochs + 1):
            self.current_epoch = epoch
            start = time.time()
            train_loss = self._train_one_epoch(self.train_loader, epoch, log_interval)
            val_loss = self._validate(self.val_loader)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))

            improved = False
            if val_loss == val_loss:  # not NaN
                if best_mode == "max":
                    improved = val_loss > (self.best_val_loss if self.best_val_loss != -math.inf else -math.inf) + min_delta
                    if improved:
                        self.best_val_loss = val_loss
                else:
                    improved = val_loss < self.best_val_loss - min_delta
                    if improved:
                        self.best_val_loss = val_loss
            if improved:
                self.best_epoch = epoch
                bad_epochs = 0
                if save_best:
                    try:
                        self.best_state_dict = {
                            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                        }
                    except Exception:
                        self.best_state_dict = None
            else:
                bad_epochs += 1

            logger.info(
                "epoch %d/%d train_loss=%.6f val_loss=%.6f time=%.2fs",
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                time.time() - start,
            )

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception as exc:
                    logger.warning("学习率调度器 step 失败: %s", exc)

            if es_enabled and self.val_loader is not None and bad_epochs >= patience:
                logger.info("早停触发（patience=%d），最优 epoch=%d", patience, self.best_epoch)
                break

        if self.best_state_dict is not None:
            try:
                self.model.load_state_dict(self.best_state_dict)
            except Exception as exc:
                logger.warning("加载最优权重失败: %s", exc)

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "model": self.model,
        }

    # ------------------------------------------------------------------
    # 核心方法 3：save_model
    # ------------------------------------------------------------------
    def save_model(
        self,
        path: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        save_optimizer_state: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        """保存模型权重与训练元信息。"""
        cfg = self.config
        if self.model is None:
            raise RuntimeError("模型尚未初始化，请先调用 initialize_model / train_model。")

        if path is None:
            ckpt_dir = str(_cfg_get(cfg, "checkpoint.checkpoint_dir", "./checkpoints") or "./checkpoints")
            pattern = str(
                _cfg_get(cfg, "checkpoint.filename_pattern", "best_{model}_{dataset}_{seed}.pt")
                or "best_{model}_{dataset}_{seed}.pt"
            )
            model_tag = str(_cfg_get(cfg, "model.name", "model") or "model")
            dataset_tag = str(
                _cfg_get(cfg, "experiment.dataset", None)
                or _cfg_get(cfg, "data.dataset_version", "default")
                or "default"
            )
            seed_tag = str(_to_int(_cfg_get(cfg, "system.seed", 42), 42))
            try:
                name = pattern.format(model=model_tag, dataset=dataset_tag, seed=seed_tag)
            except Exception:
                name = f"best_{model_tag}_{dataset_tag}_{seed_tag}.pt"
            if filename:
                name = filename
            out_path = Path(ckpt_dir) / name
        else:
            out_path = Path(path)
            if out_path.is_dir() or str(out_path).endswith(os.sep):
                out_path = out_path / (filename or "model.pt")
            elif filename:
                out_path = out_path.parent / filename

        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_opt = (
            bool(_cfg_get(cfg, "checkpoint.save_optimizer_state", True))
            if save_optimizer_state is None
            else bool(save_optimizer_state)
        )
        payload: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "model_name": type(self.model).__name__,
            "quantiles": list(self.quantiles),
            "epoch": self.current_epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": cfg,
        }
        if save_opt and self.optimizer is not None:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            try:
                payload["scheduler_state_dict"] = self.scheduler.state_dict()
            except Exception:
                pass

        torch.save(payload, str(out_path))
        self.checkpoint_path = str(out_path)
        logger.info("模型已保存到 %s", self.checkpoint_path)
        return self.checkpoint_path


# ---------------------------------------------------------------------------
# 模块级接口（与 __all__ 保持一致）
# ---------------------------------------------------------------------------
_DEFAULT_TRAINER: Optional[Trainer] = None


def _get_trainer(trainer: Optional[Trainer] = None, **kwargs: Any) -> Trainer:
    global _DEFAULT_TRAINER
    if trainer is not None:
        _DEFAULT_TRAINER = trainer
        return trainer
    if _DEFAULT_TRAINER is None:
        _DEFAULT_TRAINER = Trainer(**kwargs)
    return _DEFAULT_TRAINER


def initialize_model(
    model: Optional[nn.Module] = None,
    config: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> nn.Module:
    """构建 / 加载模型，并准备优化器、调度器与损失函数。"""
    global _DEFAULT_TRAINER
    trainer = Trainer(config=config, model=model, device=device)
    built = trainer.initialize_model(model=model, config=config, device=device, **kwargs)
    _DEFAULT_TRAINER = trainer
    return built


def train_model(
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    model: Optional[nn.Module] = None,
    config: Optional[Any] = None,
    trainer: Optional[Trainer] = None,
    epochs: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """执行训练与验证循环（含早停与最优模型选择）。"""
    active = _get_trainer(trainer, config=config, model=model)
    return active.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config=config,
        epochs=epochs,
        **kwargs,
    )


def save_model(
    path: Optional[Union[str, Path]] = None,
    trainer: Optional[Trainer] = None,
    filename: Optional[str] = None,
    save_optimizer_state: Optional[bool] = None,
    **kwargs: Any,
) -> str:
    """保存模型权重与训练元信息。"""
    active = _get_trainer(trainer)
    return active.save_model(
        path=path,
        filename=filename,
        save_optimizer_state=save_optimizer_state,
        **kwargs,
    )