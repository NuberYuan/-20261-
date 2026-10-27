# -*- coding: utf-8 -*-
"""数据集加载、预处理和划分。

支持 Electricity / Traffic / Retail / Volatility 四类时间序列数据集：

* 读取本地数据文件（csv/tsv/parquet/json/pkl/npy）；
* ``preprocess_data`` 完成时间列解析、数值化、排序与 z-score 标准化；
* ``split_dataset`` 按各数据集的时间边界划分 train / val / test；
* ``__getitem__`` 按 ``input_window`` / ``forecast_horizon`` 返回张量样本。
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:  # pragma: no cover - pyyaml 为可选依赖
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


__all__ = [
    "ElectricityUCIElectricityLoadDiagramsTrafficUCIPEMSFTrafficRetailFavoritaGrocerySalesFromKaggleVolatilityOxfordManInstituteRealizedLibraryDataset",
]

_MISSING = object()
_DEFAULT_COLUMNS = ("timestamp", "group_id", "target")
_TABLE_PATTERNS = ("*.csv", "*.tsv", "*.parquet", "*.pq", "*.json", "*.pkl", "*.pickle", "*.npy")

_DATASET_ALIASES = {
    "electricity": "Electricity",
    "electricityuci": "Electricity",
    "electricityload": "Electricity",
    "electricityloaddiagrams": "Electricity",
    "electricityucielectricityloaddiagrams": "Electricity",
    "traffic": "Traffic",
    "trafficuci": "Traffic",
    "pemsf": "Traffic",
    "pems": "Traffic",
    "pemsft": "Traffic",
    "pemsfttraffic": "Traffic",
    "retail": "Retail",
    "favorita": "Retail",
    "retailfavorita": "Retail",
    "retailfavoritagrocerysales": "Retail",
    "retailfavoritagrocerysalesfromkaggle": "Retail",
    "volatility": "Volatility",
    "oxfordman": "Volatility",
    "oxfordmaninstitute": "Volatility",
    "realizedlibrary": "Volatility",
    "volatilityoxfordmaninstituterealizedlibrary": "Volatility",
}


# ---------------------------------------------------------------------------
# 配置 / 文件工具函数
# ---------------------------------------------------------------------------
def _load_config_object(config: Any) -> Dict[str, Any]:
    """把 config（dict / yaml 路径 / 目录 / None）转换为嵌套 dict。"""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, os.PathLike)):
        path = str(config)
        if os.path.isdir(path):
            for fname in ("config.yaml", "config.yml", "config.json"):
                candidate = os.path.join(path, fname)
                if os.path.isfile(candidate):
                    path = candidate
                    break
        if not os.path.isfile(path):
            return {}
        if path.endswith(".json"):
            import json

            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
        if yaml is None:
            warnings.warn("未安装 pyyaml，无法读取配置文件 %s，使用默认配置。" % path)
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """支持点号嵌套路径与扁平点号键两种写法的配置读取。"""
    if not isinstance(cfg, dict):
        return default
    if key in cfg:
        return cfg[key]
    node = cfg
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def _normalise_dataset_name(name: Any) -> str:
    """把各种数据集别名归一化为 Electricity / Traffic / Retail / Volatility。"""
    if name is None:
        return "Electricity"
    key = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not key:
        return "Electricity"
    if key in _DATASET_ALIASES:
        return _DATASET_ALIASES[key]
    for alias, canonical in _DATASET_ALIASES.items():
        if key.startswith(alias):
            return canonical
    for alias, canonical in _DATASET_ALIASES.items():
        if alias.startswith(key):
            return canonical
    raise ValueError("未知数据集名称: %s" % name)


def _read_table(path: str) -> pd.DataFrame:
    """按扩展名读取单个数据文件。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".json":
        return pd.read_json(path)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if ext == ".npy":
        return pd.DataFrame(np.load(path, allow_pickle=True))
    raise ValueError("暂不支持的数据文件格式: %s" % path)


def _load_dataframe_from_path(path: Any) -> pd.DataFrame:
    """从文件或目录加载数据，目录下递归读取常见表格文件。"""
    if path is None:
        return pd.DataFrame(columns=list(_DEFAULT_COLUMNS))
    path = str(path)
    if os.path.isfile(path):
        return _read_table(path)
    if os.path.isdir(path):
        files: List[str] = []
        for pattern in _TABLE_PATTERNS:
            files.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        files = sorted(set(files))
        if not files:
            return pd.DataFrame(columns=list(_DEFAULT_COLUMNS))
        frames = [_read_table(item) for item in files]
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=list(_DEFAULT_COLUMNS))


def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """把配置中的日期字符串解析为 pandas Timestamp。"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(str(value))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 主数据集类
# ---------------------------------------------------------------------------
class ElectricityUCIElectricityLoadDiagramsTrafficUCIPEMSFTrafficRetailFavoritaGrocerySalesFromKaggleVolatilityOxfordManInstituteRealizedLibraryDataset(Dataset):
    """通用时间序列数据集。

    同时支持 Electricity (UCI Electricity Load Diagrams)、Traffic (UCI PEMS-SF Traffic)、
    Retail (Favorita Grocery Sales from Kaggle)、Volatility (Oxford-Man Institute Realized Library)
    四种数据集，通过 ``dataset_name`` 或配置文件中的数据集键进行区分。
    """

    def __init__(
        self,
        config: Any = None,
        dataset_name: Optional[str] = None,
        data_path: Optional[str] = None,
        split: str = "train",
        input_window: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        timestamp_column: Optional[str] = None,
        group_id_column: Optional[str] = None,
        target_column: Optional[str] = None,
        normalization: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.cfg = _load_config_object(config)
        self.dataset_name = _normalise_dataset_name(
            dataset_name if dataset_name is not None else _cfg_get(self.cfg, "dataset_name")
        )

        self.timestamp_column = (
            timestamp_column
            or _cfg_get(self.cfg, "data.timestamp_column")
            or _cfg_get(self.cfg, "data_timestamp_column")
            or "timestamp"
        )
        self.group_id_column = (
            group_id_column
            or _cfg_get(self.cfg, "data.group_id_column")
            or _cfg_get(self.cfg, "data_group_id_column")
            or "group_id"
        )
        self.target_column = (
            target_column
            or _cfg_get(self.cfg, "data.target_column")
            or _cfg_get(self.cfg, "data_target_column")
            or "target"
        )
        self.normalization = (
            normalization
            or _cfg_get(self.cfg, "data.normalization")
            or _cfg_get(self.cfg, "data_normalization")
            or "standard"
        )
        self.input_window = input_window if input_window is not None else (
            _cfg_get(self.cfg, "data.input_window") or _cfg_get(self.cfg, "data_input_window") or 168
        )
        self.forecast_horizon = forecast_horizon if forecast_horizon is not None else (
            _cfg_get(self.cfg, "data.forecast_horizon") or _cfg_get(self.cfg, "data_forecast_horizon") or 24
        )
        self.seed = seed
        self.split = split

        ds_key = "data.datasets.%s.path" % self.dataset_name
        resolved_path = (
            data_path
            or _cfg_get(self.cfg, ds_key)
            or _cfg_get(self.cfg, "data.datasets.%s" % self.dataset_name + ".path")
        )
        if resolved_path is None:
            root = _cfg_get(self.cfg, "data.root") or "./data"
            resolved_path = os.path.join(root, "tft", "default", self.dataset_name.lower())
        self.data_path = str(resolved_path)

        self.raw_df = _load_dataframe_from_path(self.data_path)
        self.df = self.preprocess_data(self.raw_df.copy())
        self.train_df, self.val_df, self.test_df = self.split_dataset(self.df)
        self.active_df = self._get_split_df(self.split)
        self._build_index()
        self._fit_normalizer()
        self._normalise_active()

    # ------------------------------------------------------------------
    def _get_split_df(self, split: str) -> pd.DataFrame:
        split = (split or "train").lower()
        if split in ("train", "training"):
            return self.train_df
        if split in ("val", "valid", "validation"):
            return self.val_df if self.val_df is not None and len(self.val_df) > 0 else self.train_df
        return self.test_df if self.test_df is not None and len(self.test_df) > 0 else self.train_df

    # ------------------------------------------------------------------
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间列解析、数值化、排序与基础清洗。"""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=list(_DEFAULT_COLUMNS))

        df = df.copy()

        ts_col = self.timestamp_column if self.timestamp_column in df.columns else None
        if ts_col is None:
            for candidate in ("timestamp", "time", "date", "datetime", "ds", "ds_", "index"):
                if candidate in df.columns:
                    ts_col = candidate
                    break
        if ts_col is not None:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            if ts_col != self.timestamp_column:
                df[self.timestamp_column] = df[ts_col]

        grp_col = self.group_id_column if self.group_id_column in df.columns else None
        if grp_col is None:
            for candidate in ("group_id", "id", "entity", "store_id", "sku", "symbol"):
                if candidate in df.columns:
                    grp_col = candidate
                    break
        if grp_col is not None and grp_col != self.group_id_column:
            df[self.group_id_column] = df[grp_col]

        tgt_col = self.target_column if self.target_column in df.columns else None
        if tgt_col is None:
            for candidate in ("target", "value", "y", "sales", "close", "rv", "load"):
                if candidate in df.columns:
                    tgt_col = candidate
                    break
        if tgt_col is not None and tgt_col != self.target_column:
            df[self.target_column] = df[tgt_col]

        if self.timestamp_column not in df.columns:
            df[self.timestamp_column] = pd.date_range("2000-01-01", periods=len(df), freq="h")
        if self.group_id_column not in df.columns:
            df[self.group_id_column] = 0
        if self.target_column not in df.columns:
            df[self.target_column] = np.nan

        df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce")
        df = df.dropna(subset=[self.target_column])

        df = df.sort_values([self.group_id_column, self.timestamp_column]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按数据集时间边界划分 train / val / test。"""
        if df is None or len(df) == 0:
            empty = pd.DataFrame(columns=list(_DEFAULT_COLUMNS))
            return empty, empty.copy(), empty.copy()

        ds = self.dataset_name

        if ds == "Electricity":
            end_train_ts = _to_timestamp(_cfg_get(self.cfg, "data.datasets.Electricity.period_end"))
            if end_train_ts is not None and self.timestamp_column in df.columns:
                train_df = df[df[self.timestamp_column] < end_train_ts]
                rest = df[df[self.timestamp_column] >= end_train_ts]
            else:
                n = len(df)
                cut = int(n * 0.9)
                train_df, rest = df.iloc[:cut], df.iloc[cut:]
            n_rest = len(rest)
            val_df = rest.iloc[: max(n_rest // 2, 1)] if n_rest > 0 else rest.copy()
            test_df = rest.iloc[max(n_rest // 2, 1):] if n_rest > 1 else rest.copy()
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

        if ds == "Traffic":
            end_ts = _to_timestamp(_cfg_get(self.cfg, "data.datasets.Traffic.period_end"))
            if end_ts is not None and self.timestamp_column in df.columns:
                train_df = df[df[self.timestamp_column] < end_ts]
                rest = df[df[self.timestamp_column] >= end_ts]
            else:
                n = len(df)
                cut = int(n * 0.9)
                train_df, rest = df.iloc[:cut], df.iloc[cut:]
            n_rest = len(rest)
            val_df = rest.iloc[: max(n_rest // 2, 1)] if n_rest > 0 else rest.copy()
            test_df = rest.iloc[max(n_rest // 2, 1):] if n_rest > 1 else rest.copy()
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

        if ds == "Retail":
            train_end = _to_timestamp(_cfg_get(self.cfg, "data.datasets.Retail.train_period_end"))
            if train_end is not None and self.timestamp_column in df.columns:
                train_df = df[df[self.timestamp_column] < train_end]
                rest = df[df[self.timestamp_column] >= train_end]
            else:
                n = len(df)
                cut = int(n * 0.9)
                train_df, rest = df.iloc[:cut], df.iloc[cut:]
            val_n = int(_cfg_get(self.cfg, "data.datasets.Retail.val_num_samples") or 50000)
            val_df = rest.iloc[:val_n] if len(rest) > 0 else rest.copy()
            test_df = rest.iloc[val_n:] if len(rest) > val_n else rest.copy()
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

        if ds == "Volatility":
            train_end = _to_timestamp(_cfg_get(self.cfg, "data.datasets.Volatility.train_end")) or _to_timestamp("2016-01-01")
            val_end = _to_timestamp(_cfg_get(self.cfg, "data.datasets.Volatility.val_end")) or _to_timestamp("2018-01-01")
            ts = pd.to_datetime(df[self.timestamp_column], errors="coerce") if self.timestamp_column in df.columns else None
            if ts is not None:
                train_df = df[ts < train_end]
                val_df = df[(ts >= train_end) & (ts < val_end)]
                test_df = df[ts >= val_end]
            else:
                n = len(df)
                cut1, cut2 = int(n * 0.7), int(n * 0.85)
                train_df, val_df, test_df = df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

        # 默认 80/10/10
        n = len(df)
        c1 = int(n * 0.8)
        c2 = int(n * 0.9)
        return (
            df.iloc[:c1].reset_index(drop=True),
            df.iloc[c1:c2].reset_index(drop=True),
            df.iloc[c2:].reset_index(drop=True),
        )

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        self._index: List[Tuple[int, int]] = []
        if self.active_df is None or len(self.active_df) == 0:
            return
        win = int(self.input_window)
        hor = int(self.forecast_horizon)
        group_vals = self.active_df[self.group_id_column].values
        start = 0
        n = len(self.active_df)
        for i in range(1, n + 1):
            if i == n or group_vals[i] != group_vals[start]:
                seg_len = i - start
                if seg_len >= win + hor:
                    for s in range(start, i - win - hor + 1):
                        self._index.append((s, s + win + hor))
                start = i

    # ------------------------------------------------------------------
    def _fit_normalizer(self) -> None:
        src = self.train_df if self.train_df is not None and len(self.train_df) > 0 else self.active_df
        if src is None or len(src) == 0:
            self._mean = 0.0
            self._std = 1.0
            return
        vals = src[self.target_column].astype(float).values
        self._mean = float(np.mean(vals))
        self._std = float(np.std(vals))
        if self._std == 0 or np.isnan(self._std):
            self._std = 1.0

    def _normalise_active(self) -> None:
        if self.active_df is not None and len(self.active_df) > 0:
            self.active_df = self.active_df.copy()
            self.active_df[self.target_column] = (
                self.active_df[self.target_column].astype(float).values - self._mean
            ) / self._std

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self._index):
            raise IndexError("index %d out of range" % idx)
        s, e = self._index[idx]
        win = int(self.input_window)
        hor = int(self.forecast_horizon)
        window_df = self.active_df.iloc[s : s + win]
        future_df = self.active_df.iloc[s + win : s + win + hor]

        x = torch.tensor(window_df[self.target_column].values, dtype=torch.float32)
        y = torch.tensor(future_df[self.target_column].values, dtype=torch.float32)
        return {"x": x, "y": y, "input_window": torch.tensor([win]), "forecast_horizon": torch.tensor([hor])}