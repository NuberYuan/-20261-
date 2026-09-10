import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class ElectricityTrafficRetailVolatilityDataset(Dataset):
    """Dataset loader and preprocessor for Electricity, Traffic, Retail and Volatility.

    The dataset supports the ``default`` version from ``config/config.yaml``.  If no
    local CSV file is found, synthetic data is generated so that the dataset object can
    still be used for development and shape checks.
    """

    def __init__(
        self,
        version: str = "default",
        config: Optional[Dict[str, Any]] = None,
        dataset_name: str = "Electricity",
        split: str = "train",
        lookback: int = 168,
        horizon: int = 24,
        local_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.version = version
        self.dataset_name = dataset_name
        self.split = split
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.local_path = local_path

        self.config = config if config is not None else self._load_config()
        self.root = (self.config.get("data", {}) or {}).get("root", None)
        self.seed = int((self.config.get("system", {}) or {}).get("seed", 42))
        self._is_synthetic = False

        if dataset_name not in self._dataset_meta():
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        self.meta = self._dataset_meta()[dataset_name]
        self.steps_per_day = int(self.meta.get("steps_per_day", 24))

        self.raw_data = self._load_raw_data()
        self.processed_data = self.preprocess_data(self.raw_data)
        self.splits = self.split_dataset(self.processed_data)
        self.windows = self._build_windows(self.splits[self.split], self.split)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config() -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        if not config_path.exists():
            logger.warning("config/config.yaml not found; using empty config.")
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_meta() -> Dict[str, Dict[str, Any]]:
        return {
            "Electricity": {
                "n_entities": 370,
                "n_samples": 500000,
                "steps_per_day": 24,
                "target_raw": "electricity_usage",
                "target_col": "electricity_usage",
                "real_cols": ["electricity_usage", "time_idx"],
                "cat_cols": ["entity_id", "day_of_week", "hour_of_day"],
                "normalization": "per_entity",
                "log_transform": False,
                "split_mode": "temporal_90_7",
            },
            "Traffic": {
                "n_entities": 440,
                "n_samples": 500000,
                "steps_per_day": 24,
                "target_raw": "traffic_occupancy",
                "target_col": "traffic_occupancy",
                "real_cols": ["traffic_occupancy", "time_idx"],
                "cat_cols": ["entity_id", "day_of_week", "hour_of_day"],
                "normalization": "global",
                "log_transform": False,
                "split_mode": "temporal_90_7",
            },
            "Retail": {
                "n_entities": 130000,
                "n_samples": 500000,
                "steps_per_day": 1,
                "target_raw": "sales",
                "target_col": "log_sales",
                "real_cols": ["log_sales", "transactions", "oil"],
                "cat_cols": [
                    "entity_id",
                    "day_of_week",
                    "open_flag",
                    "holiday_national",
                    "holiday_regional",
                    "holiday_local",
                ],
                "normalization": "global",
                "log_transform": True,
                "split_mode": "retail",
            },
            "Volatility": {
                "n_entities": 41,
                "n_samples": 200000,
                "steps_per_day": 1,
                "target_raw": "rv5_ss",
                "target_col": "log_rv5_ss",
                "real_cols": ["log_rv5_ss", "open_to_close_return"],
                "cat_cols": ["entity_id", "day_of_week", "month", "region"],
                "normalization": "global",
                "log_transform": True,
                "split_mode": "temporal_volatility",
            },
        }

    # ------------------------------------------------------------------
    # Raw data loading
    # ------------------------------------------------------------------
    def _load_raw_data(self) -> pd.DataFrame:
        def _read_csv(path: Path) -> pd.DataFrame:
            return pd.read_csv(path, encoding="utf-8")

        if self.local_path is not None:
            p = Path(self.local_path)
            if p.exists():
                self._is_synthetic = False
                return _read_csv(p)

        if self.root is not None:
            root = Path(self.root)
            version_dir = root / self.version / f"{self.dataset_name.lower()}.csv"
            plain_path = root / f"{self.dataset_name.lower()}.csv"
            if version_dir.exists():
                self._is_synthetic = False
                return _read_csv(version_dir)
            if plain_path.exists():
                self._is_synthetic = False
                return _read_csv(plain_path)

        logger.info(
            "No local CSV found for %s (version=%s); generating synthetic raw data.",
            self.dataset_name,
            self.version,
        )
        self._is_synthetic = True
        return self._make_synthetic_raw()

    def _make_synthetic_raw(self) -> pd.DataFrame:
        meta = self.meta
        n = int(meta["n_samples"])
        n_entities = int(meta["n_entities"])

        rng_seed = self.seed + sum(ord(c) for c in self.dataset_name)
        rng = np.random.default_rng(rng_seed)

        entity_id = rng.integers(0, n_entities, size=n).astype(np.int64)
        time_idx = np.arange(n, dtype=np.int64)
        day_of_week = (time_idx % 7).astype(np.int64)
        if self.steps_per_day > 1:
            hour_of_day = (time_idx % self.steps_per_day).astype(np.int64)
        else:
            hour_of_day = np.zeros(n, dtype=np.int64)

        data: Dict[str, np.ndarray] = {
            "entity_id": entity_id,
            "time_idx": time_idx,
            "day_of_week": day_of_week,
            "hour_of_day": hour_of_day,
        }

        if self.dataset_name == "Electricity":
            data["electricity_usage"] = (
                100.0
                + 10.0 * np.sin(time_idx / 24.0)
                + rng.normal(0.0, 5.0, n)
            ).astype(np.float32)
        elif self.dataset_name == "Traffic":
            data["traffic_occupancy"] = (
                50.0
                + 5.0 * np.sin(time_idx / 24.0)
                + rng.normal(0.0, 3.0, n)
            ).astype(np.float32)
        elif self.dataset_name == "Retail":
            data["sales"] = rng.poisson(lam=100.0, size=n).astype(np.float32)
            data["transactions"] = rng.poisson(lam=80.0, size=n).astype(np.float32)
            data["oil"] = rng.normal(0.0, 1.0, n).astype(np.float32)
            data["open_flag"] = (day_of_week < 6).astype(np.int64)
            data["holiday_national"] = rng.binomial(1, 0.05, n).astype(np.int64)
            data["holiday_regional"] = rng.binomial(1, 0.10, n).astype(np.int64)
            data["holiday_local"] = rng.binomial(1, 0.15, n).astype(np.int64)
        elif self.dataset_name == "Volatility":
            data["rv5_ss"] = np.exp(rng.normal(0.0, 0.5, n)).astype(np.float32)
            data["open_to_close_return"] = rng.normal(0.0, 0.02, n).astype(
                np.float32
            )
            data["month"] = ((time_idx // 30) % 12).astype(np.int64)
            data["region"] = (entity_id % 5).astype(np.int64)
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess_data(self, raw_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if raw_data is None:
            raw_data = self._load_raw_data()

        df = raw_data.copy()
        meta = self.meta

        # Create target column from raw target if needed.
        if meta["target_col"] not in df.columns:
            raw_target = meta.get("target_raw", meta["target_col"])
            if raw_target in df.columns:
                numeric_raw = pd.to_numeric(df[raw_target], errors="coerce").fillna(0)
                if meta.get("log_transform", False):
                    df[meta["target_col"]] = np.log1p(numeric_raw.clip(lower=0))
                else:
                    df[meta["target_col"]] = numeric_raw
                if meta.get("log_transform", False) and raw_target != meta["target_col"]:
                    df = df.drop(columns=[raw_target])
            else:
                df[meta["target_col"]] = 0.0

        # Ensure all real-valued columns exist and are numeric.
        for col in meta["real_cols"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Ensure all categorical columns exist and are encoded as integer codes.
        for col in meta["cat_cols"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].astype("category").cat.codes.astype(np.int64)

        # Normalize real-valued inputs.
        if meta["normalization"] == "per_entity":
            for col in meta["real_cols"]:
                col_mean = df.groupby("entity_id")[col].transform("mean")
                col_std = df.groupby("entity_id")[col].transform("std").replace(0, 1e-8)
                df[col] = ((df[col] - col_mean) / (col_std + 1e-8)).astype(np.float32)
        else:
            for col in meta["real_cols"]:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if not np.isfinite(col_std) or col_std == 0:
                    col_std = 1e-8
                df[col] = ((df[col] - col_mean) / col_std).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split_dataset(
        self, dataset: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        if dataset is None:
            dataset = self.processed_data

        df = dataset.reset_index(drop=True)
        n = len(df)
        split_mode = self.meta.get("split_mode", "temporal_90_7")

        if split_mode == "temporal_90_7":
            # Electricity and Traffic: first 90 percent training; validation is set as
            # the remaining data after a 7-day test window immediately following the
            # training set.  This preserves a non-overlapping validation split while
            # still providing the required test horizon.
            train_end = int(n * 0.9)
            test_len = 7 * self.steps_per_day
            test_start = train_end
            test_end = min(n, test_start + test_len)
            val_start = min(n, test_start + test_len)
            val_end = n
            if val_start < train_end:
                val_start = train_end

            return {
                "train": df.iloc[:train_end],
                "val": df.iloc[val_start:val_end],
                "test": df.iloc[test_start:test_end],
            }

        if split_mode == "retail":
            train_end = min(450000, n)
            val_end = min(500000, n)
            test_len = 30 * self.steps_per_day
            test_start = val_end
            test_end = min(n, test_start + test_len)

            if test_end <= test_start:
                test_start = max(0, val_end - test_len)
                test_end = val_end

            return {
                "train": df.iloc[:train_end],
                "val": df.iloc[train_end:val_end],
                "test": df.iloc[test_start:test_end],
            }

        if split_mode == "temporal_volatility":
            train_end = int(n * 0.60)
            val_end = int(n * 0.85)
            return {
                "train": df.iloc[:train_end],
                "val": df.iloc[train_end:val_end],
                "test": df.iloc[val_end:],
            }

        # Safe fallback.
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)
        return {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:],
        }

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _build_windows(self, df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        if self._is_synthetic:
            return self._build_windows_global(df)
        return self._build_windows_grouped(df)

    def _build_windows_global(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df = df.sort_values("time_idx").reset_index(drop=True)
        real_cols = list(self.meta["real_cols"])
        cat_cols = list(self.meta["cat_cols"])
        target_col = self.meta["target_col"]

        real_vals = df[real_cols].to_numpy(np.float32)
        cat_vals = df[cat_cols].to_numpy(np.int64)
        target_vals = df[target_col].to_numpy(np.float32)

        entity_idx = cat_cols.index("entity_id") if "entity_id" in cat_cols else 0
        window_len = self.lookback + self.horizon
        stride = max(1, self.horizon)

        windows: List[Dict[str, Any]] = []

        if len(df) < window_len:
            pad_len = window_len - len(df)
            real_vals = np.pad(real_vals, ((0, pad_len), (0, 0)), mode="edge")
            cat_vals = np.pad(cat_vals, ((0, pad_len), (0, 0)), mode="edge")
            target_vals = np.pad(target_vals, (0, pad_len), mode="edge")
            windows.append(
                {
                    "x_real": real_vals[: self.lookback],
                    "x_cat": cat_vals[: self.lookback],
                    "y": target_vals[self.lookback : window_len],
                    "entity_id": int(cat_vals[0, entity_idx]),
                    "time_idx": int(df["time_idx"].iloc[0]),
                }
            )
            return windows

        for start in range(0, len(df) - window_len + 1, stride):
            end = start + window_len
            windows.append(
                {
                    "x_real": real_vals[start : start + self.lookback],
                    "x_cat": cat_vals[start : start + self.lookback],
                    "y": target_vals[start + self.lookback : end],
                    "entity_id": int(cat_vals[start, entity_idx]),
                    "time_idx": int(df["time_idx"].iloc[start]),
                }
            )

        return windows

    def _build_windows_grouped(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        real_cols = list(self.meta["real_cols"])
        cat_cols = list(self.meta["cat_cols"])
        target_col = self.meta["target_col"]

        entity_idx = cat_cols.index("entity_id") if "entity_id" in cat_cols else 0
        window_len = self.lookback + self.horizon
        stride = max(1, self.horizon)

        windows: List[Dict[str, Any]] = []

        for entity_id, group in df.groupby("entity_id"):
            group = group.sort_values("time_idx")
            if len(group) == 0:
                continue

            real_vals = group[real_cols].to_numpy(np.float32)
            cat_vals = group[cat_cols].to_numpy(np.int64)
            target_vals = group[target_col].to_numpy(np.float32)

            if len(group) < window_len:
                pad_len = window_len - len(group)
                real_vals = np.pad(real_vals, ((0, pad_len), (0, 0)), mode="edge")
                cat_vals = np.pad(cat_vals, ((0, pad_len), (0, 0)), mode="edge")
                target_vals = np.pad(target_vals, (0, pad_len), mode="edge")
                windows.append(
                    {
                        "x_real": real_vals[: self.lookback],
                        "x_cat": cat_vals[: self.lookback],
                        "y": target_vals[self.lookback : window_len],
                        "entity_id": int(cat_vals[0, entity_idx]),
                        "time_idx": int(group["time_idx"].iloc[0]),
                    }
                )
                continue

            for start in range(0, len(group) - window_len + 1, stride):
                end = start + window_len
                windows.append(
                    {
                        "x_real": real_vals[start : start + self.lookback],
                        "x_cat": cat_vals[start : start + self.lookback],
                        "y": target_vals[start + self.lookback : end],
                        "entity_id": int(cat_vals[start, entity_idx]),
                        "time_idx": int(group["time_idx"].iloc[start]),
                    }
                )

        return windows

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        return {
            "x_real": torch.tensor(window["x_real"], dtype=torch.float32),
            "x_cat": torch.tensor(window["x_cat"], dtype=torch.long),
            "y": torch.tensor(window["y"], dtype=torch.float32),
            "entity_id": torch.tensor(window["entity_id"], dtype=torch.long),
            "time_idx": torch.tensor(window["time_idx"], dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Useful shape properties
    # ------------------------------------------------------------------
    @property
    def num_real_features(self) -> int:
        return len(self.meta["real_cols"])

    @property
    def num_categorical_features(self) -> int:
        return len(self.meta["cat_cols"])

    @property
    def num_entities(self) -> int:
        return int(self.meta["n_entities"])