"""PlantVillage dataset utilities: loading, preprocessing and leaf-aware splitting."""

from __future__ import annotations

import csv
import os
import random
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_IMAGE_SIZE = 256
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".webp")

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


def _cfg_get(config: Optional[Dict[str, Any]], dotted_key: str, default: Any = None) -> Any:
    """Read a value from a config dict, supporting both nested and dotted keys."""
    if not config:
        return default
    if dotted_key in config:
        return config[dotted_key]
    current: Any = config
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return default if current is None else current


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _load_leaf_mapping(path: Optional[str]) -> Dict[str, str]:
    """Load an image -> leaf-id mapping from a CSV file (best effort, tolerant to column names)."""
    mapping: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return mapping
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return mapping
        fields = [f for f in reader.fieldnames if f]
        img_key: Optional[str] = None
        leaf_key: Optional[str] = None
        for field in fields:
            low = field.lower()
            if leaf_key is None and ("leaf" in low or low in ("id", "group", "cluster", "leaf_id")):
                leaf_key = field
            if img_key is None and any(tok in low for tok in ("image", "file", "name", "path", "img")):
                img_key = field
        if img_key is None or leaf_key is None or img_key == leaf_key:
            return mapping
        for row in reader:
            raw_img = (row.get(img_key) or "").strip()
            leaf = (row.get(leaf_key) or "").strip()
            if not raw_img or not leaf:
                continue
            base = os.path.basename(raw_img)
            stem = os.path.splitext(base)[0]
            for key in (raw_img, base, stem):
                mapping[key] = leaf
                mapping[key.lower()] = leaf
    return mapping


def _leaf_aware_split(
    indices: Sequence[int],
    samples: Sequence[Tuple[str, int]],
    leaf_ids: Optional[Sequence[Any]],
    ratio: float,
    rng: random.Random,
    stratify: bool = True,
) -> Tuple[List[int], List[int]]:
    """Split ``indices`` into two parts while keeping every leaf inside a single part."""
    indices = list(indices)
    if not indices:
        return [], []
    if len(indices) == 1:
        return indices, []

    groups: Dict[Any, List[int]] = defaultdict(list)
    for idx in indices:
        leaf = leaf_ids[idx] if leaf_ids is not None else "__img__%d__" % idx
        groups[leaf].append(idx)

    group_label: Dict[Any, int] = {}
    for leaf, members in groups.items():
        labels = [samples[i][1] for i in members]
        group_label[leaf] = Counter(labels).most_common(1)[0][0]

    if stratify:
        buckets: Dict[Any, List[Any]] = defaultdict(list)
        for leaf in groups:
            buckets[group_label[leaf]].append(leaf)
    else:
        buckets = {"__all__": list(groups.keys())}

    left: List[int] = []
    right: List[int] = []
    for key in sorted(buckets, key=str):
        leaves = list(buckets[key])
        rng.shuffle(leaves)
        total = sum(len(groups[leaf]) for leaf in leaves)
        target = float(total) * float(ratio)
        current = 0
        left_leaves: List[Any] = []
        right_leaves: List[Any] = []
        for leaf in leaves:
            members = groups[leaf]
            if current < target:
                left_leaves.append(leaf)
                current += len(members)
            else:
                right_leaves.append(leaf)
        if not right_leaves and len(leaves) > 1:
            movable = min(left_leaves, key=lambda leaf: len(groups[leaf]))
            left_leaves.remove(movable)
            right_leaves.append(movable)
        elif not left_leaves and len(leaves) > 1:
            movable = min(right_leaves, key=lambda leaf: len(groups[leaf]))
            right_leaves.remove(movable)
            left_leaves.append(movable)
        for leaf in left_leaves:
            left.extend(groups[leaf])
        for leaf in right_leaves:
            right.extend(groups[leaf])

    left.sort()
    right.sort()
    return left, right


class PlantVillageDataset(Dataset):
    """PlantVillage image dataset (color / grayscale / segmented versions)."""

    def __init__(
        self,
        root: Optional[str] = None,
        version: Optional[str] = None,
        split: Optional[str] = None,
        indices: Optional[Sequence[int]] = None,
        image_size: Optional[Union[int, Sequence[int]]] = None,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        normalize: Optional[bool] = None,
        transform: Optional[Any] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        leaf_mapping_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        val_ratio: Optional[float] = None,
        reproduction_mode: Optional[str] = None,
    ) -> None:
        cfg = config or {}
        self.config = cfg

        resolved_root = root if root is not None else _cfg_get(cfg, "data.root", "./data")
        self.root = os.path.expanduser(str(resolved_root))
        self.version = str(version if version is not None else _cfg_get(cfg, "data.default_version", "color")).lower()
        self.dataset_dir = self._resolve_dataset_dir(cfg)

        self.split = split
        self.image_size = self._resolve_image_size(image_size, cfg)
        self.mean = list(mean if mean is not None else _cfg_get(cfg, "data.mean", DEFAULT_MEAN))
        self.std = list(std if std is not None else _cfg_get(cfg, "data.std", DEFAULT_STD))
        self.normalize = bool(normalize if normalize is not None else _cfg_get(cfg, "data.normalize", True))
        self.transform = transform

        if seed is None:
            seed = _cfg_get(cfg, "split_strategy.seed", 42)
        self.seed = int(seed)

        if reproduction_mode is None:
            reproduction_mode = _cfg_get(cfg, "reproduction_mode", "engineering")
        self.reproduction_mode = str(reproduction_mode).lower()

        if val_ratio is None:
            val_ratio = _cfg_get(cfg, "split_strategy.val_split_ratio", 0.2)
        self.val_ratio = float(val_ratio) if val_ratio is not None else 0.0

        self.class_to_idx: Dict[str, int] = dict(class_to_idx) if class_to_idx else {}
        self.samples: List[Tuple[str, int]] = []
        self.idx_to_class: Dict[int, str] = {}
        self._load_samples()

        if leaf_mapping_file is None:
            leaf_mapping_file = _cfg_get(cfg, "split_strategy.leaf_mapping_file", None)
        self.leaf_mapping_file = leaf_mapping_file
        self.leaf_ids: List[Any] = self._load_leaf_ids(leaf_mapping_file)

        self.parent_indices: Optional[List[int]] = None
        if indices is not None:
            selected = [int(i) for i in indices]
            self.samples = [self.samples[i] for i in selected]
            self.leaf_ids = [self.leaf_ids[i] for i in selected]
            self.parent_indices = selected

        self.labels: List[int] = [label for _, label in self.samples]

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _resolve_dataset_dir(self, cfg: Dict[str, Any]) -> str:
        candidates: List[str] = []
        candidates.append(os.path.join(self.root, self.version))
        config_path = _cfg_get(cfg, "data.%s.path" % self.version, None)
        if config_path:
            candidates.append(os.path.expanduser(str(config_path)))
        dataset_name = _cfg_get(cfg, "data.dataset_name", None)
        if dataset_name:
            candidates.append(os.path.join(self.root, str(dataset_name), self.version))
            candidates.append(os.path.join(self.root, str(dataset_name)))
        candidates.append(os.path.join(self.root, self.version.upper()))
        candidates.append(self.root)

        for candidate in candidates:
            if os.path.isdir(candidate) and self._has_class_subfolders(candidate):
                return os.path.normpath(candidate)

        # No usable dataset directory is present on disk: fall back to the expected
        # location so that the object can still be constructed (smoke tests / lazy
        # data loading) instead of failing with FileNotFoundError.
        if config_path:
            fallback = os.path.expanduser(str(config_path))
        else:
            fallback = os.path.join(self.root, self.version)
        if not os.path.isdir(fallback):
            for candidate in candidates:
                if os.path.isdir(candidate):
                    fallback = candidate
                    break
        warnings.warn(
            "PlantVillage version '%s' was not found under '%s'; "
            "expecting the dataset at '%s'. The dataset will be empty until data is available."
            % (self.version, self.root, fallback),
            RuntimeWarning,
        )
        return os.path.normpath(fallback)

    @staticmethod
    def _has_class_subfolders(directory: str) -> bool:
        try:
            entries = os.listdir(directory)
        except OSError:
            return False
        for entry in entries:
            sub = os.path.join(directory, entry)
            if os.path.isdir(sub):
                try:
                    names = os.listdir(sub)
                except OSError:
                    continue
                if any(_is_image_file(name) for name in names):
                    return True
        return False

    @staticmethod
    def _resolve_image_size(image_size: Optional[Union[int, Sequence[int]]], cfg: Dict[str, Any]) -> Tuple[int, int]:
        if image_size is None:
            hw = _cfg_get(cfg, "data.image_size_hw", None)
            if isinstance(hw, (list, tuple)) and len(hw) == 2:
                return int(hw[0]), int(hw[1])
            height = _cfg_get(cfg, "data.resize.height", None)
            width = _cfg_get(cfg, "data.resize.width", None)
            if height is not None and width is not None:
                return int(height), int(width)
            image_size = _cfg_get(cfg, "data.image_size", DEFAULT_IMAGE_SIZE)
        if isinstance(image_size, int):
            return int(image_size), int(image_size)
        seq = list(image_size)
        if len(seq) == 1:
            return int(seq[0]), int(seq[0])
        return int(seq[0]), int(seq[1])

    def _load_samples(self) -> None:
        if not os.path.isdir(self.dataset_dir):
            self.samples = []
            if not self.class_to_idx:
                self.idx_to_class = {}
            else:
                self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
            return

        try:
            class_names = sorted(
                entry
                for entry in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, entry))
            )
        except OSError:
            class_names = []
        if not class_names:
            self.samples = []
            if not self.class_to_idx:
                self.idx_to_class = {}
            else:
                self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
            return

        if not self.class_to_idx:
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        samples: List[Tuple[str, int]] = []
        for name in class_names:
            if name not in self.class_to_idx:
                continue
            class_dir = os.path.join(self.dataset_dir, name)
            label = self.class_to_idx[name]
            for filename in sorted(os.listdir(class_dir)):
                if not _is_image_file(filename):
                    continue
                samples.append((os.path.join(class_dir, filename), label))
        self.samples = samples

    def _load_leaf_ids(self, leaf_mapping_file: Optional[str]) -> List[Any]:
        mapping = _load_leaf_mapping(leaf_mapping_file)
        leaf_ids: List[Any] = []
        for idx, (path, _) in enumerate(self.samples):
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]
            try:
                rel = os.path.relpath(path, self.dataset_dir)
            except ValueError:
                rel = base
            leaf = None
            for key in (base, stem, rel, base.lower(), stem.lower(), rel.lower()):
                if key in mapping:
                    leaf = mapping[key]
                    break
            leaf_ids.append(leaf if leaf is not None else "__img__%d__" % idx)
        return leaf_ids

    # ------------------------------------------------------------------ #
    # core Dataset API
    # ------------------------------------------------------------------ #
    def preprocess_image(self, image: Union[str, os.PathLike, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert an image (path / PIL image / ndarray / tensor) into a normalised CHW float tensor."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as handle:
                pil_image = handle.convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            if array.ndim != 3:
                raise ValueError("Unsupported ndarray shape: %s" % (array.shape,))
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            if array.shape[2] > 3:
                array = array[:, :, :3]
            if array.dtype != np.uint8:
                if array.max() <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array).convert("RGB")
        else:
            raise TypeError("Unsupported image type: %r" % type(image))

        height, width = self.image_size
        pil_image = pil_image.resize((int(width), int(height)), _BILINEAR)
        array = np.asarray(pil_image, dtype=np.float32) / 255.0

        if self.normalize:
            mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3)
            array = (array - mean) / std

        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        return torch.from_numpy(array).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError("Index %d out of range for dataset of size %d" % (index, len(self.samples)))
        path, label = self.samples[index]
        if self.transform is not None:
            with Image.open(path) as handle:
                pil_image = handle.convert("RGB")
            image = self.transform(pil_image)
        else:
            image = self.preprocess_image(path)
        return image, int(label)

    # ------------------------------------------------------------------ #
    # splitting
    # ------------------------------------------------------------------ #
    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        leaf_constraint: bool = True,
        reproduction_mode: Optional[str] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        """Leaf-aware train/val/test split (validation is carved out of the training set).

        Returns a dict with 'train' / 'val' / 'test' :class:`PlantVillageDataset` objects
        plus their index lists and metadata.
        """
        if val_ratio is None:
            val_ratio = self.val_ratio
        if reproduction_mode is None:
            reproduction_mode = self.reproduction_mode
        reproduction_mode = str(reproduction_mode).lower()
        if reproduction_mode == "strict":
            val_ratio = 0.0
        if seed is None:
            seed = self.seed
        rng = random.Random(int(seed))
        if not shuffle:
            rng = random.Random(0)

        all_indices = list(range(len(self.samples)))
        leaf_ids = self.leaf_ids if leaf_constraint else None

        train_idx, test_idx = _leaf_aware_split(
            all_indices, self.samples, leaf_ids, float(train_ratio), rng, bool(stratify)
        )
        val_idx: List[int] = []
        if val_ratio and val_ratio > 0.0 and len(train_idx) > 1:
            remaining, val_idx = _leaf_aware_split(
                train_idx, self.samples, leaf_ids, 1.0 - float(val_ratio), rng, bool(stratify)
            )
            train_idx = remaining

        train_ds = self._subset(train_idx, split="train")
        val_ds = self._subset(val_idx, split="val" if val_idx else None)
        test_ds = self._subset(test_idx, split="test")

        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "train_indices": list(train_idx),
            "val_indices": list(val_idx),
            "test_indices": list(test_idx),
            "class_to_idx": dict(self.class_to_idx),
            "idx_to_class": dict(self.idx_to_class),
            "num_classes": len(self.class_to_idx),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "reproduction_mode": reproduction_mode,
            "stratify": bool(stratify),
            "leaf_constraint": bool(leaf_constraint),
        }

    def _subset(self, indices: Sequence[int], split: Optional[str] = None) -> "PlantVillageDataset":
        subset = PlantVillageDataset.__new__(PlantVillageDataset)
        subset.__dict__.update(self.__dict__)
        selected = [int(i) for i in indices]
        subset.samples = [self.samples[i] for i in selected]
        subset.leaf_ids = [self.leaf_ids[i] for i in selected]
        subset.labels = [label for _, label in subset.samples]
        subset.split = split if split is not None else self.split
        subset.class_to_idx = dict(self.class_to_idx)
        subset.idx_to_class = dict(self.idx_to_class)
        subset.parent_indices = selected
        return subset

    # ------------------------------------------------------------------ #
    # convenience helpers
    # ------------------------------------------------------------------ #
    @property
    def classes(self) -> List[str]:
        return [self.idx_to_class[i] for i in sorted(self.idx_to_class)]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def image_paths(self) -> List[str]:
        return [path for path, _ in self.samples]

    def get_image_paths(self) -> List[str]:
        return self.image_paths

    def get_labels(self) -> List[int]:
        return list(self.labels)

    def __repr__(self) -> str:
        return (
            "PlantVillageDataset(root=%r, version=%r, split=%r, samples=%d, classes=%d)"
            % (self.root, self.version, self.split, len(self.samples), len(self.class_to_idx))
        )