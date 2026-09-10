"""Synthetic algorithmic tasks dataset for time-series prediction.

Implements the six synthetic tasks from:
    LEARNING SIMPLE ALGORITHMS FROM EXAMPLES

All data is generated programmatically inside this module.  No external
files are read.  The dataset class exposes a PyTorch ``Dataset`` interface
and a ``get_dataloaders`` helper that builds train/val/test loaders.

The six tasks are:
    * copy
    * reverse
    * walk
    * addition
    * 3-number addition
    * single digit multiplication

Curriculum learning is supported through the config key
``split_strategy.curriculum``.  The dataset itself stores generated
examples; the training loop is responsible for increasing complexity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:  # pragma: no cover - yaml is a project dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------
# The paper uses a small symbolic alphabet.  We use a compact vocabulary that
# covers digits, a few letters for the walk task, and special symbols.
_DIGITS = [str(i) for i in range(10)]
_WALK_SYMBOLS = ["N", "S", "E", "W"]
_SPECIAL = ["<pad>", "<sos>", "<eos>", "+", "*", "=", " ", "-"]

VOCAB: List[str] = _SPECIAL + _DIGITS + _WALK_SYMBOLS
PAD_IDX = VOCAB.index("<pad>")
SOS_IDX = VOCAB.index("<sos>")
EOS_IDX = VOCAB.index("<eos>")
SPACE_IDX = VOCAB.index(" ")
PLUS_IDX = VOCAB.index("+")
TIMES_IDX = VOCAB.index("*")
EQUALS_IDX = VOCAB.index("=")
MINUS_IDX = VOCAB.index("-")

CHAR_TO_IDX: Dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
IDX_TO_CHAR: Dict[int, str] = {i: ch for i, ch in enumerate(VOCAB)}


# ---------------------------------------------------------------------------
# Task generation helpers
# ---------------------------------------------------------------------------
def _random_digit(rng: random.Random) -> int:
    return rng.randint(0, 9)


def _random_number(rng: random.Random, length: int, allow_leading_zero: bool = False) -> str:
    """Return a decimal string of exactly ``length`` digits."""
    if length <= 0:
        return ""
    if allow_leading_zero:
        return "".join(str(rng.randint(0, 9)) for _ in range(length))
    first = str(rng.randint(1, 9))
    rest = "".join(str(rng.randint(0, 9)) for _ in range(length - 1))
    return first + rest


def _encode_string(s: str) -> List[int]:
    return [CHAR_TO_IDX[ch] for ch in s]


def _decode_indices(indices: Sequence[int]) -> str:
    return "".join(IDX_TO_CHAR.get(int(i), "?") for i in indices)


def _pad_sequence(seq: List[int], length: int, pad_idx: int = PAD_IDX) -> List[int]:
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_idx] * (length - len(seq))


# ---------------------------------------------------------------------------
# Individual task generators
# ---------------------------------------------------------------------------
def generate_copy(length: int, rng: random.Random) -> Tuple[str, str]:
    """Copy task: input sequence is repeated as output."""
    seq = "".join(str(rng.randint(0, 9)) for _ in range(length))
    return seq, seq


def generate_reverse(length: int, rng: random.Random) -> Tuple[str, str]:
    """Reverse task: output is the reversed input sequence."""
    seq = "".join(str(rng.randint(0, 9)) for _ in range(length))
    return seq, seq[::-1]


def generate_walk(length: int, rng: random.Random) -> Tuple[str, str]:
    """Walk task.

    The input is a sequence of directions (N, S, E, W).  The output is the
    sequence of visited grid positions encoded as digits.  We keep the
    encoding simple: positions are represented by their integer coordinates
    modulo 10, concatenated as ``x_mod10 y_mod10`` for each step.
    """
    directions = [rng.choice(_WALK_SYMBOLS) for _ in range(length)]
    x, y = 0, 0
    out_parts: List[str] = []
    for d in directions:
        if d == "N":
            y += 1
        elif d == "S":
            y -= 1
        elif d == "E":
            x += 1
        else:  # W
            x -= 1
        out_parts.append(str(x % 10))
        out_parts.append(str(y % 10))
    return "".join(directions), "".join(out_parts)


def _int_to_decimal_string(value: int, min_length: int = 1) -> str:
    s = str(value)
    if len(s) < min_length:
        s = "0" * (min_length - len(s)) + s
    return s


def generate_addition(length: int, rng: random.Random) -> Tuple[str, str]:
    """Addition task.

    ``length`` is interpreted as the number of digits in each operand.
    Input format:  operand1 + operand2
    Output:        sum (decimal)
    """
    a = _random_number(rng, length)
    b = _random_number(rng, length)
    result = int(a) + int(b)
    # Output length may be length+1 due to carry.
    out = _int_to_decimal_string(result, min_length=length)
    return f"{a}+{b}", out


def generate_3number_addition(
    length: int, rng: random.Random, curriculum_easy: bool = False
) -> Tuple[str, str]:
    """Three-number addition task.

    When ``curriculum_easy`` is True, generate examples without carries and
    with many zeros, as described in the paper's curriculum for this task.
    """
    if curriculum_easy:
        # Generate digits that avoid carries: each column sum <= 9.
        digits = [[0, 0, 0] for _ in range(length)]
        for pos in range(length):
            remaining = rng.randint(0, 9)
            # Distribute remaining among the three numbers.
            for num_idx in range(3):
                if num_idx == 2:
                    digits[pos][num_idx] = remaining
                else:
                    max_val = remaining
                    val = rng.randint(0, max_val)
                    digits[pos][num_idx] = val
                    remaining -= val
        nums = []
        for num_idx in range(3):
            num_str = "".join(str(digits[pos][num_idx]) for pos in range(length))
            nums.append(num_str)
        total = sum(int(n) for n in nums)
        out = _int_to_decimal_string(total, min_length=length)
        return f"{nums[0]}+{nums[1]}+{nums[2]}", out

    nums = [_random_number(rng, length) for _ in range(3)]
    total = sum(int(n) for n in nums)
    out = _int_to_decimal_string(total, min_length=length)
    return f"{nums[0]}+{nums[1]}+{nums[2]}", out


def generate_single_digit_multiplication(
    length: int, rng: random.Random
) -> Tuple[str, str]:
    """Single digit multiplication task.

    ``length`` is interpreted as the number of digits in the multi-digit
    operand.  The single-digit multiplier is in [2, 9] to avoid trivial
    multiplication by 0 or 1.
    """
    a = _random_number(rng, length)
    b = str(rng.randint(2, 9))
    result = int(a) * int(b)
    out = _int_to_decimal_string(result, min_length=length)
    return f"{a}*{b}", out


TASK_GENERATORS = {
    "copy": generate_copy,
    "reverse": generate_reverse,
    "walk": generate_walk,
    "addition": generate_addition,
    "3_number_addition": generate_3number_addition,
    "single_digit_multiplication": generate_single_digit_multiplication,
}


# ---------------------------------------------------------------------------
# Dataset item container
# ---------------------------------------------------------------------------
@dataclass
class AlgorithmicExample:
    """A single synthetic example."""

    task: str
    input_str: str
    target_str: str
    input_ids: List[int]
    target_ids: List[int]
    complexity: int


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class SyntheticAlgorithmicTasksCopyReverseWalkAddition3NumberAdditionSingleDigitMultiplicationDataset(
    Dataset
):
    """Synthetic dataset for the six algorithmic tasks.

    Parameters
    ----------
    tasks:
        Sequence of task names to include.  Defaults to all six tasks.
    num_examples_per_task:
        Number of examples to generate per task.  The paper uses 3x10^7
        characters for training; this parameter controls example count.
    complexity:
        Base complexity (sequence length) used for generation.
    curriculum_easy_3add:
        If True, generate easy 3-number addition examples (no carries, many
        zeros).  This implements the curriculum described in the paper.
    max_length:
        Maximum encoded sequence length.  Sequences longer than this are
        truncated.  If None, no truncation is applied.
    seed:
        Random seed for reproducible generation.
    """

    def __init__(
        self,
        tasks: Optional[Sequence[str]] = None,
        num_examples_per_task: int = 1000,
        complexity: int = 6,
        curriculum_easy_3add: bool = False,
        max_length: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if tasks is None:
            tasks = list(TASK_GENERATORS.keys())
        self.tasks = list(tasks)
        self.num_examples_per_task = num_examples_per_task
        self.complexity = complexity
        self.curriculum_easy_3add = curriculum_easy_3add
        self.max_length = max_length
        self.seed = seed

        self.examples: List[AlgorithmicExample] = []
        self._generate_examples()

    def _generate_examples(self) -> None:
        rng = random.Random(self.seed)
        for task in self.tasks:
            generator = TASK_GENERATORS[task]
            for _ in range(self.num_examples_per_task):
                if task == "3_number_addition":
                    inp, tgt = generator(
                        self.complexity, rng, self.curriculum_easy_3add
                    )
                else:
                    inp, tgt = generator(self.complexity, rng)

                input_ids = _encode_string(inp)
                target_ids = _encode_string(tgt)

                if self.max_length is not None:
                    input_ids = _pad_sequence(input_ids, self.max_length)
                    target_ids = _pad_sequence(target_ids, self.max_length)

                self.examples.append(
                    AlgorithmicExample(
                        task=task,
                        input_str=inp,
                        target_str=tgt,
                        input_ids=input_ids,
                        target_ids=target_ids,
                        complexity=self.complexity,
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        input_tensor = torch.tensor(example.input_ids, dtype=torch.long)
        target_tensor = torch.tensor(example.target_ids, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "target_ids": target_tensor,
            "task": example.task,
            "complexity": example.complexity,
        }

    def set_complexity(self, complexity: int) -> None:
        """Regenerate examples with a new complexity (curriculum step)."""
        self.complexity = complexity
        self.examples = []
        self._generate_examples()

    def set_curriculum_easy_3add(self, easy: bool) -> None:
        """Toggle easy 3-number addition generation."""
        self.curriculum_easy_3add = easy
        self.examples = []
        self._generate_examples()


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch of variable-length sequences and stack them."""
    input_seqs = [item["input_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    input_lens = [len(s) for s in input_seqs]
    target_lens = [len(s) for s in target_seqs]
    max_input_len = max(input_lens) if input_lens else 0
    max_target_len = max(target_lens) if target_lens else 0

    padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(input_seqs, target_seqs)):
        padded_inputs[i, : len(inp)] = inp
        padded_targets[i, : len(tgt)] = tgt

    tasks = [item["task"] for item in batch]
    complexities = [item["complexity"] for item in batch]

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets,
        "input_lengths": torch.tensor(input_lens, dtype=torch.long),
        "target_lengths": torch.tensor(target_lens, dtype=torch.long),
        "task": tasks,
        "complexity": torch.tensor(complexities, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------
def _load_config(config: Optional[Any]) -> Dict[str, Any]:
    """Load config from a path, dict, or return empty dict if None."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        if yaml is None:
            raise ImportError("PyYAML is required to load config from a file.")
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise TypeError("config must be None, a dict, or a path string.")


def _safe_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    config: Optional[Any] = None,
    version: str = "default",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
    num_examples_per_task: Optional[int] = None,
    complexity: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders for the synthetic tasks.

    The paper uses only train/test splits.  If a validation loader is
    requested (for early stopping or model selection), it is carved out of
    the training set using an 80/20 split.  This is an engineering deviation
    from the paper and is documented as such.

    Parameters
    ----------
    config:
        Path to config.yaml, a config dict, or None.  When None, sensible
        defaults are used so the function can be called without arguments.
    version:
        Dataset version selector.  Only ``"default"`` is supported; the
        parameter is accepted for interface compatibility.
    batch_size:
        Override for training batch size.  Defaults to config value or 20.
    num_workers:
        Override for DataLoader workers.  Defaults to config value or 0.
    seed:
        Override for random seed.  Defaults to config value or 42.
    num_examples_per_task:
        Override for examples per task.  Defaults to 1000.
    complexity:
        Override for initial curriculum complexity.  Defaults to config
        ``split_strategy.curriculum.start_complexity`` or 6.
    max_length:
        Maximum encoded sequence length.  Defaults to None (no truncation).

    Returns
    -------
    Dict[str, DataLoader]
        Keys: ``"train"``, ``"val"``, ``"test"``.  ``"val"`` may be empty if
        no validation split is configured.
    """
    cfg = _load_config(config)

    # Resolve parameters with precedence: explicit argument > config > default
    if batch_size is None:
        batch_size = int(
            _safe_get(cfg, "training", "batch_size", "default", default=20)
        )
    if num_workers is None:
        num_workers = int(_safe_get(cfg, "system", "num_workers", default=0))
    if seed is None:
        seed = int(_safe_get(cfg, "system", "seed", default=42))
    if complexity is None:
        complexity = int(
            _safe_get(
                cfg,
                "split_strategy",
                "curriculum",
                "start_complexity",
                default=6,
            )
        )
    if num_examples_per_task is None:
        # The paper trains on 3x10^7 characters.  We expose example count
        # rather than character count; default keeps memory reasonable.
        num_examples_per_task = 1000

    # Curriculum easy mode for 3-number addition
    curriculum_enabled = bool(
        _safe_get(cfg, "split_strategy", "curriculum", "enabled", default=True)
    )
    curriculum_easy_3add = curriculum_enabled

    # Validation split ratio (engineering deviation from paper)
    val_split_ratio = float(
        _safe_get(cfg, "split_strategy", "val_split_ratio", default=0.2)
    )
    val_split_ratio = max(0.0, min(1.0, val_split_ratio))

    # Test configuration
    test_instances = int(
        _safe_get(cfg, "split_strategy", "curriculum", "test_instances", default=50)
    )
    test_lengths = _safe_get(
        cfg, "split_strategy", "curriculum", "test_lengths", default=[100, 1000]
    )
    if not isinstance(test_lengths, (list, tuple)):
        test_lengths = [100, 1000]

    # Build training dataset
    train_dataset = SyntheticAlgorithmicTasksCopyReverseWalkAddition3NumberAdditionSingleDigitMultiplicationDataset(
        tasks=None,
        num_examples_per_task=num_examples_per_task,
        complexity=complexity,
        curriculum_easy_3add=curriculum_easy_3add,
        max_length=max_length,
        seed=seed,
    )

    # Build test dataset(s).  We generate ``test_instances`` examples for each
    # configured test length.  For simplicity, we use the first test length
    # for the primary test loader and also expose a long-sequence test loader
    # when multiple lengths are configured.
    test_length_primary = int(test_lengths[0]) if test_lengths else 100
    test_dataset = SyntheticAlgorithmicTasksCopyReverseWalkAddition3NumberAdditionSingleDigitMultiplicationDataset(
        tasks=None,
        num_examples_per_task=max(1, test_instances // len(TASK_GENERATORS)),
        complexity=test_length_primary,
        curriculum_easy_3add=False,
        max_length=max_length,
        seed=seed + 1000,
    )

    # Split train into train/val if requested
    if val_split_ratio > 0.0:
        val_size = int(len(train_dataset) * val_split_ratio)
        train_size = len(train_dataset) - val_size
        if val_size > 0 and train_size > 0:
            generator = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size], generator=generator
            )
        else:
            train_subset = train_dataset
            val_subset = None
    else:
        train_subset = train_dataset
        val_subset = None

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    loaders: Dict[str, Optional[DataLoader]] = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    # Long-sequence test loader (length 1000) if configured
    if len(test_lengths) > 1:
        long_length = int(test_lengths[1])
        long_test_dataset = SyntheticAlgorithmicTasksCopyReverseWalkAddition3NumberAdditionSingleDigitMultiplicationDataset(
            tasks=None,
            num_examples_per_task=max(1, test_instances // len(TASK_GENERATORS)),
            complexity=long_length,
            curriculum_easy_3add=False,
            max_length=max_length,
            seed=seed + 2000,
        )
        loaders["test_long"] = DataLoader(
            long_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

    return loaders


# ---------------------------------------------------------------------------
# Convenience function for direct dataset creation
# ---------------------------------------------------------------------------
def create_dataset(
    config: Optional[Any] = None,
    version: str = "default",
    split: str = "train",
    **kwargs: Any,
) -> Dataset:
    """Create a single dataset split.

    This is a convenience wrapper around the dataset class for callers that
    only need a ``Dataset`` object rather than full dataloaders.
    """
    cfg = _load_config(config)
    seed = int(_safe_get(cfg, "system", "seed", default=42))
    complexity = int(
        _safe_get(
            cfg,
            "split_strategy",
            "curriculum",
            "start_complexity",
            default=6,
        )
    )
    curriculum_enabled = bool(
        _safe_get(cfg, "split_strategy", "curriculum", "enabled", default=True)
    )

    if split == "test":
        test_lengths = _safe_get(
            cfg,
            "split_strategy",
            "curriculum",
            "test_lengths",
            default=[100, 1000],
        )
        if isinstance(test_lengths, (list, tuple)) and test_lengths:
            complexity = int(test_lengths[0])
        seed = seed + 1000
        curriculum_enabled = False

    return SyntheticAlgorithmicTasksCopyReverseWalkAddition3NumberAdditionSingleDigitMultiplicationDataset(
        tasks=kwargs.get("tasks"),
        num_examples_per_task=kwargs.get("num_examples_per_task", 1000),
        complexity=kwargs.get("complexity", complexity),
        curriculum_easy_3add=kwargs.get("curriculum_easy_3add", curriculum_enabled),
        max_length=kwargs.get("max_length"),
        seed=kwargs.get("seed", seed),
    )