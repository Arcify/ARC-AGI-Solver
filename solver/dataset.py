"""Dataset utilities for ARC-AGI 2 benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json

import torch
from torch.utils.data import Dataset


class ARCDataset(Dataset):
    """Simple loader for ARC dataset.

    This is a lightweight loader that expects tasks to be stored as JSON files in
    the following structure::

        dataset_root/
            train/
                <task_id>.json
            test/
                <task_id>.json

    Each JSON file should contain ``train`` and ``test`` fields with lists of
    input/output grid pairs.
    """

    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path {self.root!s} does not exist")

        if not (self.root / "train").exists() or not (self.root / "test").exists():
            raise FileNotFoundError(
                "Dataset must contain 'train/' and 'test/' directories"
            )

        self.split = split
        self.tasks: List[dict] = []
        if split is not None:
            self.tasks = self._load_split(split)

    def _load_split(self, split: str) -> List[dict]:
        split_path = self.root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory {split_path!s} does not exist")
        tasks = []
        for json_file in split_path.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                tasks.append(json.load(f))
        return tasks

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:  # type: ignore[override]
        task = self.tasks[idx]

        def to_tensor(pair: dict) -> Tuple[torch.Tensor, torch.Tensor]:
            inp = torch.tensor(pair["input"], dtype=torch.long)
            out = torch.tensor(pair["output"], dtype=torch.long)
            return inp, out

        train_pairs = [to_tensor(p) for p in task["train"]]
        test_pairs = [to_tensor(p) for p in task["test"]]
        return {"train": train_pairs, "test": test_pairs}

    def load(self) -> Tuple["ARCDataset", "ARCDataset"]:
        """Return train and test splits as :class:`ARCDataset` instances."""
        train_ds = ARCDataset(self.root, "train")
        test_ds = ARCDataset(self.root, "test")
        return train_ds, test_ds
