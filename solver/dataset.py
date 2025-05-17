"""Dataset utilities for ARC-AGI 2 benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json


class ARCDataset:
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

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path {self.root!s} does not exist")

    def _load_split(self, split: str) -> List[dict]:
        split_path = self.root / split
        tasks = []
        for json_file in split_path.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                tasks.append(json.load(f))
        return tasks

    def load(self) -> Tuple[List[dict], List[dict]]:
        """Return train and test splits as lists of tasks."""
        train_tasks = self._load_split("train")
        test_tasks = self._load_split("test")
        return train_tasks, test_tasks
