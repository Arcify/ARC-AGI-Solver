"""Dataset utilities for ARC-AGI 2 benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json

import torch
from torch.utils.data import Dataset


class ARCDataset(Dataset):
    """Simple loader for the Kaggle ARC-AGI dataset.

    The actual dataset files are provided as JSON mappings in the ``dataset``
    directory.  Each split (``training``, ``evaluation`` or ``test``) is stored
    in a single JSON file containing a mapping from task IDs to dictionaries with
    ``train`` and ``test`` fields.  The ``test`` portion of the ``training`` and
    ``evaluation`` splits has its ground-truth outputs stored in a companion
    ``*_solutions.json`` file.
    """

    _SPLIT_FILES = {
        "training": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
        "evaluation": ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
        "test": ("arc-agi_test_challenges.json", None),
    }

    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path {self.root!s} does not exist")

        self.split = split
        self.tasks: List[dict] = []
        if split is not None:
            self.tasks = self._load_split(split)

    def _load_split(self, split: str) -> List[dict]:
        if split not in self._SPLIT_FILES:
            raise ValueError(f"Unknown split '{split}'")

        challenge_file, solution_file = self._SPLIT_FILES[split]
        challenge_path = self.root / challenge_file
        if not challenge_path.exists():
            raise FileNotFoundError(f"Challenge file {challenge_path!s} does not exist")

        with open(challenge_path, "r", encoding="utf-8") as f:
            challenges = json.load(f)

        solutions = {}
        if solution_file is not None:
            solution_path = self.root / solution_file
            if not solution_path.exists():
                raise FileNotFoundError(f"Solution file {solution_path!s} does not exist")
            with open(solution_path, "r", encoding="utf-8") as f:
                solutions = json.load(f)

        tasks = []
        for task_id, task in challenges.items():
            train_pairs = task.get("train", [])
            test_pairs = task.get("test", [])

            # attach solutions if available
            if solution_file is not None:
                sols = solutions.get(task_id, [])
                for pair, sol in zip(test_pairs, sols):
                    pair["output"] = sol

            tasks.append({"train": train_pairs, "test": test_pairs})

        return tasks

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:  # type: ignore[override]
        task = self.tasks[idx]

        def to_tensor(pair: dict) -> Tuple[torch.Tensor, torch.Tensor]:
            inp = torch.tensor(pair["input"], dtype=torch.long)
            out_grid = pair.get("output")
            if out_grid is None:
                out_grid = [[-1 for _ in row] for row in pair["input"]]
            out = torch.tensor(out_grid, dtype=torch.long)
            return inp, out

        train_pairs = [to_tensor(p) for p in task["train"]]
        test_pairs = [to_tensor(p) for p in task["test"]]
        return {"train": train_pairs, "test": test_pairs}

    def load(self) -> Tuple["ARCDataset", "ARCDataset"]:
        """Return train and test splits as :class:`ARCDataset` instances."""
        train_ds = ARCDataset(self.root, "training")
        eval_ds = ARCDataset(self.root, "evaluation")
        return train_ds, eval_ds


def collate_grids(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of grid pairs into tensors for batching."""

    if not batch:
        raise ValueError("Empty batch provided to collate_grids")

    max_h = max(t[0].shape[0] for t in batch)
    max_w = max(t[0].shape[1] for t in batch)

    batch_inputs = torch.full((len(batch), 1, max_h, max_w), -1, dtype=torch.long)
    batch_outputs = torch.full((len(batch), max_h, max_w), -1, dtype=torch.long)

    for i, (inp, out) in enumerate(batch):
        h, w = inp.shape
        batch_inputs[i, 0, :h, :w] = inp
        oh, ow = out.shape
        batch_outputs[i, :oh, :ow] = out

    return batch_inputs, batch_outputs
