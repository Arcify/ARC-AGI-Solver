from pathlib import Path
import json
import unittest
import tempfile

import torch

from solver.dataset import ARCDataset


class TestDataset(unittest.TestCase):
    def test_missing_dataset(self) -> None:
        tmp_path = Path("nonexistent_dir")
        with self.assertRaises(FileNotFoundError):
            ARCDataset(tmp_path)

    def test_missing_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                ARCDataset(root, "training")

    def test_loading_and_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            challenges = {
                "0000": {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]]}],
                }
            }
            solutions = {"0000": [[[3]]]} 

            with open(root / "arc-agi_training_challenges.json", "w", encoding="utf-8") as f:
                json.dump(challenges, f)
            with open(root / "arc-agi_training_solutions.json", "w", encoding="utf-8") as f:
                json.dump(solutions, f)

            ds = ARCDataset(root, "training")
            self.assertEqual(len(ds), 1)

            item = ds[0]
            self.assertIsInstance(item["train"][0][0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
