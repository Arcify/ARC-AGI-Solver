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

    def test_missing_split_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            with self.assertRaises(FileNotFoundError):
                ARCDataset(root)

    def test_loading_and_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "test").mkdir()

            data = {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]], "output": [[3]]}],
            }
            with open(root / "train" / "task.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            with open(root / "test" / "task.json", "w", encoding="utf-8") as f:
                json.dump(data, f)

            train_ds, test_ds = ARCDataset(root).load()
            self.assertEqual(len(train_ds), 1)
            self.assertEqual(len(test_ds), 1)

            item = train_ds[0]
            self.assertIsInstance(item["train"][0][0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
