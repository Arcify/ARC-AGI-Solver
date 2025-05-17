from pathlib import Path
import unittest

from solver.dataset import ARCDataset


class TestDataset(unittest.TestCase):
    def test_missing_dataset(self) -> None:
        tmp_path = Path("nonexistent_dir")
        with self.assertRaises(FileNotFoundError):
            ARCDataset(tmp_path)


if __name__ == "__main__":
    unittest.main()
