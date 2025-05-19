import unittest
import torch

from solver.dsl import (
    color_replace_op,
    rotate_cw_op,
    flip_horizontal_op,
    translate_op,
    Program,
)


class TestDSL(unittest.TestCase):
    def test_color_replace(self) -> None:
        grid = torch.tensor([[0, 1], [1, 0]])
        op = color_replace_op(1, 2)
        out = op(grid)
        expected = torch.tensor([[0, 2], [2, 0]])
        self.assertEqual(out, expected)

    def test_rotate(self) -> None:
        grid = torch.tensor([[1, 2], [3, 4]])
        op = rotate_cw_op()
        out = op(grid)
        expected = torch.tensor([[3, 1], [4, 2]])
        self.assertEqual(out, expected)

    def test_flip_horizontal(self) -> None:
        grid = torch.tensor([[1, 2], [3, 4]])
        op = flip_horizontal_op()
        out = op(grid)
        expected = torch.tensor([[2, 1], [4, 3]])
        self.assertEqual(out, expected)

    def test_translate(self) -> None:
        grid = torch.tensor([[1, 2], [3, 4]])
        op = translate_op(1, 0)
        out = op(grid)
        expected = torch.tensor([[-1, -1], [1, 2]])
        self.assertEqual(out, expected)

    def test_program(self) -> None:
        grid = torch.tensor([[0]])
        program = Program([color_replace_op(0, 1), rotate_cw_op()])
        out = program.apply(grid)
        expected = torch.tensor([[1]])
        self.assertEqual(out, expected)


if __name__ == "__main__":
    unittest.main()
