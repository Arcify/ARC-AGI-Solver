import unittest
import torch

from solver.synthesizer import NeuralGuidedSynthesizer


class TestSynthesizer(unittest.TestCase):
    def test_simple_color_replace(self) -> None:
        synth = NeuralGuidedSynthesizer(max_depth=1)
        inp = torch.tensor([[0, 0]])
        out = torch.tensor([[1, 1]])
        prog = synth.synthesize([(inp, out)])
        self.assertIsNotNone(prog)
        if prog is not None:
            self.assertEqual(prog.apply(inp), out)


if __name__ == "__main__":
    unittest.main()
