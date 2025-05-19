from __future__ import annotations

"""Simple neural-guided program synthesis utilities."""

from typing import Iterable, List, Optional, Any

from .dsl import (
    Program,
    Operation,
    color_replace_op,
    rotate_cw_op,
    flip_horizontal_op,
    translate_op,
)


class NeuralGuidedSynthesizer:
    """Enumerative program synthesizer guided by a neural model."""

    def __init__(self, model: Optional[Any] = None, max_depth: int = 2) -> None:
        self.model = model
        self.max_depth = max_depth
        self.operations = self._build_operations()

    @staticmethod
    def _build_operations() -> List[Operation]:
        ops: List[Operation] = []
        for a in range(10):
            for b in range(10):
                if a != b:
                    ops.append(color_replace_op(a, b))
        ops.append(rotate_cw_op())
        ops.append(flip_horizontal_op())
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ops.append(translate_op(dx, dy))
        return ops

    def _score_program(self, program: Program, inputs: Iterable[List[List[int]]]) -> float:
        """Return a heuristic score for ``program``.``"""

        if self.model is None:
            return -len(program.ops)

        # TODO: use neural model for scoring
        return -len(program.ops)

    def synthesize(self, train_pairs: List[tuple[List[List[int]], List[List[int]]]]) -> Optional[Program]:
        """Search for a program that matches all ``train_pairs``."""

        programs = [Program([])]
        for _ in range(self.max_depth):
            next_programs: List[Program] = []
            for program in programs:
                if all(program.apply(inp) == out for inp, out in train_pairs):
                    return program
                for op in self.operations:
                    next_programs.append(program.extend(op))
            # rank by heuristic score
            next_programs.sort(
                key=lambda p: self._score_program(p, [inp for inp, _ in train_pairs]),
                reverse=True,
            )
            programs = next_programs[:1000]

        for program in programs:
            if all(program.apply(inp) == out for inp, out in train_pairs):
                return program
        return None
