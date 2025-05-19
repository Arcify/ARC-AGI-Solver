from __future__ import annotations

"""Domain-specific language for simple grid transformations."""

from dataclasses import dataclass
from typing import Callable, List, Sequence

Grid = Sequence[Sequence[int]]


@dataclass
class Operation:
    """Encapsulates a grid transformation."""

    name: str
    func: Callable[[Grid], Grid]

    def __call__(self, grid: Grid) -> Grid:
        return self.func(grid)


@dataclass
class Program:
    """Sequence of operations that can be applied to a grid."""

    ops: List[Operation]

    def apply(self, grid: Grid) -> Grid:
        out = grid
        for op in self.ops:
            out = op(out)
        return out

    def extend(self, op: Operation) -> "Program":
        return Program(self.ops + [op])

    def __repr__(self) -> str:  # pragma: no cover - debug representation
        names = [op.name for op in self.ops]
        return f"Program({', '.join(names)})"


# Basic operations -----------------------------------------------------------

def color_replace_op(src: int, dst: int) -> Operation:
    def op(grid: Grid) -> Grid:
        return [[dst if c == src else c for c in row] for row in grid]

    return Operation(f"replace_{src}_to_{dst}", op)


def rotate_cw_op() -> Operation:
    def op(grid: Grid) -> Grid:
        if not grid:
            return []
        h = len(grid)
        w = len(grid[0])
        return [[grid[h - j - 1][i] for j in range(h)] for i in range(w)]

    return Operation("rotate_cw", op)


def flip_horizontal_op() -> Operation:
    def op(grid: Grid) -> Grid:
        return [list(reversed(row)) for row in grid]

    return Operation("flip_horizontal", op)


def translate_op(dx: int, dy: int) -> Operation:
    def op(grid: Grid) -> Grid:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        out = [[-1 for _ in range(w)] for _ in range(h)]
        for i in range(h):
            for j in range(w):
                ni = i + dx
                nj = j + dy
                if 0 <= ni < h and 0 <= nj < w:
                    out[ni][nj] = grid[i][j]
        return out

    return Operation(f"translate_{dx}_{dy}", op)

