# ARC-AGI Solver

This repository contains tooling and experiments aimed at solving the
[ARC-AGI 2](https://github.com/fchollet/ARC) benchmark.

## Repository layout

- `solver/` – Python package containing the core modules used for dataset
  handling, model definition, training and evaluation.
- `tests/` – Basic tests to ensure the project loads correctly.
- `requirements.txt` – Python dependencies.

## Getting started

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests to verify your environment:

```bash
pytest
```

The `solver.train` module contains a `train` function that expects a dataset
layout as described in `solver/dataset.py`. The implementation is currently a
placeholder and should be extended with proper ARC grid handling.
