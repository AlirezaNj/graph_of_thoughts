# Equation Solving 2 — System of Two Linear Equations (x, y)

Solve **systems of two linear equations in two variables (x and y)** using IO, CoT, ToT, and GoT. Uses the same framework as `equation_solving` but for the 2×2 case.

## Problem

- **Input:** Two linear equations, e.g. `7x + 2y = 56` and `18x + 14y = 206`
- **Output:** The numeric values of x and y (e.g. `x = 6, y = 7`)
- **Data:** `equations.csv` with columns `equation1`, `equation2`, `x`, `y` (ground truth)

## Methods

| Method | Description |
|--------|-------------|
| **IO** | Single direct prompt: solve system, output "x = <num>, y = <num>". |
| **CoT** | Step-by-step (substitution/elimination), same output format. |
| **ToT** | Multiple attempts → score → keep best → improve → score → keep best → ground truth. |
| **GoT** | Split into Equation 1 and Equation 2; each branch rewrites its equation (e.g. y in terms of x); aggregate solves for x and y. |

## Run

From repo root:

```bash
python -m examples.equation_solving2.equation_solving
```

Or in code:

```python
from examples.equation_solving2.equation_solving import run, io, cot, tot, got

spent = run(
    data_ids=[0, 1, 2],
    methods=[io, cot, tot, got],
    budget=20.0,
    lm_name="chatgpt4-mini",
)
```

Results are written under `examples/equation_solving2/results/<lm>_io-cot-tot-got_<timestamp>/`. Sample id is the row index in `equations.csv` (0-based).
