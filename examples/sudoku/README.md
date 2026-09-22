# Sudoku (Final) – Phase 1 & 2

Phase 1 implements **IO**, **CoT**, and **ToT**. Phase 2 adds **GoT-1 (Horizontal Bands)**, **GoT-2 (9 Blocks)**, and **GoT-3 (Full Board)** with evaluation metrics.

## Methods

### Phase 1
- **IO** – Single direct solve with examples (grid + 81-digit output).
- **CoT** – Same graph as IO; prompt asks for step-by-step reasoning and examples.
- **ToT** – Wider tree: 8 branches → score → keep best → 8 branches (improve) → score → keep best → ground truth.

### Phase 2 (GoT)
- **GoT-1 (Horizontal Bands)** – Split puzzle into 3 horizontal bands (rows 1–3, 4–6, 7–9). Solve each band with the LLM (4 branches per band), merge, then improve & refine (4 branches).
- **GoT-2 (9 Blocks)** – Split into 9 3×3 blocks (deterministic). Solve each block (3 branches per block), merge, then improve & refine (4 branches).
- **GoT-3 (Full Board)** – No split. Generate 5 candidates → score → keep best 3 → aggregate → score → keep best 1 → improve (4 branches) → score → keep best 1 → ground truth.

## Evaluation

After each run, call `evaluate_run(results_folder)` (or use `__main__`, which does it for the latest run). This produces:

- **evaluation.csv** – Per (method, sample_id): difficulty, success, num_errors, prompt_tokens, completion_tokens, total_tokens, cost_usd, time_sec.
- **summary.json** – Per method: success_rate, avg_errors, total_prompt_tokens, total_completion_tokens, total_cost_usd, total_time_sec, n_samples; and search_metrics (branching_factor, graph_depth, num_operations).
- **{sample_id}_metrics.json** – Per (method, sample): time_elapsed_sec, prompt_tokens, completion_tokens, cost_usd, difficulty.

### Metrics reported

| Category | Metrics |
|----------|---------|
| **Model & dataset** | LLM model (`config.json`), dataset difficulty (per sample and filter in `summary.json`) |
| **Search** | Branching factor, graph depth, num_operations (in `summary.json` → `search_metrics`) |
| **Performance** | Success rate, score/accuracy (avg_errors, success in `summary.json`) |
| **Efficiency** | time_sec, prompt_tokens, completion_tokens, cost_usd (per sample in evaluation.csv; totals in summary.json) |

## Run

From repo root:

```bash
python examples/sudoku_final/sudoku.py
```

Or in code with difficulty filter and evaluation:

```python
from examples.sudoku_final.sudoku import run, evaluate_run, io, cot, tot, got1, got2, got3

spent = run(
    data_ids=list(range(10)),
    methods=[io, cot, tot, got1, got2, got3],
    budget=100.0,
    lm_name="chatgpt4-mini",
    difficulty_levels=None,  # or ["easy", "medium", "hard"]
)
summary = evaluate_run("examples/sudoku_final/results/<your_run_folder>")
```

Results are written under `examples/sudoku_final/results/<lm>_io-cot-tot-got1-got2-got3_<timestamp>/`.

## Files

- **utils.py** – Board/band/block helpers, `extract_digits`, `split_board_horizontally`, `merge_bands`, `split_board_into_blocks`, `merge_blocks`, `band_errors`, `block_errors`, `num_errors`, `test_sudoku`, `classify_difficulty`.
- **sudoku.py** – Prompter (IO/CoT/ToT/GoT-1/2/3), Parser, io/cot/tot/got1/got2/got3, run(), evaluate_run(), _graph_metrics(), _load_data().
