# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""
Equation solving (system of 2 linear equations in x and y): IO, CoT, ToT, GoT.
Uses examples/equation_solving2/equations.csv (equation1, equation2, x, y).
"""

import os
import logging
import datetime
import json
import csv
import random
import time
from typing import Dict, List, Callable, Union

from graph_of_thoughts import controller, language_models, operations, prompter, parser

try:
    from . import utils
except ImportError:
    import utils


class EquationPrompter(prompter.Prompter):
    """
    Prompts for solving a system of two linear equations in x and y.
    """

    io_prompt = """<Instruction> Solve the following system of two linear equations for x and y. Your last line must be exactly: Output: x = <num>, y = <num> with the two numbers that satisfy both equations. No other text after that. </Instruction>

<Examples>
Equation 1: 7x + 2y = 56
Equation 2: 18x + 14y = 206
Output: x = 6, y = 7

Equation 1: 2x + 4y = 52
Equation 2: 2x + 2y = 32
Output: x = 6, y = 10
</Examples>

Equation 1: {equation1}
Equation 2: {equation2}

Output: x = <num>, y = <num>"""

    cot_prompt = """<Instruction> Solve the following system of two linear equations in x and y step by step (substitution or elimination). Your very last line must be exactly: Output: x = <num>, y = <num> with the final numeric solution. No other format. </Instruction>

<Approach>
1. From one equation, express y in terms of x (or x in terms of y).
2. Substitute into the other equation and solve for one variable.
3. Back-substitute to find the other variable.
4. End with exactly: Output: x = <num>, y = <num>
</Approach>

<Examples>
Equation 1: 7x + 2y = 56
Equation 2: 18x + 14y = 206
Step 1: From eq1, 2y = 56 - 7x, so y = (56 - 7x)/2. Substitute into eq2: 18x + 14(56-7x)/2 = 206 → 18x + 392 - 49x = 206 → -31x = -186 → x = 6.
Step 2: y = (56 - 42)/2 = 7.
Output: x = 6, y = 7

Equation 1: 2x + 4y = 52
Equation 2: 2x + 2y = 32
Step 1: From eq2, 2y = 32 - 2x, y = 16 - x. Substitute into eq1: 2x + 4(16-x) = 52 → 2x + 64 - 4x = 52 → -2x = -12 → x = 6. Then y = 10.
Output: x = 6, y = 10
</Examples>

Equation 1: {equation1}
Equation 2: {equation2}

Solve step by step. Your last line must be exactly:
Output: x = <num>, y = <num>"""

    tot_initial_prompt = """<Instruction> Solve the following system of two linear equations for x and y. Your last line must be exactly: Output: x = <num>, y = <num> with the two numbers that satisfy both equations. No other format. </Instruction>

<Examples>
Equation 1: 7x + 2y = 56
Equation 2: 18x + 14y = 206
Output: x = 6, y = 7

Equation 1: 2x + 4y = 52
Equation 2: 2x + 2y = 32
Output: x = 6, y = 10
</Examples>

Equation 1: {equation1}
Equation 2: {equation2}

Your final line only:
Output: x = <num>, y = <num>"""

    tot_improve_prompt = """<Instruction> The following system and an incorrect attempted solution are given. Find the correct values of x and y that satisfy both equations. Your last line must be exactly: Output: x = <num>, y = <num> </Instruction>

<Approach>
Use substitution or elimination: from one equation express y in terms of x (or x in terms of y), substitute into the other equation, solve for one variable, then back-substitute to get the other.
</Approach>

<Example>
Equation 1: 7x + 2y = 56
Equation 2: 18x + 14y = 206
Incorrect attempt: x = 10, y = 0
Correct: From eq1, y = (56 - 7x)/2. Substitute into eq2: 18x + 14(56-7x)/2 = 206 → 18x + 392 - 49x = 206 → -31x = -186 → x = 6. Then y = (56-42)/2 = 7.
Output: x = 6, y = 7
</Example>

Equation 1: {equation1}
Equation 2: {equation2}

Incorrect attempt: {current}

Correct solution. End with exactly:
Output: x = <num>, y = <num>"""

    got_split_prompt = """<Instruction> You are given a system of two linear equations. Output a valid JSON with exactly two keys "Equation 1" and "Equation 2", where the values are the full equation strings. No other text. </Instruction>

<Example>
Equation 1: 7x + 2y = 56
Equation 2: 18x + 14y = 206
Output:
{{"Equation 1": "7x + 2y = 56", "Equation 2": "18x + 14y = 206"}}
</Example>

Equation 1: {equation1}
Equation 2: {equation2}

Output:"""

    # GoT phase 1: rewrite one equation (e.g. solve for y in terms of x)
    got_rewrite_prompt = """<Instruction> From the following linear equation in x and y, express y in terms of x. Output only one line in the form y = <expression in x>, e.g. "y = (56 - 7x) / 2". Do not substitute any numbers for x. </Instruction>

<Examples>
Equation: 7x + 2y = 56
y in terms of x: y = (56 - 7x) / 2

Equation: 18x + 14y = 206
y in terms of x: y = (206 - 18x) / 14

Equation: 2x + 4y = 52
y in terms of x: y = (52 - 2x) / 4
</Examples>

Equation: {step_input}

y in terms of x (one line only):"""

    aggregation_prompt_template = """<Instruction> You are given two relations from a system of linear equations. Each relation gives y in terms of x (or is an equation in x and y). Find the numeric values of x and y that satisfy both.

<How to solve>
1. You have Relation 1: (y = ... in x) and Relation 2: (y = ... in x).
2. Set the two right-hand sides (RHS) equal to each other, because both equal y. So: RHS_of_eq1 = RHS_of_eq2.
3. Solve that single equation for x.
4. Substitute the value of x into either relation to find y.
5. Your last line must be exactly: Output: x = <num>, y = <num>
</How to solve>

<Example>
Relation from Equation 1: y = (56 - 7x) / 2
Relation from Equation 2: y = (206 - 18x) / 14
Step 1: Set RHS equal: (56 - 7x) / 2 = (206 - 18x) / 14
Step 2: Multiply by 14: 7(56 - 7x) = 206 - 18x → 392 - 49x = 206 - 18x → 186 = 31x → x = 6
Step 3: y = (56 - 42) / 2 = 7
Output: x = 6, y = 7
</Example>

Relation from Equation 1: {eq1}
Relation from Equation 2: {eq2}

Solve by setting both RHS equal, find x, then find y. End with:
Output: x = <num>, y = <num>"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        assert len(state_dicts) == 2, "Expected two states (Equation 1 and Equation 2)."
        sorted_states = sorted(
            state_dicts,
            key=lambda s: (0 if s.get("part") == "Equation 1" else 1),
        )
        eq1 = sorted_states[0].get("current", "")
        eq2 = sorted_states[1].get("current", "")
        return self.aggregation_prompt_template.format(eq1=eq1, eq2=eq2)

    def generate_prompt(
        self,
        num_branches: int,
        equation1: str = "",
        equation2: str = "",
        current: str = "",
        method: str = "",
        phase: int = 0,
        part: str = "",
        step_input: str = "",
        **kwargs,
    ) -> str:
        assert num_branches == 1, "Branching should be done via multiple requests."
        if method == "io":
            return self.io_prompt.format(equation1=equation1, equation2=equation2)
        elif method == "cot":
            return self.cot_prompt.format(equation1=equation1, equation2=equation2)
        elif method == "tot":
            if not current or current.strip() == "":
                return self.tot_initial_prompt.format(
                    equation1=equation1, equation2=equation2
                )
            return self.tot_improve_prompt.format(
                equation1=equation1, equation2=equation2, current=current
            )
        elif method == "got":
            if phase == 0:
                return self.got_split_prompt.format(
                    equation1=equation1, equation2=equation2
                )
            return self.got_rewrite_prompt.format(
                step_input=step_input or equation1 or equation2
            )
        return self.io_prompt.format(equation1=equation1, equation2=equation2)

    def improve_prompt(self, **kwargs) -> str:
        return self.tot_improve_prompt.format(
            equation1=kwargs.get("equation1", ""),
            equation2=kwargs.get("equation2", ""),
            current=kwargs.get("current", ""),
        )

    def validation_prompt(self, **kwargs) -> str:
        return ""

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        return ""


class EquationParser(parser.Parser):
    """Parse (x, y) from responses; for GoT parse JSON split and rewrite outputs."""

    def __init__(self) -> None:
        self.cache = {}

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        assert len(states) == 2
        new_states = []
        for text in texts:
            x_val, y_val = utils.parse_two_numbers(text or "")
            if x_val is not None and y_val is not None:
                value_str = f"{x_val},{y_val}"
            else:
                value_str = ""
            base = {**states[0], **states[1]}
            base.pop("part", None)
            base.pop("step_input", None)
            new_states.append({**base, "current": value_str, "phase": 2})
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        if not texts:
            return {}
        x_val, y_val = utils.parse_two_numbers(texts[0])
        if x_val is not None and y_val is not None:
            return {"current": f"{x_val},{y_val}"}
        return {}

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        method = state.get("method", "")
        phase = state.get("phase", 0)

        if method == "got" and phase == 0:
            new_states = []
            for text in texts:
                try:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        data = json.loads(text[start:end])
                        for key in ["Equation 1", "Equation 2"]:
                            if key in data:
                                val = data[key]
                                if isinstance(val, list):
                                    val = str(val)
                                new_states.append({
                                    "part": key,
                                    "step_input": str(val).strip(),
                                    "current": "",
                                    "phase": 1,
                                })
                    if not new_states:
                        logging.warning(
                            f"Could not parse GoT split JSON: {text[:200]}"
                        )
                except Exception as e:
                    logging.warning(f"GoT split parse error: {e}. Text: {text[:200]}")
            if not new_states:
                return [{"current": "", "phase": 0}]
            return new_states

        # IO, CoT, ToT: extract x, y. GoT phase 1: keep rewrite as expression.
        new_states = []
        for text in texts:
            text_stripped = (text or "").strip()
            if method == "got" and phase == 1:
                step_input = state.get("step_input", "")
                # Keep first line that looks like y = ... or an equation with x and y
                expr = text_stripped
                for line in text_stripped.split("\n"):
                    line = line.strip()
                    if ("y" in line or "x" in line) and (
                        "=" in line or "*" in line or "+" in line or "-" in line
                    ):
                        expr = line
                        break
                new_states.append({"current": expr, "phase": 2})
            else:
                x_val, y_val = utils.parse_two_numbers(text_stripped)
                if x_val is not None and y_val is not None:
                    value_str = f"{x_val},{y_val}"
                else:
                    value_str = ""
                new_states.append({"current": value_str})
        if not new_states:
            new_states.append({"current": ""})
        return new_states

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        return True

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        return [0.0] * len(states)


def io() -> operations.GraphOfOperations:
    gop = operations.GraphOfOperations()
    gop.append_operation(operations.Generate(1, 1))
    gop.append_operation(operations.Score(1, False, utils.num_errors))
    gop.append_operation(operations.GroundTruth(utils.test_equation))
    return gop


def cot() -> operations.GraphOfOperations:
    gop = operations.GraphOfOperations()
    gop.append_operation(operations.Generate(1, 1))
    gop.append_operation(operations.Score(1, False, utils.num_errors))
    gop.append_operation(operations.GroundTruth(utils.test_equation))
    return gop


def tot() -> operations.GraphOfOperations:
    gop = operations.GraphOfOperations()
    gop.append_operation(operations.Generate(1, 5))
    gop.append_operation(operations.Score(1, False, utils.num_errors))
    gop.append_operation(operations.KeepBestN(1, False))
    gop.append_operation(operations.Improve())
    gop.append_operation(operations.Generate(1, 3))
    gop.append_operation(operations.Score(1, False, utils.num_errors))
    gop.append_operation(operations.KeepBestN(1, False))
    gop.append_operation(operations.GroundTruth(utils.test_equation))
    return gop


def got() -> operations.GraphOfOperations:
    """GoT: split into Equation 1 and Equation 2, rewrite each (y in terms of x), then aggregate."""
    gop = operations.GraphOfOperations()

    split = operations.Generate(1, 1)
    gop.append_operation(split)

    for part_name in ["Equation 1", "Equation 2"]:
        sel = operations.Selector(
            lambda thoughts, p=part_name: [
                t for t in thoughts if t.state.get("part") == p
            ]
        )
        sel.add_predecessor(split)
        gop.add_operation(sel)
        rewrite = operations.Generate(1, 3)
        rewrite.add_predecessor(sel)
        gop.add_operation(rewrite)
        score_part = operations.Score(1, False, utils.num_errors_part)
        score_part.add_predecessor(rewrite)
        gop.add_operation(score_part)
        keep = operations.KeepBestN(1, False)
        keep.add_predecessor(score_part)
        gop.add_operation(keep)

    agg = operations.Aggregate(4)
    gop.append_operation(agg)
    gop.append_operation(operations.Score(1, False, utils.num_errors))
    gop.append_operation(operations.KeepBestN(1, False))
    gop.append_operation(operations.GroundTruth(utils.test_equation))
    return gop


def _get_lm(lm_name: str, config_path: str, cache: bool = True):
    """Instantiate the appropriate LM (ChatGPT, Gemini, or Llama2HF) from config key."""
    gemini_keys = {"gemini", "gemini-lite", "gemini-pro"}
    llama_keys = {"llama13b-hf", "llama4-17b", "llama70b-hf"}
    if lm_name in llama_keys:
        return language_models.Llama2HF(config_path, model_name=lm_name, cache=cache)
    if lm_name in gemini_keys:
        return language_models.Gemini(config_path, model_name=lm_name, cache=cache)
    return language_models.ChatGPT(config_path, model_name=lm_name, cache=cache)


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Run each method on selected samples from equation_solving2/equations.csv.
    CSV columns: equation1, equation2, x, y (no id column; row index is used as id).
    """
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "equations.csv")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            data.append({
                "id": idx,
                "equation1": row["equation1"].strip(),
                "equation2": row["equation2"].strip(),
                "x": row["x"].strip(),
                "y": row["y"].strip(),
            })
    for row in data:
        row["solution"] = f"{row['x']},{row['y']}"

    if not data:
        logging.warning("No data in equations.csv")
        return 0.0

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected = [data[i] for i in data_ids if i < len(data)]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra = f"{lm_name}_{'-'.join([m.__name__ for m in methods])}"
    folder_name = f"{extra}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data_ids": data_ids,
        "methods": [m.__name__ for m in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__), exist_ok=True)

    config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )

    for sample in selected:
        sample_id = sample["id"]
        equation1 = sample["equation1"]
        equation2 = sample["equation2"]
        solution = sample["solution"]
        logging.info(
            f"Running sample {sample_id}: {equation1} | {equation2} -> {solution}"
        )
        if budget <= 0.0:
            break
        for method in methods:
            if budget <= 0.0:
                break
            logging.info(f"Method {method.__name__}, budget left: {budget}")
            lm = _get_lm(lm_name, config_path, cache=True)
            operations_graph = method()
            initial_state = {
                "equation1": equation1,
                "equation2": equation2,
                "solution": solution,
                "result": solution,
                "current": "",
                "method": method.__name__,
            }
            if method.__name__ == "got":
                initial_state["phase"] = 0
            executor = controller.Controller(
                lm,
                operations_graph,
                EquationPrompter(),
                EquationParser(),
                initial_state,
            )
            response_time_sec = None
            try:
                t0 = time.perf_counter()
                executor.run()
                response_time_sec = time.perf_counter() - t0
            except Exception as e:
                logging.error(f"Exception: {e}")
            out_path = os.path.join(
                results_folder, method.__name__, f"{sample_id}.json"
            )
            executor.output_graph(out_path)
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    output = json.load(f)
                if output and isinstance(output[-1], dict):
                    output[-1]["difficulty"] = "easy"
                    if response_time_sec is not None:
                        output[-1]["response_time"] = round(response_time_sec, 2)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to add response_time/difficulty to {out_path}: {e}")
            budget -= getattr(lm, "cost", 0.0)

    return orig_budget - budget


# Model -> number of random samples for run_per_model_samples (equation_solving)
EQUATION_SOLVING_MODEL_SAMPLES = {
    "chatgpt4": 5,       # ChatGPT 4.1
    "chatgpt4-mini": 5,  # ChatGPT 4.1 mini
    "chatgpt4-nano": 5,  # ChatGPT 4.1 nano
    "gemini": 5,         # Gemini 2.5 flash
    "gemini-lite": 5,    # Gemini 2.5 flash lite
    "llama4-17b": 5,     # Llama 4 17B
}


def run_per_model_samples(
    budget: float,
    methods: List[Callable[[], operations.GraphOfOperations]],
    seed: int = None,
) -> float:
    """
    Run equation_solving with a fixed number of random samples per model:
    - chatgpt4 (4.1): 5 random samples
    - chatgpt4-nano (4.1 nano): 5 random samples
    - gemini (2.5 flash): 5 random samples
    - gemini-lite (2.5 flash lite): 5 random samples
    - llama4-17b: 5 random samples

    :param budget: Total budget (dollars) shared across all model runs.
    :param methods: List of method functions (e.g. [io, cot, tot, got]).
    :param seed: Optional RNG seed for reproducible random sample indices.
    :return: Total spent budget.
    """
    data_path = os.path.join(os.path.dirname(__file__), "equations.csv")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            data.append({
                "id": idx,
                "equation1": row["equation1"].strip(),
                "equation2": row["equation2"].strip(),
                "x": row["x"].strip(),
                "y": row["y"].strip(),
            })
    n_total = len(data)
    if n_total == 0:
        logging.warning("No data in equations.csv")
        return 0.0

    if seed is not None:
        random.seed(seed)

    total_spent = 0.0
    for lm_name, n_samples in EQUATION_SOLVING_MODEL_SAMPLES.items():
        if budget <= 0.0:
            logging.warning("Budget depleted, skipping remaining models.")
            break
        n = min(n_samples, n_total)
        data_ids = random.sample(range(n_total), n)
        logging.info(f"Running {lm_name} with {n} random samples: {data_ids}")
        try:
            spent = run(data_ids, methods, budget, lm_name)
            total_spent += spent
            budget -= spent
        except OSError as e:
            if "gated" in str(e).lower() or "401" in str(e) or "unauthorized" in str(e).lower():
                logging.warning("Skipping %s: gated/restricted model. %s", lm_name, e)
            else:
                raise
        except Exception as e:
            if "GatedRepo" in type(e).__name__ or "401" in str(e):
                logging.warning("Skipping %s: %s", lm_name, e)
            else:
                raise

    return total_spent


if __name__ == "__main__":
    budget = 500.0
    approaches = [io, cot, tot, got]
    # Run with 5 random samples per model (chatgpt4, chatgpt4-nano, gemini, gemini-lite, llama4-17b)
    spent = run_per_model_samples(budget, approaches, seed=42)
    logging.info(f"Spent {spent} out of {budget} budget.")
