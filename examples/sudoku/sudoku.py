import os
import sys

# Prefer the graph_of_thoughts package from this repo (graph-of-thoughts-new), not another install.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import re
import random
import time
import logging
import datetime
import json
import csv
from typing import Dict, List, Callable, Union, Optional, Any
from graph_of_thoughts import controller, language_models, operations, prompter, parser

try:
    from . import utils
except ImportError:
    import utils


class SudokuPrompter(prompter.Prompter):
    """
    SudokuPrompter provides the generation of prompts specific to the Sudoku
    example for the language models. Follows the same structure as sorting and
    set_intersection: Instruction, Examples, then Input.

    Puzzles are shown as 81-digit strings.
    The input puzzle in each prompt is sent as a single line of 81 digits (same format as the dataset; 0 for empty).
    Example grids in the prompts match the first two rows of sudoku.csv and are valid.

    Inherits from the Prompter class and implements its abstract methods.
    """

    solve_prompt_io = """<Instruction> Solve the following 9x9 Sudoku puzzle. The input is 81 digits, row by row, 0 for empty. Fill every row, column, and 3x3 box with digits 1-9 without repetition.
Each row must contain digits 1–9 exactly once. Each column must contain digits 1–9 exactly once.Each 3x3 box must contain digits 1–9 exactly once.
Output only the solved board as a single line of 81 digits (row by row, no spaces or newlines). Do not include any other text. </Instruction>
<Examples>
Input Puzzle:
004300209005009001070060043006002087190007400050083000600000105003508690042910300
Output: 864371259325849761971265843436192587198657432257483916689734125713528694542916378

Input Puzzle:
040100050107003960520008000000000017000906800803050620090060543600080700250097100
Output: 346179258187523964529648371965832417472916835813754629798261543631485792254397186
</Examples>

Input puzzle:
{puzzle}

Output:"""


    solve_prompt_cot = """<Instruction> Solve the following 9x9 Sudoku puzzle. The input is 81 digits, row by row, 0 for empty. Fill every row, column, and 3x3 box with digits 1-9 without repetition.
    Think step by step: identify cells with only one possible value, fill them, and repeat until the grid is complete. Each row, column, and 3x3 box must contain digits 1-9 exactly once. The final output must be the solved board as a single line of 81 digits, prefixed with "Output: ". </Instruction>

<Approach>
To solve the Sudoku step by step:
1. List the empty cells and, for each, which digits are still possible in its row, column, and 3x3 box.
2. Fill any cell that has only one possible digit (naked single).
3. Repeat step 1-2 until no more naked singles; then look for hidden singles (a digit that can go in only one cell in a row/column/box).
4. Continue until the grid is full. Output the final 81-digit string.
</Approach>

<Examples>
Input Puzzle:
004300209005009001070060043006002087190007400050083000600000105003508690042910300

Reasoning (short trace):
1) In the top-left box, column constraints force r2c1=8 and r3c1=6.
2) Row1 reduces to {{1,5,6,7}}; column/box interaction fixes r1c1=8 and r1c2=6.
3) Middle-left box resolves; row4 becomes 436192587.
4) Column and box propagation complete rows 5 and 6.
5) Bottom-right box resolves uniquely; remaining cells fill by hidden singles.

Output: 864371259325849761971265843436192587198657432257483916689734125713528694542916378

Input Puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605

Reasoning:
1) Box (r4–6,c4–6) forces r4c5=8, then r5c5=2; row4 becomes {{4,7}} → r4c1=7, r4c7=4.
2) Box (r7–9,c7–9) forces r7c9=1; then column/box updates force r8c7=7 and r3c7=8.
3) Column/box propagation fixes remaining values in column 7, then column 8 forces r9c8=9.
4) Top-middle box (r1–3,c4–6) resolves next; row1 completes to 496837152.
5) Continue applying naked/hidden singles with propagation until all rows and columns complete uniquely.

Output: 496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Examples>

Input puzzle:
{puzzle}

Think step by step, then give the final answer in the form:
Output:"""

    tot_improve_prompt = """<Instruction> The following is a 9x9 Sudoku puzzle and an attempted solution. The attempted solution is incorrect (some cells violate row/column/box rules or do not match the given clues). Fix it so that:
1. Every row contains digits 1-9 exactly once.
2. Every column contains digits 1-9 exactly once.
3. Every 3x3 box contains digits 1-9 exactly once.
4. All given clues in the puzzle are preserved in the solution.

Output only the corrected solution as a single line of 81 digits (row by row). Prefix with "Output: ". </Instruction>

<Approach>
To fix an incorrect Sudoku solution:
1. Compare the attempted solution to the original puzzle: ensure every non-empty clue in the puzzle appears in the same position in the solution.
2. For each row, column, and 3x3 box, check for duplicate digits; identify and correct the wrong cells.
3. Ensure no digit is missing in any row/column/box. Output the corrected 81-digit string.
</Approach>

<Examples>
Input Puzzle:
004300209005009001070060043006002087190007400050083000600000105003508690042910300
Incorrect attempt: 864371259325849761971265843436192587198657432257483916689734125713528694542916370
Reason: Last digit should be 8 not 0; and the solution must have 81 digits 1-9.
Output: 864371259325849761971265843436192587198657432257483916689734125713528694542916378

Input Puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605
Incorrect attempt: 496837152278541936315269874732685419189724563564913287657498321923156748841372690
Output: 496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Examples>

Input puzzle:
{puzzle}

Incorrect attempt:
{current}

Output:"""

    got1_split_prompt = """<Instruction>
Split the following Sudoku board (81 digits, 0 for empty) into 3 horizontal bands.
Band 1 = rows 1-3 (first 27 digits), Band 2 = rows 4-6 (next 27), Band 3 = rows 7-9 (last 27).
Output a valid JSON with keys "Band 1", "Band 2", "Band 3" and values the 27-digit strings. No other text.
{{
  "Band 1": "...",
  "Band 2": "...",
  "Band 3": "..."
}}
</Instruction>

<Example>
Board:
406807100000000036015209000032605019180704000000000200050400320900100008040372605

Output:
{{
  "Band 1": "406807100000000036015209000",
  "Band 2": "032605019180704000000000200",
  "Band 3": "050400320900100008040372605"
}}
</Example>

Board:
{board}

Output:"""

    got1_solve_band_prompt = """<Instruction>
You are solving one horizontal band (3 rows) of a Sudoku. The full puzzle is given below; fill only the empty cells in this band so that each row has 1-9, each column (in the full board) has 1-9, and each 3x3 box has 1-9.
Return only the 27-digit string for this band (rows concatenated), no explanation.
</Instruction>

<Example>
Full puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605

Band to fill (Band 1):
406807100000000036015209000

Output:
496837152278541936315269874
</Example>

Full puzzle (81 digits, 0 for empty):
{puzzle}

Band to fill ({band_id}, 27 digits): {band}
Output:"""

    got1_merge_prompt = """<Instruction>
Merge these 3 Sudoku bands into one 81-digit board. Band 1 is rows 1-3, Band 2 rows 4-6, Band 3 rows 7-9. Output only the 81-digit string.
</Instruction>


<Example>
Band 1:
496837152278541936315269874
Band 2:
732685419189724563564913287
Band 3:
657498321923156748841372695
Output:
496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Example>

Band 1: {band1}
Band 2: {band2}
Band 3: {band3}
Output:"""


    got1_improve_prompt = """<Instruction>
The following Sudoku board was built from 3 bands and may have conflicts (some cells may violate row/column/box rules or do not match the given clues). Fix it so that:
1. Every row contains digits 1-9 exactly once.
2. Every column contains digits 1-9 exactly once.
3. Every 3x3 box contains digits 1-9 exactly once.
4. All given clues in the puzzle are preserved in the solution.
Output only the corrected 81-digit string
</Instruction>

<Example>
Original puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605
Current board:
496837152278541936315269874732685419189724563564913287657498321923156748841372690
(conflict in last digit)
Output:
496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Example>

Original puzzle (81 digits):
{puzzle}

Current board: {board}
Output:"""

    got2_solve_block_prompt = """<Instruction>
You are solving ONE 3x3 block of a Sudoku (block {block_id}, counted row-major from 1 to 9). The full puzzle is given; fill only the empty cells in this block so it has digits 1-9 and respects row/column/box constraints. Return only the 9-digit string for this block (row by row within the block).
Rules:
- Fill only the empty cells in this block.
- The block must contain digits 1-9 exactly once.
- All row and column constraints from the full puzzle must be respected.
- Do NOT modify given digits.
- Return ONLY the 9-digit block string (row by row inside the block).
No explanation.
</Instruction>

<Example>
Full puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605

Block 1 (top-left 3x3):
406000015

Output:
496278315
</Example>

Full puzzle (81 digits):
{puzzle}

Block {block_id} (9 digits): {block}
Output (9 digits only):"""

    got2_merge_prompt = """<Instruction>
These 9 completed 3x3 blocks form a Sudoku.
Blocks are ordered row-major:
Block 1 = top-left, Block 9 = bottom-right.

Concatenate them properly into a full 81-digit Sudoku board.
Output ONLY the 81-digit string.
No explanation.</Instruction>

<Example>
Block 1: 496278315 Block 2: 837541269 Block 3: 152936874
Block 4: 732189724 Block 5: 685456913 Block 6: 419563287
Block 7: 657923156 Block 8: 498748372 Block 9: 321841695

Output:
496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Example>

Block 1: {b1} Block 2: {b2} Block 3: {b3}
Block 4: {b4} Block 5: {b5} Block 6: {b6}
Block 7: {b7} Block 8: {b8} Block 9: {b9}
Output (81 digits):"""

    got2_improve_prompt = """<Instruction>
The following Sudoku was built from 9 solved blocks but may contain conflicts.

Fix it so that:
- Every row contains digits 1-9 exactly once.
- Every column contains digits 1-9 exactly once.
- Every 3x3 box contains digits 1-9 exactly once.
- All original clues are preserved.

Output ONLY the corrected 81-digit string.
No explanation.
</Instruction>

<Example>
Original puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605

Current board (one incorrect value at end):
496837152278541936315269874732685419189724563564913287657498321923156748841372690

Output:
496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Example>

Original puzzle (81 digits):
{puzzle}

Current board: {board}
Output (81 digits only):"""

    got3_aggregate_prompt = """<Instruction>
You are given:
- A Sudoku puzzle.
- {n} candidate solutions generated independently.

Each candidate may contain errors.
Your task:
- Compare the candidates.
- Keep correct values that satisfy Sudoku rules.
- Fix conflicts.
- Preserve all original puzzle clues.
- Produce ONE fully valid solution.

Output ONLY the final 81-digit string.
No explanation.
</Instruction>

<Example>
Puzzle:
406807100000000036015209000032605019180704000000000200050400320900100008040372605

Candidates:
Candidate 1: 496837152278541936315269874732685419189724563564913287657498321923156748841372690
Candidate 2: 436897152278541936315269874732685419189724563564913287657498321923156748841372695
Candidate 3: 496837152278541936315269874732685419189724563564913287657498321923156748841372695

Output:
496837152278541936315269874732685419189724563564913287657498321923156748841372695
</Example>

Puzzle:
{puzzle}

Candidates:
{candidates}
Output (81 digits only):"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        if not state_dicts:
            return ""
        method = (state_dicts[0].get("method") or "").lower()
        puzzle = state_dicts[0].get("puzzle", "")
        if method == "got1":
            ordered = sorted(state_dicts, key=lambda s: (s.get("part") or ""))
            b1 = ordered[0].get("current", "")[:27] if ordered else ""
            b2 = ordered[1].get("current", "")[:27] if len(ordered) > 1 else ""
            b3 = ordered[2].get("current", "")[:27] if len(ordered) > 2 else ""
            return self.got1_merge_prompt.format(band1=b1, band2=b2, band3=b3)
        if method == "got2":
            def _block_num(s):
                m = re.search(r"\d+", str(s.get("part", "1")))
                return int(m.group(0)) if m else 0
            ordered = sorted(state_dicts, key=_block_num)
            blocks = [utils.extract_digits(s.get("current", ""), 9) for s in ordered]
            while len(blocks) < 9:
                blocks.append("0" * 9)
            return self.got2_merge_prompt.format(
                b1=blocks[0], b2=blocks[1], b3=blocks[2],
                b4=blocks[3], b5=blocks[4], b6=blocks[5],
                b7=blocks[6], b8=blocks[7], b9=blocks[8],
            )
        if method == "got3":
            candidates = "\n".join(
                f"Candidate {i+1}: {utils.extract_digits(s.get('current', ''), 81)}"
                for i, s in enumerate(state_dicts)
            )
            return self.got3_aggregate_prompt.format(n=len(state_dicts), puzzle=puzzle, candidates=candidates)
        return ""

    def generate_prompt(
        self,
        num_branches: int,
        puzzle: str,
        solution: str,
        current: str,
        method: str,
        **kwargs,
    ) -> str:
        if method.startswith("io"):
            return self.solve_prompt_io.format(puzzle=puzzle)
        elif method.startswith("cot"):
            return self.solve_prompt_cot.format(puzzle=puzzle)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.solve_prompt_io.format(puzzle=puzzle)
            return self.tot_improve_prompt.format(
                puzzle=puzzle,
                current=current,
            )
        elif method == "got1":
            phase = kwargs.get("phase", 0)
            if phase == 0:
                return self.got1_split_prompt.format(board=puzzle)
            if phase == 1:
                band = kwargs.get("subset", "")
                part = kwargs.get("part", "Band 1")
                bid = part.split()[-1] if part else "1"
                return self.got1_solve_band_prompt.format(
                    puzzle=puzzle,
                    band_id=bid,
                    band=band,
                )
            if phase == 2:
                return self.got1_improve_prompt.format(puzzle=puzzle, board=current or puzzle)
            return self.solve_prompt_io.format(puzzle=puzzle)
        elif method == "got2":
            phase = kwargs.get("phase", 1)
            if phase == 1:
                block = kwargs.get("subset", "")
                part = kwargs.get("part", "Block 1")
                bid = part.split()[-1] if part else "1"
                return self.got2_solve_block_prompt.format(
                    puzzle=puzzle,
                    block_id=bid,
                    block=block,
                )
            if phase == 2:
                return self.got2_improve_prompt.format(puzzle=puzzle, board=current or puzzle)
            return self.solve_prompt_io.format(puzzle=puzzle)
        elif method == "got3":
            if current is None or current == "":
                return self.solve_prompt_io.format(puzzle=puzzle)
            return self.tot_improve_prompt.format(puzzle=puzzle, current=current)
        else:
            raise AssertionError(f"Unknown method: {method}")

    def improve_prompt(self, **kwargs) -> str:
        """Return the appropriate improve prompt for ToT/got1/got2 from thought state."""
        puzzle = (kwargs.get("puzzle") or "").strip()
        current = (kwargs.get("current") or "").strip()
        method = (kwargs.get("method") or "").lower()
        if method == "got1":
            return self.got1_improve_prompt.format(puzzle=puzzle, board=current or puzzle)
        if method == "got2":
            return self.got2_improve_prompt.format(puzzle=puzzle, board=current or puzzle)
        # tot, got3, or default: fix incorrect attempt
        return self.tot_improve_prompt.format(puzzle=puzzle, current=current)

    def validation_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass


class SudokuParser(parser.Parser):
    """
    SudokuParser provides the parsing of language model responses specific to
    the Sudoku example. Extracts an 81-digit solution from the response.
    """

    def __init__(self) -> None:
        self.cache = {}

    @staticmethod
    def extract_solution(text: str) -> str:
        """Extract 81-digit solution from response. Prefer 'Output:' then digits, or <Solution> tags, then any 81 digits."""
        text = text.strip()
        # Prefer "Output: 81 digits" pattern (like sorting/set_intersection)
        if "Output:" in text:
            idx = text.find("Output:")
            rest = text[idx + len("Output:") :].strip()
            digits = re.sub(r"\D", "", rest)
            if len(digits) >= 81:
                return digits[:81]
        # Then content between <Solution> and </Solution>
        tag_start, tag_end = "<Solution>", "</Solution>"
        if tag_start in text and tag_end in text:
            start = text.rfind(tag_start) + len(tag_start)
            end = text.rfind(tag_end)
            chunk = text[start:end].strip()
            digits = re.sub(r"\D", "", chunk)
            if len(digits) >= 81:
                return digits[:81]
        # Any 81 consecutive digits
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 81:
            return digits[:81]
        # Do not pad short responses: num_errors() returns 81.0 when len(current) != 81
        if len(digits) > 0:
            return digits
        return ""

    def _extract_27(self, text: str) -> str:
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 27:
            return digits[:27]
        return digits.ljust(27, "0")[:27]

    def _extract_9(self, text: str) -> str:
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 9:
            return digits[:9]
        return digits.ljust(9, "0")[:9]

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        if not states:
            return []
        method = (states[0].get("method") or "").lower()
        new_states = []
        if method == "got2":
            # Merge 9 blocks deterministically (same order as aggregation_prompt)
            def _block_num(s):
                m = re.search(r"\d+", str(s.get("part", "1")))
                return int(m.group(0)) if m else 0
            ordered = sorted(states, key=_block_num)
            blocks = [utils.extract_digits(s.get("current", ""), 9) for s in ordered]
            merged = utils.merge_blocks(blocks)
            new_state = states[0].copy()
            new_state["current"] = merged
            new_state["phase"] = 2
            new_states.append(new_state)
            return new_states
        for text in texts:
            sol = self.extract_solution(text)
            new_state = states[0].copy()
            new_state["current"] = sol
            if method == "got1":
                new_state["phase"] = 2
            new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        method = (state.get("method") or "").lower()
        phase = state.get("phase", 0)
        new_states = []
        if method == "got1":
            if phase == 0:
                for text in texts:
                    try:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        if start >= 0 and end > start:
                            j = json.loads(text[start:end])
                            for k in ["Band 1", "Band 2", "Band 3"]:
                                band = utils.extract_digits(j.get(k, ""), 27)
                                new_states.append({
                                    **state,
                                    "subset": band,
                                    "phase": 1,
                                    "part": k,
                                    "current": "",
                                })
                            break
                    except Exception:
                        bands = utils.split_board_horizontally(state.get("puzzle", ""))
                        for i, b in enumerate(bands):
                            new_states.append({
                                **state,
                                "subset": b,
                                "phase": 1,
                                "part": f"Band {i+1}",
                                "current": "",
                            })
                        break
                if not new_states:
                    bands = utils.split_board_horizontally(state.get("puzzle", "0" * 81))
                    for i, b in enumerate(bands):
                        new_states.append({
                            **state,
                            "subset": b,
                            "phase": 1,
                            "part": f"Band {i+1}",
                            "current": "",
                        })
                return new_states
            if phase == 1:
                for text in texts:
                    band = self._extract_27(text)
                    new_states.append({**state, "current": band, "phase": 2})
                return new_states
            if phase == 2:
                for text in texts:
                    sol = self.extract_solution(text)
                    new_states.append({**state, "current": sol})
                return new_states
        if method == "got2":
            if phase == 1:
                for text in texts:
                    block = self._extract_9(text)
                    new_states.append({**state, "current": block, "phase": 2})
                return new_states
            if phase == 2:
                for text in texts:
                    sol = self.extract_solution(text)
                    new_states.append({**state, "current": sol})
                return new_states
        for text in texts:
            sol = self.extract_solution(text)
            new_state = state.copy()
            new_state["current"] = sol
            new_states.append(new_state)
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        if not texts:
            return {}
        sol = self.extract_solution(texts[0])
        return {"current": utils.extract_digits(sol, 81) if sol else (state.get("current") or "")}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.
    Simple: one generate, score, then ground truth check.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sudoku))
    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.
    Same graph as IO; the prompt asks the LLM to think step by step.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sudoku))
    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    ToT: generate multiple candidates, keep best, then improve (fix errors) and
    generate refinements, keep best. Mirrors equation_solving / set_intersection pattern.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()
    branch_factor = 8

    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.KeepBestN(1, False))
    # Improve: single fix step using tot_improve_prompt (via improve_prompt)
    operations_graph.append_operation(operations.Improve())
    # Level 2: generate refinements from improved thought, keep best
    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sudoku))
    return operations_graph


def got1() -> operations.GraphOfOperations:
    """
    GoT-1: Horizontal Bands. Split into 3 bands, solve each, merge, improve & refine.
    """
    ops = operations.GraphOfOperations()
    split = operations.Generate(1, 1)
    ops.append_operation(split)
    keep_ops = []
    for i in range(1, 4):
        band_name = f"Band {i}"
        sel = operations.Selector(
            lambda thoughts, band_name=band_name: [
                t for t in thoughts if t.state.get("part") == band_name
            ]
        )
        sel.add_predecessor(split)
        ops.add_operation(sel)
        gen = operations.Generate(1, 4)
        gen.add_predecessor(sel)
        ops.add_operation(gen)
        score = operations.Score(1, False, utils.band_errors)
        score.add_predecessor(gen)
        ops.add_operation(score)
        keep = operations.KeepBestN(1, False)
        keep.add_predecessor(score)
        ops.add_operation(keep)
        keep_ops.append(keep)
    agg = operations.Aggregate(1)
    for k in keep_ops:
        agg.add_predecessor(k)
    ops.add_operation(agg)
    ops.append_operation(operations.Generate(1, 4))
    ops.append_operation(operations.Score(1, False, utils.num_errors))
    ops.append_operation(operations.KeepBestN(1, False))
    ops.append_operation(operations.GroundTruth(utils.test_sudoku))
    return ops


def got2() -> operations.GraphOfOperations:
    """
    GoT-2: 9 Blocks. Split into 9 blocks (deterministic), solve each, merge, improve & refine.
    """
    ops = operations.GraphOfOperations()

    def make_nine_thoughts(thoughts):
        if not thoughts:
            return []
        state = thoughts[0].state
        puzzle = state.get("puzzle", "0" * 81)
        from graph_of_thoughts.operations.thought import Thought
        return [
            Thought({
                **state,
                "part": f"Block {i + 1}",
                "subset": utils.get_block_from_board(puzzle, i),
                "phase": 1,
                "current": "",
            })
            for i in range(9)
        ]

    sel_root = operations.Selector(make_nine_thoughts)
    ops.add_operation(sel_root)
    keep_ops = []
    for i in range(1, 10):
        block_name = f"Block {i}"
        sel = operations.Selector(
            lambda thoughts, block_name=block_name: [
                t for t in thoughts if t.state.get("part") == block_name
            ]
        )
        sel.add_predecessor(sel_root)
        ops.add_operation(sel)
        gen = operations.Generate(1, 3)
        gen.add_predecessor(sel)
        ops.add_operation(gen)
        score = operations.Score(1, False, utils.block_errors)
        score.add_predecessor(gen)
        ops.add_operation(score)
        keep = operations.KeepBestN(1, False)
        keep.add_predecessor(score)
        ops.add_operation(keep)
        keep_ops.append(keep)
    agg = operations.Aggregate(1)
    for k in keep_ops:
        agg.add_predecessor(k)
    ops.add_operation(agg)
    ops.append_operation(operations.Generate(1, 4))
    ops.append_operation(operations.Score(1, False, utils.num_errors))
    ops.append_operation(operations.KeepBestN(1, False))
    ops.append_operation(operations.GroundTruth(utils.test_sudoku))
    return ops


def got3() -> operations.GraphOfOperations:
    """
    GoT-3: Full Board. Generate multiple candidates, aggregate best, improve & refine.
    """
    ops = operations.GraphOfOperations()
    ops.append_operation(operations.Generate(1, 5))
    ops.append_operation(operations.Score(1, False, utils.num_errors))
    ops.append_operation(operations.KeepBestN(3, False))
    ops.append_operation(operations.Aggregate(3))
    ops.append_operation(operations.Score(1, False, utils.num_errors))
    ops.append_operation(operations.KeepBestN(1, False))
    ops.append_operation(operations.Generate(1, 4))
    ops.append_operation(operations.Score(1, False, utils.num_errors))
    ops.append_operation(operations.KeepBestN(1, False))
    ops.append_operation(operations.GroundTruth(utils.test_sudoku))
    return ops


def _load_data():
    """Load sudoku.csv with difficulty labels."""
    data_path = os.path.join(os.path.dirname(__file__), "sudoku.csv")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unsolved = (row.get("UnsolvedBoared") or row.get("UnsolvedBoard") or "").strip()
            solved = (row.get("SolvedBoared") or row.get("SolvedBoard") or "").strip()
            if unsolved and solved:
                data.append({
                    "puzzle": unsolved,
                    "solution": solved,
                    "difficulty": utils.classify_difficulty(unsolved),
                })
    return data


def _select_difficulty_balanced_samples(
    last_n: int = 50,
    per_difficulty: int = 5,
) -> List[int]:
    """
    From the last `last_n` samples in the dataset, select up to `per_difficulty`
    easy, medium, and hard each. Returns list of indices into the full dataset.
    """
    data = _load_data()
    if last_n >= len(data):
        pool_indices = list(range(len(data)))
    else:
        pool_indices = list(range(len(data) - last_n, len(data)))
    by_diff: Dict[str, List[int]] = {"easy": [], "medium": [], "hard": []}
    for i in pool_indices:
        d = data[i]["difficulty"]
        if d in by_diff and len(by_diff[d]) < per_difficulty:
            by_diff[d].append(i)
    result = []
    for d in ("easy", "medium", "hard"):
        result.extend(by_diff[d][:per_difficulty])
    return sorted(result)


def _get_lm(lm_name: str, config_path: str, cache: bool = True):
    """Instantiate the appropriate LM (ChatGPT, Gemini, or Llama2HF) from config key."""
    gemini_keys = {"gemini", "gemini-lite", "gemini-pro"}
    llama_keys = {"llama4-17b", "llama70b-hf", "llama13b-hf"}
    if lm_name in llama_keys:
        return language_models.Llama2HF(config_path, model_name=lm_name, cache=cache)
    if lm_name in gemini_keys:
        return language_models.Gemini(config_path, model_name=lm_name, cache=cache)
    return language_models.ChatGPT(config_path, model_name=lm_name, cache=cache)


def run(
    data_ids: Optional[List[int]],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
    difficulty_levels: Optional[List[str]] = None,
) -> tuple:
    """
    Execute each method on each sample. Optionally filter by difficulty (easy/medium/hard).
    Records timing and token/cost in result JSONs for evaluation.
    Returns (spent_budget, results_folder path).
    """
    orig_budget = budget
    data = _load_data()
    if difficulty_levels:
        data = [d for d in data if d["difficulty"] in difficulty_levels]
    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([m.__name__ for m in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data_ids": data_ids,
        "methods": [m.__name__ for m in methods],
        "lm": lm_name,
        "budget": budget,
        "difficulty_filter": difficulty_levels,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    log_path = os.path.join(results_folder, "log.log")
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)

    logging.info(f"Running model: {lm_name} with {len(selected_data)} samples")

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__), exist_ok=True)

    for idx, sample in enumerate(selected_data):
        data_id = data_ids[idx] if idx < len(data_ids) else idx
        puzzle = sample["puzzle"]
        solution = sample["solution"]
        difficulty = sample.get("difficulty", "unknown")
        logging.info(f"Running sample {data_id} difficulty={difficulty}")
        if budget <= 0.0:
            break
        for method in methods:
            if budget <= 0.0:
                break
            logging.info(f"Method {method.__name__}, budget left: {budget}")
            lm = _get_lm(
                lm_name,
                os.path.join(
                    os.path.dirname(__file__),
                    "../../graph_of_thoughts/language_models/config.json",
                ),
                cache=True,
            )
            operations_graph = method()
            initial_state = {
                "puzzle": puzzle,
                "solution": solution,
                "current": "",
                "method": method.__name__,
            }
            if method.__name__ == "got1":
                initial_state["phase"] = 0
            if method.__name__ == "got2":
                initial_state["phase"] = 1
            executor = controller.Controller(
                lm,
                operations_graph,
                SudokuPrompter(),
                SudokuParser(),
                initial_state,
            )
            response_time_sec = None
            t0 = time.perf_counter()
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            t1 = time.perf_counter()
            response_time_sec = t1 - t0
            path = os.path.join(results_folder, method.__name__, f"{data_id}.json")
            executor.output_graph(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    output = json.load(f)
                if output and isinstance(output[-1], dict):
                    output[-1]["difficulty"] = difficulty
                    output[-1]["response_time"] = round(response_time_sec, 2)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to add response_time/difficulty to {path}: {e}")
            budget -= lm.cost

    return orig_budget - budget, results_folder


def _graph_metrics(method_fn: Optional[Callable[[], operations.GraphOfOperations]]) -> Dict[str, Any]:
    """Compute branching factor, depth, and node count for a method's graph."""
    if method_fn is None:
        return {}
    g = method_fn()
    ops = g.operations
    num_ops = len(ops)
    generate_ops = [o for o in ops if getattr(o, "num_branches_response", None) is not None]
    max_branch = max((getattr(o, "num_branches_response", 1) for o in generate_ops), default=1)
    depth = 0
    if g.roots:
        visited = set()
        stack = [(g.roots[0], 0)]
        while stack:
            op, d = stack.pop()
            if op in visited:
                continue
            visited.add(op)
            depth = max(depth, d)
            for s in op.successors:
                stack.append((s, d + 1))
    return {
        "branching_factor": max_branch,
        "graph_depth": depth,
        "num_operations": num_ops,
    }


def evaluate_run(results_folder: str) -> Dict[str, Any]:
    """
    Load all result JSONs in results_folder and compute evaluation metrics.
    Returns a dict with per-method and per-sample metrics, and writes evaluation.csv and summary.json.
    """
    config_path = os.path.join(results_folder, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        config = json.load(f)
    methods = config.get("methods", [])
    lm_name = config.get("lm", "")
    difficulty_filter = config.get("difficulty_filter")

    method_fns = {"io": io, "cot": cot, "tot": tot, "got1": got1, "got2": got2, "got3": got3}
    rows = []
    by_method = {m: {"success": 0, "total": 0, "errors_sum": 0.0, "tokens_prompt": 0, "tokens_completion": 0, "cost_sum": 0.0, "time_sum": 0.0} for m in methods}

    for method_name in methods:
        method_dir = os.path.join(results_folder, method_name)
        if not os.path.isdir(method_dir):
            continue
        graph_meta = _graph_metrics(method_fns.get(method_name))
        for fn in os.listdir(method_dir):
            if not fn.endswith(".json") or fn.endswith("_metrics.json"):
                continue
            sample_id = fn.replace(".json", "").replace("_metrics", "")
            if "_metrics" in fn:
                continue
            json_path = os.path.join(method_dir, fn)
            metrics_path = os.path.join(method_dir, f"{sample_id}_metrics.json")
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception:
                continue
            solved = False
            final_errors = 81.0
            prompt_tokens = 0
            completion_tokens = 0
            cost_usd = 0.0
            for item in data:
                if not isinstance(item, dict):
                    continue
                if item.get("operation") == "ground_truth_evaluator":
                    if "problem_solved" in item:
                        solved = any(item["problem_solved"])
                    if "thoughts" in item and item["thoughts"]:
                        final_state = item["thoughts"][-1]
                        final_errors = utils.num_errors(final_state)
                if "prompt_tokens" in item:
                    prompt_tokens = item.get("prompt_tokens", 0)
                    completion_tokens = item.get("completion_tokens", 0)
                    cost_usd = item.get("cost", 0.0)
            time_sec = 0.0
            difficulty = "unknown"
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path) as f:
                        m = json.load(f)
                    time_sec = m.get("time_elapsed_sec", 0.0)
                    difficulty = m.get("difficulty", "unknown")
                    prompt_tokens = m.get("prompt_tokens", prompt_tokens)
                    completion_tokens = m.get("completion_tokens", completion_tokens)
                    cost_usd = m.get("cost_usd", cost_usd)
                except Exception:
                    pass
            # Read from main JSON last block when we write difficulty/response_time there
            for item in data:
                if isinstance(item, dict) and "response_time" in item:
                    time_sec = item.get("response_time", time_sec)
                if isinstance(item, dict) and "difficulty" in item:
                    difficulty = item.get("difficulty", difficulty)
            rows.append({
                "method": method_name,
                "sample_id": sample_id,
                "difficulty": difficulty,
                "success": solved,
                "num_errors": final_errors,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd,
                "time_sec": time_sec,
            })
            by_method[method_name]["total"] += 1
            if solved:
                by_method[method_name]["success"] += 1
            by_method[method_name]["errors_sum"] += final_errors
            by_method[method_name]["tokens_prompt"] += prompt_tokens
            by_method[method_name]["tokens_completion"] += completion_tokens
            by_method[method_name]["cost_sum"] += cost_usd
            by_method[method_name]["time_sum"] += time_sec

    summary = {
        "model": lm_name,
        "difficulty_filter": difficulty_filter,
        "methods": {},
        "search_metrics": {},
    }
    for m in methods:
        summary["methods"][m] = {
            "success_rate": by_method[m]["success"] / by_method[m]["total"] if by_method[m]["total"] else 0,
            "avg_errors": by_method[m]["errors_sum"] / by_method[m]["total"] if by_method[m]["total"] else 81,
            "total_prompt_tokens": by_method[m]["tokens_prompt"],
            "total_completion_tokens": by_method[m]["tokens_completion"],
            "total_tokens": by_method[m]["tokens_prompt"] + by_method[m]["tokens_completion"],
            "total_cost_usd": round(by_method[m]["cost_sum"], 6),
            "total_time_sec": round(by_method[m]["time_sum"], 3),
            "n_samples": by_method[m]["total"],
        }
        summary["search_metrics"][m] = _graph_metrics(method_fns.get(m))

    csv_path = os.path.join(results_folder, "evaluation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sample_id", "difficulty", "success", "num_errors", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "time_sec"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(results_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


SUDOKU_MODELS = [
    # "chatgpt4-nano",      # ChatGPT 4.1 nano
    "chatgpt5.2",         # ChatGPT 5.2
    "chatgpt5.1",         # ChatGPT 5.1
    "chatgpt5-mini",      # ChatGPT 5 mini
    "chatgpt5-nano",      # ChatGPT 5 nano
    "gemini",             # Gemini 2.5 flash
    "gemini-lite",        # Gemini 2.5 flash lite
    "gemini-pro",         # Gemini 2.5 pro
    "llama70b-hf",        # Llama 2 70B
]


SUDOKU_MODEL_SAMPLES = {
    "chatgpt4": 3,
    "gemini": 1,
}


def _select_random_easy_medium(n: int, seed: Optional[int] = None) -> List[int]:
    """
    Select n sample indices so both easy and medium appear in results when available.
    Picks ~2 easy, ~2 medium, rest from pool to reach n.
    """
    data = _load_data()
    n_total = len(data)
    if n_total == 0 or n <= 0:
        return []
    if seed is not None:
        random.seed(seed)
    by_diff: Dict[str, List[int]] = {"easy": [], "medium": [], "hard": []}
    for i in range(n_total):
        d = data[i].get("difficulty", "medium")
        if d in by_diff:
            by_diff[d].append(i)
    chosen = []
    n_easy = min(len(by_diff["easy"]), 2)
    n_medium = min(len(by_diff["medium"]), max(0, n - n_easy))
    n_rest = n - n_easy - n_medium
    if n_easy:
        chosen.extend(random.sample(by_diff["easy"], n_easy))
    if n_medium:
        chosen.extend(random.sample(by_diff["medium"], n_medium))
    pool = [i for i in range(n_total) if i not in chosen]
    if n_rest > 0 and pool:
        chosen.extend(random.sample(pool, min(n_rest, len(pool))))
    return chosen[:n]


def run_per_model_samples(
    budget: float,
    methods: List[Callable[[], operations.GraphOfOperations]],
    seed: Optional[int] = None,
) -> float:
    """
    Run sudoku with 5 random samples per model (mix of easy and medium in results).
    Models: chatgpt4, chatgpt4-nano, chatgpt4-mini, gemini, gemini-lite, llama4-17b.
    Each result JSON gets difficulty (from sample) and response_time in the last block.
    """
    data = _load_data()
    n_total = len(data)
    if n_total == 0:
        logging.warning("No data in sudoku.csv")
        return 0.0

    total_spent = 0.0
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )
    for lm_name, n_samples in SUDOKU_MODEL_SAMPLES.items():
        if budget <= 0.0:
            logging.warning("Budget depleted, skipping remaining models.")
            break
        n = min(n_samples, n_total)
        data_ids = _select_random_easy_medium(n, seed=seed)
        if len(data_ids) < n:
            data_ids = data_ids + random.sample([i for i in range(n_total) if i not in data_ids], n - len(data_ids))
        data_ids = data_ids[:n]
        logging.info("Running %s with %s samples (ids=%s)", lm_name, len(data_ids), data_ids)
        try:
            spent, _ = run(data_ids, methods, budget, lm_name, difficulty_levels=None)
            total_spent += spent
            budget -= spent
        except OSError as e:
            if "gated" in str(e).lower() or "401" in str(e) or "unauthorized" in str(e).lower():
                logging.warning("Skipping %s: %s", lm_name, e)
            else:
                raise
        except Exception as e:
            if "GatedRepo" in type(e).__name__ or "401" in str(e):
                logging.warning("Skipping %s: %s", lm_name, e)
            else:
                raise
    return total_spent


def run_multi_model(
    models: List[str],
    budget_per_model: float = 5000.0,
    last_n: int = 50,
    per_difficulty: int = 5,
    methods: Optional[List[Callable[[], operations.GraphOfOperations]]] = None,
) -> List[str]:
    """
    Run sudoku for each model on 5 easy + 5 medium + 5 hard samples from the last `last_n`.
    Returns list of results folder paths.
    """
    if methods is None:
        methods = [io, cot, tot, got1, got2, got3]
    sample_ids = _select_difficulty_balanced_samples(last_n=last_n, per_difficulty=per_difficulty)
    results_folders = []
    for lm_name in models:
        try:
            spent, folder = run(sample_ids, methods, budget_per_model, lm_name, difficulty_levels=None)
            with open(os.path.join(folder, "log.log"), "a", encoding="utf-8") as lf:
                lf.write(f"Spent {spent:.2f} for {lm_name}\n")
            results_folders.append(folder)
        except Exception as e:
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            fallback_log = os.path.join(results_dir, "multi_run_errors.log")
            with open(fallback_log, "a", encoding="utf-8") as lf:
                lf.write(f"Model {lm_name} failed: {e}\n")
    return results_folders


if __name__ == "__main__":
    """
    Run sudoku_final for 5 easy + 5 medium + 5 hard (from last 50 samples) with all 6 methods,
    for each of: ChatGPT 4.1 nano, 5.2, 5.1, 5 mini, 5 nano; Gemini 2.5 flash, flash lite, pro; Llama 2.
    Logs go to each model's results folder in log.log (no console logging).
    """
    
    budget = 1000
    approaches = [io, cot, tot, got1, got2, got3]
    spent = run_per_model_samples(budget, approaches, seed=41)
    logging.info(f"Spent {spent} out of {budget} budget.")