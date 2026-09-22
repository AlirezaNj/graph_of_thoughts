# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import re
from typing import Dict, List

# Standard 9x9 Sudoku grid size
SUDOKU_SIZE = 9
BOX_SIZE = 3


def board_string_to_grid(board: str) -> List[List[str]]:
    """
    Convert an 81-character board string (row-major, '0' or '.' for empty)
    into a 9x9 grid of single-character strings.

    :param board: String of 81 characters (digits 0-9 or . for empty).
    :type board: str
    :return: 9x9 grid where each cell is a single character.
    :rtype: List[List[str]]
    """
    board = board.strip().replace(".", "0")
    if len(board) != 81:
        raise ValueError(f"Board must have 81 characters, got {len(board)}")
    grid = []
    for r in range(SUDOKU_SIZE):
        row = []
        for c in range(SUDOKU_SIZE):
            ch = board[r * SUDOKU_SIZE + c]
            if ch not in "0123456789":
                ch = "0"
            row.append(ch)
        grid.append(row)
    return grid


def grid_to_board_string(grid: List[List[str]]) -> str:
    """
    Convert a 9x9 grid to an 81-character board string.

    :param grid: 9x9 grid of single-character strings.
    :type grid: List[List[str]]
    :return: 81-character string.
    :rtype: str
    """
    return "".join(
        grid[r][c].replace(".", "0")[0]
        for r in range(SUDOKU_SIZE)
        for c in range(SUDOKU_SIZE)
    )


def format_board_display(board: str) -> str:
    """
    Format an 81-character board string as a readable 9x9 grid with box borders.
    Uses '.' for empty cells (0).

    :param board: 81-character board string.
    :type board: str
    :return: Multi-line string for display.
    :rtype: str
    """
    grid = board_string_to_grid(board)
    lines = []
    for r in range(SUDOKU_SIZE):
        row_str = ""
        for c in range(SUDOKU_SIZE):
            if c > 0 and c % BOX_SIZE == 0:
                row_str += "| "
            row_str += (grid[r][c] if grid[r][c] != "0" else ".") + " "
        if r > 0 and r % BOX_SIZE == 0:
            lines.append("------+-------+------")
        lines.append(row_str.strip())
    return "\n".join(lines)


def count_constraint_errors(board: str) -> int:
    """
    Count the number of constraint violations in a filled board (duplicates in
    row, column, or 3x3 box).

    :param board: 81-character board string (use '0' for empty).
    :type board: str
    :return: Total number of duplicate violations.
    :rtype: int
    """
    grid = board_string_to_grid(board)
    errors = 0
    for i in range(SUDOKU_SIZE):
        row_vals = [grid[i][j] for j in range(SUDOKU_SIZE) if grid[i][j] != "0"]
        errors += len(row_vals) - len(set(row_vals))
        col_vals = [grid[j][i] for j in range(SUDOKU_SIZE) if grid[j][i] != "0"]
        errors += len(col_vals) - len(set(col_vals))
    for br in range(0, SUDOKU_SIZE, BOX_SIZE):
        for bc in range(0, SUDOKU_SIZE, BOX_SIZE):
            box_vals = [
                grid[r][c]
                for r in range(br, br + BOX_SIZE)
                for c in range(bc, bc + BOX_SIZE)
                if grid[r][c] != "0"
            ]
            errors += len(box_vals) - len(set(box_vals))
    return errors


def test_sudoku(state: Dict) -> bool:
    """
    Ground truth check: whether the current solution matches the expected
    solution.

    :param state: Thought state with "current" (solution) and "solution" (expected).
    :type state: Dict
    :return: True if current matches solution.
    :rtype: bool
    """
    try:
        current = state.get("current") or ""
        solution = state.get("solution") or ""
        current = current.strip().replace(" ", "").replace("\n", "").replace(".", "0")
        solution = solution.strip().replace(" ", "").replace("\n", "").replace(".", "0")
        if len(current) != 81 or len(solution) != 81:
            return False
        return current == solution
    except Exception:
        return False


def num_errors(state: Dict) -> float:
    """
    Score = number of cell differences from the ground truth solution.
    Lower is better. Used with KeepBestN(..., higher_is_better=False).

    :param state: Thought state with "current" and "solution".
    :type state: Dict
    :return: Number of errors (float for compatibility).
    :rtype: float
    """
    try:
        current = (state.get("current") or "").strip().replace(" ", "").replace("\n", "").replace(".", "0")
        if len(current) != 81:
            return 81.0
        solution = state.get("solution")
        if solution is not None and solution != "":
            solution = str(solution).strip().replace(" ", "").replace("\n", "").replace(".", "0")
            if len(solution) == 81:
                return float(sum(1 for a, b in zip(current, solution) if a != b))
        return float(count_constraint_errors(current))
    except Exception:
        return 81.0


def extract_digits(text: str, expected_len: int) -> str:
    """
    Extract up to expected_len digits from text. Pad with zeros if too short.
    """
    if text is None:
        return "0" * expected_len
    digits = re.sub(r"\D", "", str(text))
    if len(digits) >= expected_len:
        return digits[:expected_len]
    return digits.ljust(expected_len, "0")[:expected_len]


def split_board_horizontally(board_str: str) -> List[str]:
    """Split 81-char board into 3 horizontal bands (rows 0-2, 3-5, 6-8), each 27 chars."""
    board_str = extract_digits(board_str, 81)
    return [board_str[i * 27 : (i + 1) * 27] for i in range(3)]


def merge_bands(bands: List[str]) -> str:
    """Merge 3 bands (27 chars each) into 81-char board."""
    assert len(bands) == 3
    return "".join(extract_digits(b, 27) for b in bands)


def block_index_to_slice(block_index: int):
    """block_index 0..8 (row-major: 0=top-left, 8=bottom-right). Returns (r0, r1, c0, c1)."""
    br, bc = block_index // 3, block_index % 3
    return (br * 3, br * 3 + 3, bc * 3, bc * 3 + 3)


def get_block_from_board(board_str: str, block_index: int) -> str:
    """Extract 9-char block (row-major within block) from 81-char board."""
    board_str = extract_digits(board_str, 81)
    grid = board_string_to_grid(board_str)
    r0, r1, c0, c1 = block_index_to_slice(block_index)
    chars = [grid[r][c] for r in range(r0, r1) for c in range(c0, c1)]
    return "".join(chars)


def split_board_into_blocks(board_str: str) -> List[str]:
    """Split 81-char board into 9 blocks (block 0..8), each 9 chars."""
    board_str = extract_digits(board_str, 81)
    return [get_block_from_board(board_str, i) for i in range(9)]


def merge_blocks(blocks: List[str]) -> str:
    """Merge 9 blocks (9 chars each) into 81-char board (row-major)."""
    assert len(blocks) == 9
    grid = [[None] * SUDOKU_SIZE for _ in range(SUDOKU_SIZE)]
    for block_index in range(9):
        r0, r1, c0, c1 = block_index_to_slice(block_index)
        b = extract_digits(blocks[block_index], 9)
        idx = 0
        for r in range(r0, r1):
            for c in range(c0, c1):
                grid[r][c] = b[idx]
                idx += 1
    return "".join(grid[r][c] for r in range(SUDOKU_SIZE) for c in range(SUDOKU_SIZE))


def band_errors(state: Dict) -> float:
    """Errors for a 27-char band: 27 minus matches with ground-truth band. Lower is better."""
    try:
        current = extract_digits(state.get("current", ""), 27)
        solution = state.get("solution") or ""
        solution = extract_digits(solution, 81)
        part = state.get("part", "Band 1")
        band_idx = 1
        if isinstance(part, str) and "Band" in part:
            try:
                band_idx = int(part.split()[-1])
            except Exception:
                pass
        start = (band_idx - 1) * 27
        gt_band = solution[start : start + 27]
        if len(gt_band) != 27:
            return 27.0
        return float(27 - sum(1 for a, b in zip(current, gt_band) if a == b))
    except Exception:
        return 27.0


def block_errors(state: Dict) -> float:
    """Errors for a 9-char block: 9 minus matches with ground-truth block. Lower is better."""
    try:
        current = extract_digits(state.get("current", ""), 9)
        solution = state.get("solution") or ""
        solution = extract_digits(solution, 81)
        part = state.get("part", "Block 1")
        block_idx = 0
        if isinstance(part, str) and "Block" in part:
            try:
                block_idx = int(part.split()[-1]) - 1
            except Exception:
                pass
        block_idx = max(0, min(8, block_idx))
        gt_block = get_block_from_board(solution, block_idx)
        if len(gt_block) != 9:
            return 9.0
        return float(9 - sum(1 for a, b in zip(current, gt_block) if a == b))
    except Exception:
        return 9.0


def count_clues(board: str) -> int:
    """Number of given (non-empty) cells."""
    return sum(1 for ch in extract_digits(board, 81) if ch in "123456789")


def classify_difficulty(board: str) -> str:
    """Heuristic: easy >= 35 clues, medium 28-34, hard < 28."""
    n = count_clues(board)
    if n >= 35:
        return "easy"
    if n >= 28:
        return "medium"
    return "hard"
