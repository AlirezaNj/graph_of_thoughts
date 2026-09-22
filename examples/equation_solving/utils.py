# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""
Utilities for the equation_solving2 example: system of two linear equations in x and y.
Parsing (x, y), scoring, and ground truth.
"""

import re
from typing import Dict, Optional, Tuple


# Tolerance for float comparison in ground truth
TOLERANCE = 1e-6


def parse_two_numbers(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract two numeric values (x, y) from a string.
    Prefers the *last* occurrence of "x = ..." and "y = ..." (final answer in step-by-step text).
    Fallback: last "Output: a, b", then last two numbers in the string.

    :param text: Raw LLM output or thought state string.
    :return: (x, y) or (None, None) if parsing fails.
    """
    if not text or not isinstance(text, str):
        return None, None
    text = text.strip()
    # Prefer *last* "x = ..." and "y = ..." so we get the final answer in long CoT/ToT outputs
    x_matches = list(re.finditer(r"x\s*=\s*(-?\d+\.?\d*)", text, re.IGNORECASE))
    y_matches = list(re.finditer(r"y\s*=\s*(-?\d+\.?\d*)", text, re.IGNORECASE))
    if x_matches and y_matches:
        mx = x_matches[-1]
        my = y_matches[-1]
        try:
            return float(mx.group(1)), float(my.group(1))
        except ValueError:
            pass
    # Try last "Output: a, b" in text (in case model uses that format at the end)
    m_outputs = list(re.finditer(
        r"Output\s*:\s*(-?\d+\.?\d*)\s*[, \t]+\s*(-?\d+\.?\d*)", text, re.IGNORECASE
    ))
    if m_outputs:
        m = m_outputs[-1]
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    # Fallback: use the *last* two numbers in the string (final answer)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if len(numbers) >= 2:
        try:
            return float(numbers[-2]), float(numbers[-1])
        except ValueError:
            pass
    return None, None


def _solution_to_xy(solution: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse solution/result string 'x,y' into (x, y)."""
    if not solution:
        return None, None
    parts = str(solution).strip().replace(" ", "").split(",")
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return None, None


def num_errors(state: Dict) -> float:
    """
    Score by sum of absolute errors for x and y. Lower is better.

    :param state: "current" (LLM answer as "x,y" or text), "solution" or "result" (ground truth "x,y").
    :return: |x_cur - x_gt| + |y_cur - y_gt|, or 1000.0 on parse failure.
    """
    try:
        gt = state.get("solution") or state.get("result") or ""
        x_gt, y_gt = _solution_to_xy(gt)
        cur = state.get("current") or ""
        x_cur, y_cur = parse_two_numbers(cur)
        if x_cur is None or y_cur is None:
            x_cur, y_cur = _solution_to_xy(cur)
        if x_gt is None or y_gt is None or x_cur is None or y_cur is None:
            return 1000.0
        return abs(x_cur - x_gt) + abs(y_cur - y_gt)
    except Exception:
        return 1000.0


def num_errors_part(state: Dict) -> float:
    """Score for GoT per-equation step; no per-part ground truth."""
    return 0.0


def test_equation(state: Dict) -> bool:
    """
    Ground truth: both x and y match within tolerance.

    :param state: "current" (LLM answer), "result" or "solution" (ground truth "x,y").
    """
    try:
        gt = state.get("result") or state.get("solution") or ""
        x_gt, y_gt = _solution_to_xy(gt)
        cur = state.get("current") or ""
        x_cur, y_cur = parse_two_numbers(cur)
        if x_cur is None or y_cur is None:
            x_cur, y_cur = _solution_to_xy(cur)
        if x_gt is None or y_gt is None or x_cur is None or y_cur is None:
            return False
        return abs(x_cur - x_gt) <= TOLERANCE and abs(y_cur - y_gt) <= TOLERANCE
    except Exception:
        return False
