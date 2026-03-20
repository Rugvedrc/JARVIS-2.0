"""Validator layer: objective verification of every agent action result.

Called by the orchestrator immediately after each shell or file-write action.
Returns a ValidationResult so the orchestrator can:
  - update RunMetrics counters
  - feed failure context back into the agent's memory
  - emit a "validation_result" event for display

Design principle: zero LLM calls.  All checks are deterministic.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field


# ── Failure signal patterns (regex, case-insensitive) ────────────────────────
# Ordered from most-specific to least-specific.

_FAIL_PATTERNS: list[re.Pattern] = [
    re.compile(r, re.IGNORECASE) for r in [
        r"\bTraceback \(most recent call last\)",
        r"\bSyntaxError\b",
        r"\bNameError\b",
        r"\bTypeError\b",
        r"\bValueError\b",
        r"\bImportError\b",
        r"\bModuleNotFoundError\b",
        r"\bFileNotFoundError\b",
        r"\bPermissionError\b",
        r"\bcommand not found\b",
        r"\bnot recognized as an internal or external command\b",
        r"\bNo such file or directory\b",
        r"\bERROR:\b",
        r"\bFAILED\b",
        r"exit code [1-9]\d*",
    ]
]

_SUCCESS_HINTS: list[re.Pattern] = [
    re.compile(r, re.IGNORECASE) for r in [
        r"\bOK\b",
        r"\bPASSED\b",
        r"\bsuccess(?:fully)?\b",
        r"\bcreated\b",
        r"\bwritten\b",
        r"\bdone\b",
    ]
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)

    def as_context(self) -> str:
        """Short string to append to agent action feedback."""
        emoji = "✓" if self.passed else "✗"
        return f"[VALIDATOR {emoji}] {self.reason}"


# ── Core validators ───────────────────────────────────────────────────────────

def validate_shell_output(cmd: str, output: str) -> ValidationResult:
    """Infer pass/fail from shell command output using heuristic patterns.

    Rules (in priority order):
      1. Explicit ERROR: prefix (case-insensitive) → fail
      2. Any fail pattern matched → fail
      3. Empty output with no success hint → neutral (pass, not penalised)
      4. Otherwise → pass
    """
    if output.upper().startswith("ERROR:"):
        return ValidationResult(
            passed=False,
            reason=output[:120],
            details={"cmd": cmd[:80]},
        )
    for pat in _FAIL_PATTERNS:
        m = pat.search(output)
        if m:
            return ValidationResult(
                passed=False,
                reason=f"failure pattern '{pat.pattern}' found",
                details={"match": m.group(0)[:80], "cmd": cmd[:80]},
            )
    return ValidationResult(
        passed=True,
        reason="no failure patterns detected",
        details={"cmd": cmd[:80]},
    )


def validate_python_syntax(code: str) -> ValidationResult:
    """Check that a Python code string has valid syntax via ast.parse."""
    try:
        ast.parse(code)
        return ValidationResult(passed=True, reason="syntax OK")
    except SyntaxError as e:
        return ValidationResult(
            passed=False,
            reason=f"SyntaxError at line {e.lineno}: {e.msg}",
            details={"lineno": e.lineno, "msg": e.msg},
        )


def validate_file_exists(path: str, min_bytes: int = 1) -> ValidationResult:
    """Verify that a file was actually written and is non-empty."""
    if not os.path.exists(path):
        return ValidationResult(
            passed=False,
            reason=f"file does not exist: {path}",
            details={"path": path},
        )
    size = os.path.getsize(path)
    if size < min_bytes:
        return ValidationResult(
            passed=False,
            reason=f"file is empty: {path}",
            details={"path": path, "size": size},
        )
    return ValidationResult(
        passed=True,
        reason=f"file exists ({size} bytes)",
        details={"path": path, "size": size},
    )


def validate_file_write(path: str, content: str) -> ValidationResult:
    """Validate a file_write action: check existence + Python syntax if .py."""
    exist_result = validate_file_exists(path)
    if not exist_result.passed:
        return exist_result
    if path.endswith(".py"):
        syn = validate_python_syntax(content)
        if not syn.passed:
            return ValidationResult(
                passed=False,
                reason=f"Python syntax error in {path}: {syn.reason}",
                details=syn.details,
            )
    return ValidationResult(
        passed=True,
        reason=exist_result.reason + (" (syntax OK)" if path.endswith(".py") else ""),
        details=exist_result.details,
    )
