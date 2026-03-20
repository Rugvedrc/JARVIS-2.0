"""Objective, measurable run metrics that replace LLM self-scoring.

Instead of asking the model to grade itself (which is biased and unreliable),
we track hard counters during execution and derive a score from them.

Metrics tracked per run:
  shell_calls       — total shell commands executed
  shell_passed      — commands whose output contained no failure signal
  shell_failed      — commands whose output contained a failure signal
  file_writes       — file_write actions executed
  files_validated   — written files that were subsequently verified to exist
  validation_errors — files that failed post-write validation
  total_actions     — all orchestrator action executions
  duration          — wall-clock seconds for the full run
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RunMetrics:
    shell_calls: int = 0
    shell_passed: int = 0
    shell_failed: int = 0
    file_writes: int = 0
    files_validated: int = 0
    validation_errors: int = 0
    total_actions: int = 0
    duration: float = 0.0

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def pass_rate(self) -> float:
        """Fraction of shell commands that passed (0.0–1.0)."""
        total = self.shell_passed + self.shell_failed
        return self.shell_passed / total if total else 1.0  # no calls ⇒ not penalised

    @property
    def file_validation_rate(self) -> float:
        """Fraction of written files that were validated successfully."""
        return self.files_validated / self.file_writes if self.file_writes else 1.0

    def compute_score(self) -> float:
        """Objective score 0–10 derived purely from measurable outcomes.

        Formula (all terms 0–10, then weighted average):
          pass_rate_score      = pass_rate × 10          (weight 0.5)
          file_val_score       = file_validation_rate × 10 (weight 0.3)
          action_penalty       = max(0, 1 - validation_errors/5) × 10 (weight 0.2)
        """
        pass_score = self.pass_rate * 10
        file_score = self.file_validation_rate * 10
        # Each validation error costs 2 points (capped at 10 errors → 0 score)
        error_score = max(0.0, 10.0 - self.validation_errors * 2)

        score = pass_score * 0.5 + file_score * 0.3 + error_score * 0.2
        return round(min(10.0, max(0.0, score)), 2)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunMetrics":
        return cls(
            shell_calls=d.get("shell_calls", 0),
            shell_passed=d.get("shell_passed", 0),
            shell_failed=d.get("shell_failed", 0),
            file_writes=d.get("file_writes", 0),
            files_validated=d.get("files_validated", 0),
            validation_errors=d.get("validation_errors", 0),
            total_actions=d.get("total_actions", 0),
            duration=d.get("duration", 0.0),
        )

    def summary_str(self) -> str:
        """One-line human-readable summary."""
        return (
            f"shell {self.shell_passed}✓/{self.shell_failed}✗  "
            f"files {self.files_validated}/{self.file_writes}  "
            f"val_errors {self.validation_errors}  "
            f"score {self.compute_score():.1f}/10"
        )
