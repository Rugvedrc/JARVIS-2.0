"""Persistent cross-run memory for the JARVIS self-improvement loop.

State is stored as JSON in `jarvis_memory.json` (next to the project root).
Each run appends a RunRecord; the system prompt addon and global learnings
accumulate over time, giving the agent genuine RL-style continuity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

MEMORY_FILE = "jarvis_memory.json"

MAX_RUNS_IN_CONTEXT = 5  # how many recent runs to include in the prompt


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    run_id: int
    timestamp: str
    goal: str
    iterations: int
    total_actions: int
    duration: float
    success: bool
    self_score: Optional[float] = None   # 0.0–10.0, from the agent's self_evaluate
    self_feedback: Optional[str] = None  # agent's written critique
    lessons: list = field(default_factory=list)


@dataclass
class PersistentMemory:
    run_count: int = 0
    runs: list = field(default_factory=list)           # list[dict] (RunRecord dicts)
    global_learnings: list = field(default_factory=list)  # accumulated facts
    performance_scores: list = field(default_factory=list)  # float per run
    system_prompt_addon: str = ""  # evolves via update_prompt actions
    skill_profile: dict = field(default_factory=dict)  # area -> avg score

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PersistentMemory":
        mem = cls()
        mem.run_count = d.get("run_count", 0)
        mem.runs = d.get("runs", [])
        mem.global_learnings = d.get("global_learnings", [])
        mem.performance_scores = d.get("performance_scores", [])
        mem.system_prompt_addon = d.get("system_prompt_addon", "")
        mem.skill_profile = d.get("skill_profile", {})
        return mem

    # ── Helpers ───────────────────────────────────────────────────────────────

    def add_run(self, record: RunRecord) -> None:
        self.run_count += 1
        self.runs.append(asdict(record))
        if record.self_score is not None:
            self.performance_scores.append(record.self_score)
        if record.lessons:
            for lesson in record.lessons:
                if lesson and lesson not in self.global_learnings:
                    self.global_learnings.append(lesson)

    def apply_prompt_update(self, addon: str) -> None:
        """Append a unique instruction to the evolving system prompt addon."""
        addon = addon.strip()
        if addon and addon not in self.system_prompt_addon:
            if self.system_prompt_addon:
                self.system_prompt_addon += "\n" + addon
            else:
                self.system_prompt_addon = addon

    def average_score(self) -> Optional[float]:
        if not self.performance_scores:
            return None
        return round(sum(self.performance_scores) / len(self.performance_scores), 2)

    def recent_trend(self) -> str:
        scores = self.performance_scores[-MAX_RUNS_IN_CONTEXT:]
        if not scores:
            return "no data"
        avg = sum(scores) / len(scores)
        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            direction = "↑ improving" if delta > 0 else ("↓ declining" if delta < 0 else "→ stable")
        else:
            direction = "→ baseline"
        return f"{avg:.1f}/10 ({direction})"

    def build_memory_context(self) -> str:
        """Build a text block injected into every agent's system prompt."""
        lines = ["\n\n=== PERSISTENT MEMORY (from previous runs) ==="]
        lines.append(f"Total runs completed: {self.run_count}")
        lines.append(f"Performance trend (last {MAX_RUNS_IN_CONTEXT}): {self.recent_trend()}")

        if self.global_learnings:
            lines.append("\nGlobal learnings (accumulated across all runs):")
            for gl in self.global_learnings[-20:]:  # cap at 20
                lines.append(f"  • {gl}")

        recent = self.runs[-MAX_RUNS_IN_CONTEXT:]
        if recent:
            lines.append("\nRecent run summaries:")
            for r in recent:
                score = r.get("self_score")
                score_str = f", score={score}/10" if score is not None else ""
                lines.append(
                    f"  Run #{r['run_id']} | goal: {r['goal'][:80]} | "
                    f"{'✓' if r['success'] else '✗'}{score_str}"
                )
                if r.get("self_feedback"):
                    lines.append(f"    feedback: {r['self_feedback'][:200]}")
                if r.get("lessons"):
                    for ls in r["lessons"][:3]:
                        lines.append(f"    lesson: {ls}")

        if self.system_prompt_addon:
            lines.append("\nEVOLVED INSTRUCTIONS (self-generated, follow strictly):")
            lines.append(self.system_prompt_addon)

        lines.append("=== END PERSISTENT MEMORY ===")
        return "\n".join(lines)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_memory(path: str = MEMORY_FILE) -> PersistentMemory:
    """Load memory from disk; return a fresh instance if file is missing/corrupt."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return PersistentMemory.from_dict(json.load(f))
        except Exception:
            pass
    return PersistentMemory()


def save_memory(mem: PersistentMemory, path: str = MEMORY_FILE) -> None:
    """Persist memory to disk atomically via a temp file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mem.to_dict(), f, indent=2)
    os.replace(tmp, path)
