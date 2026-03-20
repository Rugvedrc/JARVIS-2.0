"""Persistent cross-run memory for the JARVIS self-improvement loop.

State is stored as JSON in `jarvis_memory.json` (next to the project root).
Each run appends a RunRecord; the system prompt addon and global learnings
accumulate over time, giving the agent genuine RL-style continuity.

Prompt instructions are ranked (scored), deduplicated, and pruned instead of
being blindly appended — preventing unbounded prompt bloat and contradiction.
"""

from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

MEMORY_FILE = "jarvis_memory.json"

MAX_RUNS_IN_CONTEXT = 5       # how many recent runs to include in the prompt
PROMPT_MAX_INSTRUCTIONS = 15  # hard cap on stored instructions
PROMPT_MIN_SCORE = 0.4        # instructions below this are pruned
PROMPT_JACCARD_THRESHOLD = 0.50  # Jaccard similarity for "same topic" detection


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
    metrics: dict = field(default_factory=dict)  # RunMetrics.to_dict()
    objective_score: Optional[float] = None      # derived from RunMetrics.compute_score()
    self_feedback: Optional[str] = None          # agent's written reflection (kept)
    lessons: list = field(default_factory=list)


@dataclass
class PersistentMemory:
    run_count: int = 0
    runs: list = field(default_factory=list)                # list[dict] (RunRecord dicts)
    global_learnings: list = field(default_factory=list)    # accumulated facts
    performance_scores: list = field(default_factory=list)  # objective score per run
    # Ranked prompt instructions: list of {text, score, run_id}
    prompt_instructions: list = field(default_factory=list)
    system_prompt_addon: str = ""   # compiled from prompt_instructions (cache)
    skill_profile: dict = field(default_factory=dict)       # reserved for future use

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
        mem.prompt_instructions = d.get("prompt_instructions", [])
        mem.skill_profile = d.get("skill_profile", {})
        # Recompile the addon string in case prompt_instructions were loaded
        mem._recompile_addon()
        # Back-compat: if old JSON has system_prompt_addon but no instructions, migrate
        if not mem.prompt_instructions and d.get("system_prompt_addon"):
            for line in d["system_prompt_addon"].splitlines():
                line = line.strip()
                if line:
                    mem.prompt_instructions.append({"text": line, "score": 1.0, "run_id": 0})
            mem._recompile_addon()
        return mem

    # ── Prompt instruction management ─────────────────────────────────────────

    def apply_prompt_update(self, addon: str, run_id: int = 0) -> str:
        """Rank, replace, or prune prompt instructions.

        Algorithm:
          1. Compute Jaccard word-overlap with every existing instruction.
          2. If overlap >= threshold → it's the "same topic":
               - Identical text: reinforce (score += 0.5)
               - Different text: replace if new is longer/more detailed, then score up
          3. Otherwise: add as a new instruction with score 1.0.
          4. Prune: drop below PROMPT_MIN_SCORE, then cap at PROMPT_MAX_INSTRUCTIONS.

        Returns a short string describing the action taken: 'added', 'reinforced',
        'replaced', or 'skipped'.
        """
        addon = addon.strip()
        if not addon:
            return "skipped"

        words_new = set(addon.lower().split())

        best_idx: Optional[int] = None
        best_overlap = 0.0
        for i, instr in enumerate(self.prompt_instructions):
            words_old = set(instr["text"].lower().split())
            union = words_new | words_old
            if not union:
                continue
            overlap = len(words_new & words_old) / len(union)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx is not None and best_overlap >= PROMPT_JACCARD_THRESHOLD:
            existing = self.prompt_instructions[best_idx]
            if addon == existing["text"]:
                # Exact duplicate — reinforce
                existing["score"] = round(existing["score"] + 0.5, 2)
                action = "reinforced"
            else:
                # Same topic — replace with the longer/newer version, boost score
                existing["score"] = round(existing["score"] + 0.3, 2)
                if len(addon) >= len(existing["text"]) or existing["score"] < 1.0:
                    existing["text"] = addon
                    existing["run_id"] = run_id
                action = "replaced"
        else:
            self.prompt_instructions.append({"text": addon, "score": 1.0, "run_id": run_id})
            action = "added"

        self._prune()
        self._recompile_addon()
        return action

    def _prune(self) -> None:
        """Drop low-score instructions and enforce the hard cap."""
        self.prompt_instructions = [
            i for i in self.prompt_instructions if i["score"] >= PROMPT_MIN_SCORE
        ]
        if len(self.prompt_instructions) > PROMPT_MAX_INSTRUCTIONS:
            self.prompt_instructions = heapq.nlargest(
                PROMPT_MAX_INSTRUCTIONS,
                self.prompt_instructions,
                key=lambda i: i["score"],
            )

    def _recompile_addon(self) -> None:
        """Re-derive the plain-text addon string from ranked instructions."""
        ordered = sorted(self.prompt_instructions, key=lambda i: i["score"], reverse=True)
        self.system_prompt_addon = "\n".join(
            i["text"] for i in ordered if i["score"] >= PROMPT_MIN_SCORE
        )

    # ── Run tracking ──────────────────────────────────────────────────────────

    def add_run(self, record: RunRecord) -> None:
        self.run_count += 1
        self.runs.append(asdict(record))
        if record.objective_score is not None:
            self.performance_scores.append(record.objective_score)
        if record.lessons:
            for lesson in record.lessons:
                if lesson and lesson not in self.global_learnings:
                    self.global_learnings.append(lesson)

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
        return f"{avg:.1f}/10 ({direction}, objective)"

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
                score = r.get("objective_score")
                score_str = f", objective_score={score}/10" if score is not None else ""
                m = r.get("metrics", {})
                pass_rate = ""
                if m.get("shell_calls", 0):
                    pct = round(100 * m.get("shell_passed", 0) / m["shell_calls"])
                    pass_rate = f", shell_pass={pct}%"
                lines.append(
                    f"  Run #{r['run_id']} | goal: {r['goal'][:80]} | "
                    f"{'✓' if r['success'] else '✗'}{score_str}{pass_rate}"
                )
                if r.get("self_feedback"):
                    lines.append(f"    feedback: {r['self_feedback'][:200]}")
                if r.get("lessons"):
                    for ls in r["lessons"][:3]:
                        lines.append(f"    lesson: {ls}")

        if self.prompt_instructions:
            lines.append("\nEVOLVED INSTRUCTIONS (ranked, follow strictly):")
            ordered = sorted(self.prompt_instructions, key=lambda i: i["score"], reverse=True)
            for instr in ordered[:PROMPT_MAX_INSTRUCTIONS]:
                lines.append(f"  [{instr['score']:.1f}] {instr['text']}")

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
