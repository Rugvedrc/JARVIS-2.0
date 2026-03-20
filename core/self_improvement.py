"""Self-improvement (RL-like) loop for JARVIS.

Each cycle:
  1. Load persistent memory (what was learned in all previous runs).
  2. Generate a goal (either provided externally or auto-generated via LLM).
  3. Run the multi-agent orchestrator with memory context injected into prompts.
  4. Persist the run record, self-evaluation, and prompt updates.
  5. (Repeat N times.)

The system never forgets between runs: global learnings, evolved system prompt
addons, and run summaries are stored in jarvis_memory.json.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, Callable

from core.environment import discover_environment
from core.llm import llm
from core.memory import (
    PersistentMemory,
    RunRecord,
    load_memory,
    save_memory,
    MEMORY_FILE,
)
from core.metrics import RunMetrics
from core.orchestrator import MultiAgentOrchestrator


# ── Goal-generator prompt ─────────────────────────────────────────────────────

GOAL_GENERATOR_SYSTEM = """
You are a goal generator for an autonomous AI agent.
Your job: propose ONE concrete, achievable task for the agent to practise on.

Rules:
- The task must be self-contained and testable (the agent can verify success itself).
- Prefer tasks that build on previous learnings and stretch the agent a little further.
- Tasks should be practical: writing & running code, file manipulation, web queries, etc.
- Keep the goal under 120 characters.
- Respond with ONLY the goal string — no explanations, no bullets, no JSON.
"""


FALLBACK_GOAL = (
    "Write a Python script that lists all .py files in the current directory "
    "and counts lines in each, then run it."
)


def _generate_goal(memory: PersistentMemory) -> str:
    """Ask the LLM to generate an appropriate next goal based on run history."""
    history_lines = []
    for r in memory.runs[-5:]:
        score = r.get("objective_score")
        score_str = f" [objective_score {score}/10]" if score is not None else ""
        history_lines.append(f"  • {r['goal'][:100]}{score_str}")

    history_text = (
        "\n".join(history_lines)
        if history_lines
        else "  (no previous runs — this is the first run)"
    )

    learnings_text = ""
    if memory.global_learnings:
        learnings_text = "\nKnown facts about the environment:\n" + "\n".join(
            f"  • {l}" for l in memory.global_learnings[-10:]
        )

    user_msg = (
        f"Previous tasks attempted:\n{history_text}\n"
        f"{learnings_text}\n"
        "Generate one new task that is slightly more challenging than the previous ones."
    )

    response = llm(
        GOAL_GENERATOR_SYSTEM,
        [{"role": "user", "content": user_msg}],
        print_fn=lambda *a, **kw: None,
    )
    goal = response.strip().strip('"').strip("'")
    if not goal or len(goal) < 5:
        goal = FALLBACK_GOAL
    return goal[:200]


# ── Self-improvement loop ─────────────────────────────────────────────────────

class SelfImprovementLoop:
    """Orchestrates N self-improvement cycles with persistent memory."""

    def __init__(
        self,
        memory_path: str = MEMORY_FILE,
        event_callback: Optional[Callable] = None,
    ):
        self.memory_path = memory_path
        self.event_callback = event_callback
        self.memory: PersistentMemory = load_memory(memory_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_cycle(self, goal: Optional[str] = None) -> dict:
        """Run a single self-improvement cycle and return a summary dict."""
        self.memory = load_memory(self.memory_path)  # reload in case of concurrent writes

        if goal is None:
            self._notify("goal_gen_start", message="Generating goal from memory…")
            goal = _generate_goal(self.memory)
            self._notify("goal_generated", goal=goal)

        run_id = self.memory.run_count + 1
        self._notify("cycle_start", run_id=run_id, goal=goal)

        # Build memory context to inject into all agent prompts
        memory_context = self.memory.build_memory_context()

        # Discover the live environment
        env_snapshot = discover_environment()

        # Create orchestrator
        orch = MultiAgentOrchestrator(event_callback=self.event_callback)

        start = time.time()
        orch.run(goal, env_snapshot, memory_context=memory_context)
        duration = round(time.time() - start, 1)

        # ── Objective metrics (replaces self-score) ───────────────────────────
        metrics: RunMetrics = orch.run_metrics
        metrics.duration = duration
        objective_score = metrics.compute_score()

        # ── Aggregate self-evaluations (feedback + lessons only) ──────────────
        evals = orch.self_evaluations
        combined_feedback = " | ".join(
            e["feedback"] for e in evals if e.get("feedback")
        ) or None
        all_lessons: list[str] = []
        for e in evals:
            all_lessons.extend(e.get("lessons", []))

        # ── Build run record ──────────────────────────────────────────────────
        record = RunRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            goal=goal,
            iterations=orch.stats["iterations"],
            total_actions=orch.stats["total_actions"],
            duration=duration,
            success=not orch.stop_event.is_set(),
            metrics=metrics.to_dict(),
            objective_score=objective_score,
            self_feedback=combined_feedback,
            lessons=all_lessons,
        )

        # ── Update memory ─────────────────────────────────────────────────────
        self.memory.add_run(record)

        # Apply prompt updates with ranking/dedup/pruning
        prompt_actions: list[str] = []
        for addon in orch.prompt_updates:
            action_taken = self.memory.apply_prompt_update(addon, run_id=run_id)
            prompt_actions.append(f"{action_taken}: {addon[:60]}")

        # Persist
        save_memory(self.memory, self.memory_path)

        summary = {
            "run_id": run_id,
            "goal": goal,
            "success": record.success,
            "objective_score": objective_score,
            "metrics": metrics.to_dict(),
            "metrics_summary": metrics.summary_str(),
            "feedback": combined_feedback,
            "lessons": all_lessons,
            "iterations": record.iterations,
            "total_actions": record.total_actions,
            "duration": duration,
            "prompt_updates": orch.prompt_updates,
            "prompt_actions": prompt_actions,
            "avg_score_all_time": self.memory.average_score(),
            "trend": self.memory.recent_trend(),
            "prompt_instructions_count": len(self.memory.prompt_instructions),
        }

        self._notify("cycle_complete", **summary)
        return summary

    def run_n_cycles(
        self,
        n: int = 5,
        goals: Optional[list[str]] = None,
    ) -> list[dict]:
        """Run N cycles, optionally with pre-specified goals.  Returns all summaries."""
        summaries = []
        for i in range(n):
            goal = goals[i] if (goals and i < len(goals)) else None
            summary = self.run_cycle(goal=goal)
            summaries.append(summary)
        return summaries

    # ── Internal ───────────────────────────────────────────────────────────────

    def _notify(self, event_type: str, **kwargs):
        if self.event_callback:
            try:
                self.event_callback({"type": event_type, **kwargs})
            except Exception:
                pass
