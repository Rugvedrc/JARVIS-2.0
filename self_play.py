#!/usr/bin/env python3
"""self_play.py — Run N self-improvement cycles and print a live report.

Usage:
    python self_play.py                   # auto-generate goals, 5 cycles
    python self_play.py --cycles 3        # 3 auto-generated cycles
    python self_play.py --goal "..."      # same goal repeated N times
    python self_play.py --reset           # wipe memory and start fresh
    python self_play.py --show-memory     # print current memory state and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from core.memory import load_memory, save_memory, PersistentMemory, MEMORY_FILE
from core.self_improvement import SelfImprovementLoop


# ── Pretty-printing helpers ───────────────────────────────────────────────────

CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 10 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.1f}/10"


def _event_callback(event: dict):
    t = event.get("type", "")

    if t == "cycle_start":
        print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
        print(f"{BOLD}{CYAN}  RUN #{event['run_id']} — {event['goal']}{RESET}")
        print(f"{BOLD}{CYAN}{'═'*70}{RESET}")

    elif t == "goal_gen_start":
        print(f"\n{YELLOW}  ⚙  {event.get('message','Generating goal…')}{RESET}")

    elif t == "goal_generated":
        print(f"  {GREEN}Goal:{RESET} {event['goal']}")

    elif t == "run_start":
        print(f"\n  {BOLD}▶ Starting orchestrator — goal: {event['goal'][:80]}{RESET}")

    elif t == "iteration":
        print(f"\n  ─── Iteration {event['iteration']} ───")

    elif t == "agent_update":
        status = event.get("status", "")
        action = event.get("current_action", "")
        suffix = f" ({action})" if action else ""
        print(f"    [{event.get('agent','?')}] {status}{suffix}")

    elif t == "action_result":
        cmd = event.get("cmd", "")[:60]
        result = (event.get("result", "") or "")[:200]
        print(f"    [{event.get('agent','?')}:{event.get('action_type','?')}] {cmd}")
        if result:
            print(f"      → {result}")

    elif t == "learning":
        print(f"  {YELLOW}  📚 LEARNED: {event['fact']}{RESET}")

    elif t == "self_evaluate":
        score = event.get("score", 0)
        feedback = event.get("feedback", "")
        lessons = event.get("lessons", [])
        bar = _bar(score)
        colour = GREEN if score >= 7 else (YELLOW if score >= 4 else RED)
        print(f"\n  {colour}  🧠 SELF-EVAL {bar}{RESET}")
        if feedback:
            print(f"     feedback: {feedback[:200]}")
        for ls in lessons[:3]:
            print(f"     lesson  : {ls}")

    elif t == "prompt_update":
        print(f"  {CYAN}  📝 PROMPT UPDATE: {event.get('addon','')[:120]}{RESET}")

    elif t == "agent_spawn":
        print(f"  {CYAN}  ↳ SPAWN {event['agent']} (by {event.get('spawned_by','?')}){RESET}")

    elif t == "agent_done":
        print(f"  {GREEN}  ✓ {event.get('agent','?')} done{RESET}")

    elif t == "run_complete":
        status = f"{GREEN}✓ Complete{RESET}" if event["success"] else f"{RED}■ Stopped{RESET}"
        print(
            f"\n  {status} — {event['iterations']} iters, "
            f"{event['actions']} actions, {event['duration']}s"
        )

    elif t == "error":
        print(f"  {RED}  [ERROR:{event.get('agent','?')}] {event.get('message','')}{RESET}")

    elif t == "parse_error":
        print(f"  {RED}  [PARSE ERROR] {event.get('preview','')[:200]}{RESET}")

    elif t == "cycle_complete":
        score = event.get("score")
        bar_str = _bar(score) if score is not None else "N/A"
        colour = GREEN if (score or 0) >= 7 else (YELLOW if (score or 0) >= 4 else RED)
        print(f"\n{BOLD}  ── Cycle #{event.get('run_id')} complete ──{RESET}")
        print(f"  Score   : {colour}{bar_str}{RESET}")
        print(f"  Trend   : {event.get('trend','?')}")
        if event.get("prompt_updates"):
            print(f"  Prompt updates this run: {len(event['prompt_updates'])}")
        print()


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_final_summary(summaries: list[dict], memory: PersistentMemory):
    print(f"\n{BOLD}{'═'*70}")
    print("  SELF-PLAY COMPLETE — SUMMARY")
    print(f"{'═'*70}{RESET}")
    print(f"  {'RUN':<5} {'SCORE':>7}  {'ITERS':>5}  {'ACTIONS':>7}  {'SEC':>6}  GOAL")
    print(f"  {'─'*5} {'─'*7}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*40}")
    for s in summaries:
        score_str = f"{s['score']:.1f}" if s.get("score") is not None else "  N/A"
        ok = "✓" if s["success"] else "✗"
        goal_short = s["goal"][:40]
        print(
            f"  {ok} #{s['run_id']:<3} {score_str:>7}  {s['iterations']:>5}  "
            f"{s['total_actions']:>7}  {s['duration']:>6}s  {goal_short}"
        )

    print()
    print(f"  All-time avg score : {memory.average_score() or 'N/A'}")
    print(f"  Performance trend  : {memory.recent_trend()}")
    print(f"  Total runs in memory: {memory.run_count}")
    print(f"  Global learnings   : {len(memory.global_learnings)}")
    if memory.system_prompt_addon:
        print(f"\n  {BOLD}Evolved system prompt addon:{RESET}")
        for line in memory.system_prompt_addon.splitlines():
            print(f"    {line}")
    print(f"{BOLD}{'═'*70}{RESET}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run JARVIS self-improvement cycles (RL-like loop)."
    )
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of self-improvement cycles to run (default: 5)")
    parser.add_argument("--goal", type=str, default=None,
                        help="Fixed goal to use for every cycle (omit to auto-generate)")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe persistent memory before starting")
    parser.add_argument("--show-memory", action="store_true",
                        help="Print current memory state and exit")
    parser.add_argument("--memory-file", type=str, default=MEMORY_FILE,
                        help=f"Path to memory JSON (default: {MEMORY_FILE})")
    args = parser.parse_args()

    if args.reset:
        mem = PersistentMemory()
        save_memory(mem, args.memory_file)
        print(f"{GREEN}Memory reset.{RESET}")

    if args.show_memory:
        mem = load_memory(args.memory_file)
        print(json.dumps(mem.to_dict(), indent=2))
        return

    print(f"{BOLD}JARVIS Self-Play — {args.cycles} cycle(s){RESET}")
    print(f"Memory file: {args.memory_file}")
    print(f"Started    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    loop = SelfImprovementLoop(
        memory_path=args.memory_file,
        event_callback=_event_callback,
    )

    goals = [args.goal] * args.cycles if args.goal else None
    summaries = loop.run_n_cycles(n=args.cycles, goals=goals)

    _print_final_summary(summaries, loop.memory)


if __name__ == "__main__":
    main()
