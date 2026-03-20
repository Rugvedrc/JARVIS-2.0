"""CLI entry point — runs a single goal without the web UI."""
import sys
from core.environment import discover_environment
from core.orchestrator import MultiAgentOrchestrator


def cli_callback(event: dict):
    t = event.get("type", "")
    ts = event.get("ts", "")
    agent = event.get("agent", "")

    if t == "run_start":
        print(f"\n{'='*60}\nGOAL: {event['goal']}\n{'='*60}")
    elif t == "iteration":
        print(f"\n─── Iteration {event['iteration']} ───")
    elif t == "agent_update":
        print(f"  [{agent}] {event['status']}", end="")
        if event.get("current_action"):
            print(f" ({event['current_action']})", end="")
        print()
    elif t == "action_result":
        cmd = event.get("cmd", "")
        result = event.get("result", "")
        print(f"  [{agent}:{event['action_type']}] {cmd[:80]}")
        if result:
            print(f"    → {result[:300]}")
    elif t == "learning":
        print(f"  [LEARNED] {event['fact']}")
    elif t == "agent_spawn":
        print(f"  [SPAWN] {event['agent']} (by {event.get('spawned_by','?')})")
    elif t == "agent_done":
        print(f"  ✓ {agent} done")
    elif t == "run_complete":
        status = "✓ Complete" if event["success"] else "■ Stopped"
        print(f"\n{'='*60}")
        print(f"{status} — {event['iterations']} iters, {event['actions']} actions, {event['duration']}s")
        print(f"{'='*60}\n")
    elif t == "error":
        print(f"  [ERROR:{agent}] {event.get('message','')}")
    elif t == "parse_error":
        print(f"  [PARSE ERROR:{agent}] {event.get('preview','')[:200]}")


def main():
    goal = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("Goal: ").strip()
    if not goal:
        print("No goal provided.")
        sys.exit(1)

    print("[discovering environment...]")
    env = discover_environment()
    print(env)

    orch = MultiAgentOrchestrator(event_callback=cli_callback)
    orch.run(goal, env)


if __name__ == "__main__":
    main()
