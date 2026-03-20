import re
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from config import MAX_ITERATIONS
from core.llm import llm
from core.metrics import RunMetrics
from core.tools import shell, shell_background, shell_wait, file_op
from core.validator import validate_shell_output, validate_file_write


# ── Prompts ───────────────────────────────────────────────────────────────────

AGENT_INSTRUCTIONS = """
You are an autonomous agent in a multi-agent system.
Respond ONLY with a raw JSON array of action objects. No prose, no markdown, no backticks.

Available actions:
  {"type":"shell","cmd":"<command>"}
  {"type":"shell_background","cmd":"<command>"}
  {"type":"shell_wait","cmd":"<command>","seconds":<N>}
  {"type":"file_read","path":"<path>"}
  {"type":"file_write","path":"<path>","content":"<text>"}
  {"type":"file_list","path":"<directory>"}
  {"type":"message","from":"<your name>","to":"<agent name>","content":"<text>"}
  {"type":"spawn_agent","name":"<unique name>","system_prompt":"<full instructions>"}
  {"type":"learn","fact":"<discovered fact all agents should know>"}
  {"type":"self_evaluate","feedback":"<honest written reflection on this run>","lessons":["<lesson1>","<lesson2>"]}
  {"type":"update_prompt","addon":"<one standing instruction to permanently improve future behavior>"}
  {"type":"done"}

CRITICAL RULES:
1. Output MUST be a single valid JSON array. All actions go inside ONE array.
2. Read ENVIRONMENT SNAPSHOT — use ONLY confirmed available commands.
3. Read RUNTIME LEARNINGS — treat as facts, never repeat a failed approach.
4. Read PERSISTENT MEMORY — use knowledge from previous runs; build on it, never repeat mistakes.
5. NEVER use shell for servers or long processes — use shell_background instead.
6. After shell_background, use shell_wait to verify startup before proceeding.
7. On failure: read the error, adapt, try differently. Never retry the exact same thing.
8. Record discoveries with {"type":"learn"} so all agents benefit.
9. Spawn sub-agents for specialist sub-tasks. Keep agents focused.
10. Before {"type":"done"}, always emit {"type":"self_evaluate"} with honest feedback and concrete lessons.
    NOTE: do NOT include a score — your performance is measured objectively from execution results.
11. emit {"type":"update_prompt"} ONLY for a genuinely new standing instruction; the system will
    deduplicate, rank and prune it automatically — do NOT repeat or contradict existing instructions.
12. {"type":"done"} only when YOUR task is fully complete and verified.
"""

SUPERVISOR_PROMPT = """
You are the Supervisor Agent — the master orchestrator of this multi-agent system.

Responsibilities:
1. Fully analyse the user's goal and the PERSISTENT MEMORY from previous runs.
2. Break the goal into focused sub-tasks; avoid repeating mistakes from past runs.
3. Spawn specialist agents for complex sub-tasks (researcher, coder, tester, etc.).
4. Monitor their progress; reassign tasks if an agent gets stuck.
5. Verify the final result yourself before declaring done.
6. You may also execute simple tasks directly without spawning agents.
7. Before finishing, emit self_evaluate with honest feedback and concrete lessons (no score needed).
8. If you identify a genuinely new standing improvement, emit update_prompt.

Always read the ENVIRONMENT SNAPSHOT and PERSISTENT MEMORY before taking any action.
"""


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    name: str
    system_prompt: str
    memory: list = field(default_factory=list)
    status: str = "idle"
    action_count: int = 0
    current_action: Optional[str] = None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    def __init__(self, event_callback: Optional[Callable] = None):
        self.event_callback = event_callback
        self.agents: dict[str, AgentState] = {}
        self.message_bus: list[dict] = []
        self.learnings: list[str] = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.active_agents: set = set()
        self.completed_agents: set = set()
        self.stats = {"iterations": 0, "total_actions": 0, "start_time": None}
        # Collected from self_evaluate / update_prompt actions during this run
        self.self_evaluations: list[dict] = []
        self.prompt_updates: list[str] = []
        self._memory_context: str = ""
        # Objective metrics — accumulated during _execute calls
        self.run_metrics: RunMetrics = RunMetrics()

    # ── Public API ─────────────────────────────────────────────────────────────

    def stop(self):
        self.stop_event.set()

    def run(self, goal: str, env_snapshot: str, memory_context: str = ""):
        self.stop_event.clear()
        with self.lock:
            self.agents.clear()
            self.message_bus.clear()
            self.learnings.clear()
            self.active_agents.clear()
            self.completed_agents.clear()
            self.self_evaluations.clear()
            self.prompt_updates.clear()
            self._memory_context = memory_context
            self.run_metrics = RunMetrics()
            self.stats = {"iterations": 0, "total_actions": 0, "start_time": time.time()}

        # Register supervisor
        with self.lock:
            self.agents["supervisor"] = AgentState("supervisor", SUPERVISOR_PROMPT)
            self.active_agents.add("supervisor")

        self._emit("run_start", goal=goal)

        for iteration in range(1, MAX_ITERATIONS + 1):
            if self.stop_event.is_set():
                break

            with self.lock:
                self.stats["iterations"] = iteration
                active = list(self.active_agents - self.completed_agents)

            if not active:
                break

            self._emit("iteration", iteration=iteration)

            # Run all active agents in parallel
            max_workers = min(len(active), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._run_agent_iter, name, goal, env_snapshot): name
                    for name in active
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        done = future.result()
                        if done:
                            with self.lock:
                                self.completed_agents.add(name)
                            self._emit("agent_done", agent=name)
                    except Exception as e:
                        self._emit("error", agent=name, message=str(e))

            # Add any newly spawned agents
            with self.lock:
                new = set(self.agents.keys()) - self.active_agents
                self.active_agents |= new
                # Keep agents alive if they have pending messages
                for m in self.message_bus:
                    if m["to"] in self.agents:
                        self.active_agents.add(m["to"])
                        self.completed_agents.discard(m["to"])

            with self.lock:
                remaining = self.active_agents - self.completed_agents
            if not remaining:
                break

        duration = round(time.time() - self.stats["start_time"], 1)
        with self.lock:
            self.run_metrics.total_actions = self.stats["total_actions"]
            self.run_metrics.duration = duration
        self._emit(
            "run_complete",
            success=not self.stop_event.is_set(),
            iterations=self.stats["iterations"],
            actions=self.stats["total_actions"],
            duration=duration,
            metrics=self.run_metrics.to_dict(),
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, **kwargs):
        if self.event_callback:
            try:
                self.event_callback({
                    "type": event_type,
                    "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    **kwargs,
                })
            except Exception:
                pass

    def _run_agent_iter(self, name: str, goal: str, env_snapshot: str) -> bool:
        with self.lock:
            agent = self.agents[name]
            inbox = [m for m in self.message_bus if m["to"] == name]
            self.message_bus[:] = [m for m in self.message_bus if m["to"] != name]

        inbox_text = ""
        if inbox:
            inbox_text = "\n\nInbox:\n" + "\n".join(
                f"[from {m['from']}]: {m['content']}" for m in inbox
            )

        task = (
            goal if name == "supervisor"
            else "Complete your assigned role. Use environment snapshot and runtime learnings. Message supervisor if stuck."
        )

        with self.lock:
            agent.memory.append({"role": "user", "content": f"Task: {task}{inbox_text}"})
            agent.status = "thinking"
            memory_copy = list(agent.memory)
            system = self._build_system(agent.system_prompt, env_snapshot)

        self._emit("agent_update", agent=name, status="thinking")

        response = llm(system, memory_copy, print_fn=lambda *a, **k: None)

        with self.lock:
            agent.memory.append({"role": "assistant", "content": response})

        actions = self._parse(response, name)
        if not actions:
            with self.lock:
                agent.status = "idle"
            self._emit("agent_update", agent=name, status="idle", action_count=agent.action_count)
            return False

        done = False
        results = []

        for action in actions:
            if self.stop_event.is_set():
                break

            atype = action.get("type", "?")
            cmd_str = self._cmd_str(action)

            with self.lock:
                agent.status = "executing"
                agent.current_action = atype
                agent.action_count += 1
                self.stats["total_actions"] += 1

            self._emit("agent_update", agent=name, status="executing",
                       current_action=atype, action_count=agent.action_count)

            result = self._execute(action, name)

            if result == "__DONE__":
                done = True
                self._emit("action_result", agent=name, action_type="done",
                           cmd=cmd_str, result="")
            else:
                results.append(f"[{atype}] → {result}")
                self._emit("action_result", agent=name, action_type=atype,
                           cmd=cmd_str, result=result[:2000])

        context = "\n".join(results)
        if context:
            with self.lock:
                agent.memory.append({"role": "user", "content": f"Results:\n{context}"})

        with self.lock:
            agent.status = "done" if done else "idle"
            agent.current_action = None

        self._emit("agent_update", agent=name,
                   status="done" if done else "idle",
                   action_count=agent.action_count)
        return done

    def _build_system(self, base: str, env: str) -> str:
        with self.lock:
            learnings = list(self.learnings)
            memory_ctx = self._memory_context
        ltext = ""
        if learnings:
            ltext = "\n\nRUNTIME LEARNINGS (treat as facts):\n"
            ltext += "\n".join(f"  - {l}" for l in learnings)
        return base + f"\n\n{env}" + memory_ctx + ltext + "\n\n" + AGENT_INSTRUCTIONS

    def _parse(self, response: str, agent_name: str) -> list[dict]:
        try:
            raw = response.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                end = next((i for i in range(1, len(lines)) if lines[i].strip() == "```"), len(lines))
                raw = "\n".join(lines[1:end])
            if not raw.startswith("["):
                objs = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw, re.DOTALL)
                if objs:
                    raw = "[" + ",".join(objs) + "]"
            actions = json.loads(raw)
            if isinstance(actions, dict):
                actions = [actions]
            return [a for a in actions if isinstance(a, dict)]
        except Exception:
            self._emit("parse_error", agent=agent_name, preview=response[:300])
            return []

    def _cmd_str(self, action: dict) -> str:
        t = action.get("type", "")
        if t in ("shell", "shell_background", "shell_wait"):
            return action.get("cmd", "")
        if t in ("file_read", "file_write", "file_list"):
            return action.get("path", "")
        if t == "message":
            return f"{action.get('from','?')} → {action.get('to','?')}: {action.get('content','')[:80]}"
        if t == "spawn_agent":
            return action.get("name", "")
        if t == "learn":
            return action.get("fact", "")
        if t == "self_evaluate":
            return action.get("feedback", "")[:80]
        if t == "update_prompt":
            return action.get("addon", "")[:80]
        if t == "done":
            return "Task complete"
        return ""

    def _execute(self, action: dict, agent_name: str) -> str:
        t = action.get("type", "")
        if t == "shell":
            result = shell(action["cmd"])
            low = result.lower()
            if any(x in low for x in ["was not found", "is not recognized", "no such file", "command not found"]):
                fact = f"command '{action['cmd'].strip().split()[0]}' is NOT available"
                with self.lock:
                    if fact not in self.learnings:
                        self.learnings.append(fact)
                self._emit("learning", fact=fact)
            # ── Validator ──────────────────────────────────────────────────
            vr = validate_shell_output(action["cmd"], result)
            with self.lock:
                self.run_metrics.shell_calls += 1
                if vr.passed:
                    self.run_metrics.shell_passed += 1
                else:
                    self.run_metrics.shell_failed += 1
            self._emit("validation_result", agent=agent_name, action_type="shell",
                       passed=vr.passed, reason=vr.reason)
            return result + "\n" + vr.as_context()
        if t == "shell_background":
            return shell_background(action["cmd"])
        if t == "shell_wait":
            return shell_wait(action["cmd"], int(action.get("seconds", 3)))
        if t == "file_read":
            return file_op("read", action["path"])
        if t == "file_write":
            raw = file_op("write", action["path"], action.get("content", ""))
            # ── Validator ──────────────────────────────────────────────────
            vr = validate_file_write(action["path"], action.get("content", ""))
            with self.lock:
                self.run_metrics.file_writes += 1
                if vr.passed:
                    self.run_metrics.files_validated += 1
                else:
                    self.run_metrics.validation_errors += 1
            self._emit("validation_result", agent=agent_name, action_type="file_write",
                       passed=vr.passed, reason=vr.reason, path=action["path"])
            return raw + "\n" + vr.as_context()
        if t == "file_list":
            return file_op("list", action.get("path", "."))
        if t == "message":
            with self.lock:
                self.message_bus.append({
                    "from": action.get("from", agent_name),
                    "to": action["to"],
                    "content": action["content"],
                })
            return f"[message queued → {action['to']}]"
        if t == "spawn_agent":
            name = action["name"]
            with self.lock:
                if name not in self.agents:
                    self.agents[name] = AgentState(name, action["system_prompt"])
            self._emit("agent_spawn", agent=name, spawned_by=agent_name)
            return f"[agent spawned: {name}]"
        if t == "learn":
            fact = action.get("fact", "")
            if fact:
                with self.lock:
                    if fact not in self.learnings:
                        self.learnings.append(fact)
                self._emit("learning", fact=fact)
            return "[learning recorded]"
        if t == "self_evaluate":
            # Score is intentionally ignored here — objective metrics drive scoring
            feedback = action.get("feedback", "")
            lessons = action.get("lessons", [])
            eval_data = {"feedback": feedback, "lessons": lessons}
            with self.lock:
                self.self_evaluations.append(eval_data)
            self._emit("self_evaluate", agent=agent_name,
                       feedback=feedback, lessons=lessons)
            return "[self-evaluation recorded]"
        if t == "update_prompt":
            addon = action.get("addon", "").strip()
            if addon:
                with self.lock:
                    self.prompt_updates.append(addon)
                self._emit("prompt_update", agent=agent_name, addon=addon)
            return "[prompt update recorded]"
        if t == "done":
            return "__DONE__"
        return f"unknown action: {t}"
