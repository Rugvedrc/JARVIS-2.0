import os
import json
import subprocess
import requests
import platform
import time

NVIDIA_API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "mistralai/mistral-small-4-119b-2603"
MAX_ITERATIONS = 30

# ── Shared state ──────────────────────────────────────────────────────────────
message_bus: list[dict] = []
agent_registry: dict[str, dict] = {}
runtime_learnings: list[str] = []


# ── Environment discovery ─────────────────────────────────────────────────────
def shell_raw(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""


def discover_environment() -> str:
    results = {}
    results["os"] = f"{platform.system()} {platform.release()}"
    results["cwd"] = os.getcwd()
    results["files_in_cwd"] = shell_raw("dir /b" if os.name == "nt" else "ls")

    for cmd in ["py --version", "python --version", "python3 --version"]:
        out = shell_raw(cmd)
        key = cmd.split()[0]
        results[f"cmd_{key}"] = out if out and "not found" not in out.lower() and "error" not in out.lower() else "NOT AVAILABLE"

    python_cmd = None
    for cmd in ["py", "python", "python3"]:
        out = shell_raw(f"{cmd} --version")
        if out and "Python" in out:
            python_cmd = cmd
            break
    results["python_command"] = python_cmd if python_cmd else "NONE FOUND"
    results["shell"] = "PowerShell/cmd (Windows)" if os.name == "nt" else "bash (Unix)"
    results["command_chaining"] = "use ';' not '&&'" if os.name == "nt" else "use '&&' or ';'"
    results["background_process"] = "use shell_background action for servers and long-running processes"

    for tool in ["node", "npm", "git", "pip", "curl"]:
        out = shell_raw(f"{tool} --version")
        results[tool] = out if out and "not found" not in out.lower() else "NOT AVAILABLE"

    lines = ["=== ENVIRONMENT (verified by running real commands) ==="]
    for k, v in results.items():
        lines.append(f"  {k}: {v}")
    lines.append("=== END ENVIRONMENT ===")
    return "\n".join(lines)


# ── LLM call (streaming) ──────────────────────────────────────────────────────
def llm(system: str, messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.2,
        "max_tokens": 4096,  # reduced from 4096
        "stream": True,
    }
    for attempt in range(3):
        try:
            r = requests.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,  # reduced from 120
                stream=True,
            )
            r.raise_for_status()

            result = []
            print("  [llm thinking", end="", flush=True)
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            result.append(delta)
                            print(".", end="", flush=True)
                    except Exception:
                        pass
            print("]")
            return "".join(result)

        except Exception as e:
            print(f"\n  [llm error attempt {attempt+1}/3]: {e}")
            time.sleep(3)

    return "[]"  # return empty action list on total failure


# ── Primitives ────────────────────────────────────────────────────────────────
def shell(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        out = (result.stdout + result.stderr).strip()
        out = out[:8000] if out else "(no output, exit code 0 — success)"

        low = out.lower()
        if any(x in low for x in ["was not found", "is not recognized", "no such file", "command not found"]):
            cmd_used = cmd.strip().split()[0]
            learning = f"LEARNED: command '{cmd_used}' does not work on this system. Try an alternative."
            if learning not in runtime_learnings:
                runtime_learnings.append(learning)
                print(f"  [learning recorded] {learning}")

        return out
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30s — if running a server, use shell_background instead"
    except Exception as e:
        return f"ERROR: {e}"


def shell_background(cmd: str) -> str:
    """Start a long-running process in background. Returns immediately."""
    try:
        if os.name == "nt":
            # Windows: use Popen with CREATE_NEW_CONSOLE flag equivalent
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        return f"started in background: {cmd} (use shell with curl/Invoke-WebRequest to verify)"
    except Exception as e:
        return f"ERROR starting background process: {e}"


def shell_wait(cmd: str, seconds: int) -> str:
    """Run a command, wait N seconds, return output. For startup delays."""
    try:
        time.sleep(seconds)
        return shell(cmd)
    except Exception as e:
        return f"ERROR: {e}"


def file_op(op: str, path: str, content: str = "") -> str:
    try:
        if op == "read":
            with open(path, "r", errors="replace") as f:
                return f.read()
        elif op == "write":
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"written: {path}"
        elif op == "list":
            entries = []
            for root, dirs, files in os.walk(path or "."):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for file in files:
                    entries.append(os.path.join(root, file))
            return "\n".join(entries[:200])
    except Exception as e:
        return f"ERROR: {e}"


# ── Action executor ───────────────────────────────────────────────────────────
def execute_action(action: dict, agent_name: str = "") -> str:
    t = action.get("type", "")
    if t == "shell":
        return shell(action["cmd"])
    elif t == "shell_background":
        return shell_background(action["cmd"])
    elif t == "shell_wait":
        return shell_wait(action["cmd"], int(action.get("seconds", 3)))
    elif t == "file_read":
        return file_op("read", action["path"])
    elif t == "file_write":
        return file_op("write", action["path"], action.get("content", ""))
    elif t == "file_list":
        return file_op("list", action.get("path", "."))
    elif t == "message":
        message_bus.append({
            "from": action["from"],
            "to": action["to"],
            "content": action["content"],
        })
        return f"[message queued → {action['to']}]"
    elif t == "spawn_agent":
        name = action["name"]
        agent_registry[name] = {
            "system_prompt": action["system_prompt"],
            "memory": [],
        }
        return f"[agent spawned: {name}]"
    elif t == "learn":
        learning = action.get("fact", "")
        if learning and learning not in runtime_learnings:
            runtime_learnings.append(learning)
            print(f"  [agent learning] {learning}")
        return "[learning recorded]"
    elif t == "done":
        return "__DONE__"
    else:
        return f"unknown action type: {t}"


# ── Agent instructions ────────────────────────────────────────────────────────
AGENT_INSTRUCTIONS = """
You are an autonomous agent in a multi-agent system.
Respond ONLY with a raw JSON array of action objects. No prose, no markdown, no backticks.

Available actions:
  {"type":"shell","cmd":"<command>"}                          — run a command, wait for output (max 30s)
  {"type":"shell_background","cmd":"<command>"}              — start server/long process, returns immediately
  {"type":"shell_wait","cmd":"<command>","seconds":<N>}      — wait N seconds then run command (for startup delays)
  {"type":"file_read","path":"<path>"}
  {"type":"file_write","path":"<path>","content":"<text>"}
  {"type":"file_list","path":"<directory>"}
  {"type":"message","from":"<your name>","to":"<agent name>","content":"<text>"}
  {"type":"spawn_agent","name":"<unique name>","system_prompt":"<full prompt>"}
  {"type":"learn","fact":"<something discovered that all agents should know>"}
  {"type":"done"}

CRITICAL RULES:
1. Output must be a single valid JSON array. All actions go inside ONE array.
2. Read ENVIRONMENT SNAPSHOT — use only confirmed available commands.
3. Read RUNTIME LEARNINGS — treat them as facts, never repeat a failed approach.
4. NEVER use shell for servers or long processes — use shell_background instead.
5. After starting a server with shell_background, use shell_wait to give it time to start, then verify with curl or Invoke-WebRequest.
6. If a command fails, read the error, adapt, try differently. Never retry the exact same thing.
7. Record discoveries with {"type":"learn"} so all agents benefit.
8. Spawn sub-agents for specialist tasks. Message agents when you need their info.
9. {"type":"done"} only when YOUR task is fully complete and verified.
"""


def build_system_prompt(base_prompt: str, env_snapshot: str) -> str:
    learnings_text = ""
    if runtime_learnings:
        learnings_text = "\n\nRUNTIME LEARNINGS (discovered this session — treat as facts):\n"
        learnings_text += "\n".join(f"  - {l}" for l in runtime_learnings)
    return (
        base_prompt
        + f"\n\n{env_snapshot}"
        + learnings_text
        + "\n\n"
        + AGENT_INSTRUCTIONS
    )


# ── Agent runner ──────────────────────────────────────────────────────────────
def run_agent(name: str, task: str, env_snapshot: str) -> tuple[str, bool]:
    agent = agent_registry[name]

    inbox = [m for m in message_bus if m["to"] == name]
    inbox_text = ""
    if inbox:
        inbox_text = "\n\nInbox:\n" + "\n".join(
            f"[from {m['from']}]: {m['content']}" for m in inbox
        )
        message_bus[:] = [m for m in message_bus if m["to"] != name]

    user_msg = f"Task: {task}{inbox_text}"
    agent["memory"].append({"role": "user", "content": user_msg})

    system = build_system_prompt(agent["system_prompt"], env_snapshot)
    response = llm(system, agent["memory"])
    agent["memory"].append({"role": "assistant", "content": response})

    # parse — repair common model mistakes
    try:
        raw = response.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        # repair: model outputs objects side by side instead of array
        if not raw.startswith("["):
            import re
            objects = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw, re.DOTALL)
            if objects:
                raw = "[" + ",".join(objects) + "]"
        actions = json.loads(raw)
        if isinstance(actions, dict):
            actions = [actions]
        actions = [a for a in actions if isinstance(a, dict)]
    except Exception:
        print(f"  [parse error] raw:\n{response[:400]}")
        actions = []

    results = []
    done = False
    for action in actions:
        result = execute_action(action, name)
        if result == "__DONE__":
            done = True
        else:
            results.append(f"[{action.get('type','?')}] → {result}")

    context = "\n".join(results)
    if context:
        agent["memory"].append({"role": "user", "content": f"Results:\n{context}"})

    return context, done


# ── Supervisor ────────────────────────────────────────────────────────────────
SUPERVISOR_PROMPT = """
You are the Supervisor Agent. You receive a raw user goal and must fully achieve it.
Read the ENVIRONMENT SNAPSHOT before doing anything — it tells you OS, shell, tools, and python command.
Read RUNTIME LEARNINGS — respect them as facts.
For servers or long-running processes always use shell_background, then shell_wait to confirm startup.
Spawn specialist agents when needed. Monitor their results. Reassign if something fails.
You are done only when the goal is fully achieved and verified.
"""


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_system(user_goal: str):
    print(f"\n{'='*60}")
    print(f"GOAL: {user_goal}")
    print(f"{'='*60}")

    print("\n[discovering environment...]")
    env_snapshot = discover_environment()
    print(env_snapshot)

    agent_registry["supervisor"] = {
        "system_prompt": SUPERVISOR_PROMPT,
        "memory": [],
    }

    active_agents = {"supervisor"}
    completed_agents = set()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n─── Iteration {iteration} ───")

        for name in list(active_agents):
            if name in completed_agents:
                continue

            task = (
                user_goal
                if name == "supervisor"
                else "Complete your assigned role. Use environment snapshot and runtime learnings. Message supervisor if stuck."
            )

            context, done = run_agent(name, task, env_snapshot)

            print(f"\n[{name}]")
            if context:
                print(context[:2000])

            for agent_name in agent_registry:
                if agent_name not in active_agents:
                    active_agents.add(agent_name)
                    print(f"  → new agent: {agent_name}")

            if done:
                completed_agents.add(name)
                print(f"  ✓ {name} completed")

        for m in message_bus:
            if m["to"] in agent_registry:
                active_agents.add(m["to"])

        remaining = active_agents - completed_agents
        if not remaining:
            print(f"\n{'='*60}")
            print("✓ Goal complete.")
            print(f"{'='*60}\n")
            return

    print("\n[max iterations reached]")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Goal: ").strip()
    run_system(goal)