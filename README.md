# JARVIS 2.0

> **J**ust **A** **R**ather **V**ery **I**ntelligent **S**ystem — an autonomous, self-improving multi-agent AI platform powered by NVIDIA's LLM API.

JARVIS 2.0 orchestrates a team of AI agents that collaboratively pursue goals by writing and running code, managing files, spawning sub-agents, and learning from every run. Between sessions, it persists a structured memory so each run builds on the knowledge of all previous ones.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Web UI (recommended)](#web-ui-recommended)
  - [CLI Mode](#cli-mode)
  - [Self-Play / Self-Improvement Loop](#self-play--self-improvement-loop)
- [Agent Action System](#agent-action-system)
- [Persistent Memory & Self-Improvement](#persistent-memory--self-improvement)
- [Objective Metrics & Scoring](#objective-metrics--scoring)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Description |
|---|---|
| **Multi-Agent Orchestration** | Supervisor agent spawns and coordinates specialist sub-agents in parallel |
| **Persistent Memory** | Cross-run learning stored in `jarvis_memory.json`; every session builds on previous ones |
| **Self-Improvement Loop** | RL-like training loop that auto-generates goals, scores runs objectively, and evolves agent prompts |
| **Real-Time Web UI** | FastAPI + WebSocket dashboard for live monitoring of agent activity |
| **Tool System** | Agents can run shell commands, read/write files, send messages, and spawn new agents |
| **Objective Scoring** | Performance measured by hard shell-pass rates and file-validation rates — no biased self-grading |
| **Prompt Evolution** | Ranked instruction store that deduplicates, reinforces, and prunes prompt addons automatically |
| **Cross-Platform** | Runs on Linux, macOS, and Windows; auto-detects shell and Python command |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│          Web UI (FastAPI + WebSocket)  │  CLI            │
└───────────────────────┬─────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  MultiAgentOrchestrator  │
              │  (core/orchestrator.py)  │
              └─────┬──────────┬───┘
                    │          │
         ┌──────────▼──┐  ┌───▼──────────┐
         │  Supervisor  │  │  Specialist   │
         │    Agent     │  │   Agents      │
         └──────┬───────┘  └───┬──────────┘
                │              │
         ┌──────▼──────────────▼──────┐
         │         Tool Layer          │
         │  shell / file / message /   │
         │  spawn_agent / learn / done │
         └──────────────┬─────────────┘
                        │
         ┌──────────────▼─────────────┐
         │       Persistent Memory     │
         │     (core/memory.py)        │
         │   jarvis_memory.json        │
         └────────────────────────────┘
```

**Key flows:**

1. **Goal → Run**: The orchestrator seeds a `supervisor` agent with the user goal and environment snapshot.
2. **Iteration loop**: Each iteration runs all active agents in parallel (up to 8 threads). Agents respond with JSON action arrays.
3. **Action execution**: The orchestrator executes actions (shell, file I/O, spawn, message, learn, etc.) and feeds results back into each agent's memory.
4. **Learning**: Agents emit `learn` facts that are shared across all agents immediately. `update_prompt` addons are ranked and persisted for future sessions.
5. **Memory**: After each run, a `RunRecord` is appended to `jarvis_memory.json`. The memory context is injected into every agent system prompt on the next run.

---

## Project Structure

```
JARVIS-2.0/
├── main.py                  # CLI entry point
├── server.py                # FastAPI web server + WebSocket
├── self_play.py             # Self-improvement / RL loop CLI
├── agent.py                 # Standalone legacy agent runner
├── rugved.py                # Simple calculator CLI (example tool)
├── config.py                # Configuration (reads .env)
├── requirements.txt
├── ui/
│   └── index.html           # Single-page web dashboard
└── core/
    ├── orchestrator.py      # MultiAgentOrchestrator + prompts
    ├── memory.py            # Persistent cross-run memory
    ├── self_improvement.py  # SelfImprovementLoop
    ├── llm.py               # NVIDIA streaming LLM wrapper
    ├── tools.py             # shell / file_op primitives
    ├── metrics.py           # Objective RunMetrics + scoring
    ├── validator.py         # Deterministic action validation
    ├── environment.py       # Live environment discovery
    └── __init__.py
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [NVIDIA API key](https://integrate.api.nvidia.com/) with access to `mistralai/mistral-small-4-119b-2603` (or another supported model)

### Installation

```bash
# Clone the repository
git clone https://github.com/Rugvedrc/JARVIS-2.0.git
cd JARVIS-2.0

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root (copy the values below and fill in your key):

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
BASE_URL=https://integrate.api.nvidia.com/v1/chat/completions
MODEL=mistralai/mistral-small-4-119b-2603
MAX_ITERATIONS=30
MAX_TOKENS=4096
LLM_TEMPERATURE=0.2
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

All settings have sensible defaults. The only value you **must** supply is `NVIDIA_API_KEY`.

---

## Usage

### Web UI (recommended)

Start the server and open the dashboard in your browser:

```bash
python server.py
# → http://localhost:8000
```

The dashboard connects via WebSocket and streams live events: agent status, actions, learnings, validation results, and run completion. Pass `--no-browser` to suppress auto-opening the browser:

```bash
python server.py --no-browser
```

**API endpoints exposed by the server:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/api/run` | Start a new run `{"goal": "..."}` |
| `POST` | `/api/stop` | Stop the current run |
| `GET` | `/api/status` | Current run stats and agent states |
| `WS` | `/ws` | WebSocket stream of all events |

### CLI Mode

Run a single goal from the command line:

```bash
python main.py "Write a Python script that prints the Fibonacci sequence up to 100"
```

If no goal is provided as an argument, you will be prompted interactively:

```bash
python main.py
# Goal: <type your goal here>
```

### Self-Play / Self-Improvement Loop

Run multiple cycles where JARVIS auto-generates goals and improves over time:

```bash
# 5 auto-generated cycles (default)
python self_play.py

# 3 cycles with a fixed goal
python self_play.py --cycles 3 --goal "Create a web scraper for Hacker News headlines"

# Reset persistent memory and start fresh
python self_play.py --reset

# Print the current memory state and exit
python self_play.py --show-memory

# Use a custom memory file
python self_play.py --memory-file my_memory.json
```

After each cycle a score summary is printed:

```
══════════════════════════════════════════════════════════════════════
  SELF-PLAY COMPLETE — SUMMARY
══════════════════════════════════════════════════════════════════════
  RUN   SCORE  PASS%  ITERS     SEC  GOAL
  ─────────────────────────────────────────────────────────────────
  ✓ #1     8.5    90%     4    12.3s  Write a Python script…
  ✓ #2     9.0   100%     3     9.8s  List all .py files…
```

---

## Agent Action System

Agents respond exclusively with a JSON array of action objects. Available actions:

| Action | Description |
|---|---|
| `{"type":"shell","cmd":"..."}` | Run a shell command (30 s timeout) |
| `{"type":"shell_background","cmd":"..."}` | Start a long-running process (server, etc.) |
| `{"type":"shell_wait","cmd":"...","seconds":N}` | Wait N seconds then run a command |
| `{"type":"file_read","path":"..."}` | Read a file |
| `{"type":"file_write","path":"...","content":"..."}` | Write a file (creates parent dirs) |
| `{"type":"file_list","path":"..."}` | Recursively list files in a directory |
| `{"type":"message","from":"...","to":"...","content":"..."}` | Send a message to another agent |
| `{"type":"spawn_agent","name":"...","system_prompt":"..."}` | Spawn a specialist sub-agent |
| `{"type":"learn","fact":"..."}` | Record a fact shared with all agents |
| `{"type":"self_evaluate","feedback":"...","lessons":[...]}` | Agent reflects on its own run |
| `{"type":"update_prompt","addon":"..."}` | Propose a new standing instruction |
| `{"type":"done"}` | Signal that this agent's task is complete |

Every `shell` and `file_write` result is automatically validated by a deterministic rule engine (`core/validator.py`) with zero LLM calls.

---

## Persistent Memory & Self-Improvement

JARVIS stores everything it learns in `jarvis_memory.json` (excluded from version control via `.gitignore`).

The memory contains:

- **Run records** — goal, timestamps, metrics, agent self-feedback, and lessons for every past run
- **Global learnings** — facts accumulated from `learn` actions across all runs
- **Ranked prompt instructions** — standing instructions contributed by agents, ranked by a Jaccard-similarity deduplication algorithm and pruned when they fall below a minimum confidence score
- **Performance trend** — rolling objective score history

On every new run, the full memory context is injected into each agent's system prompt, giving them continuity across sessions.

**Prompt instruction lifecycle:**

```
Agent emits update_prompt ──► Jaccard similarity check
                                ├── exact duplicate → reinforce score (+0.5)
                                ├── same topic      → replace + boost score (+0.3)
                                └── new topic       → add with score 1.0
                              ──► prune below min_score (0.4)
                              ──► cap at 15 instructions
                              ──► recompile system_prompt_addon
```

---

## Objective Metrics & Scoring

Each run is scored deterministically using execution outcomes — the model never grades itself:

| Metric | Weight | Description |
|---|---|---|
| Shell pass rate | 50% | Fraction of shell commands with no failure signal |
| File validation rate | 30% | Fraction of written files that exist and are non-empty (+ Python syntax check for `.py` files) |
| Validation error penalty | 20% | −2 points per validation error (capped at 10 errors) |

**Final score formula (0–10):**

```
score = (pass_rate × 10 × 0.5)
      + (file_validation_rate × 10 × 0.3)
      + (max(0, 10 − validation_errors × 2) × 0.2)
```

---

## Contributing

Pull requests are welcome! For significant changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and ensure the code runs cleanly
4. Open a pull request

---

## License

This project is open source. See the repository for license details.
