import os
import subprocess
import time


def shell(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        return out[:8000] if out else "(no output — exit code 0, success)"
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 30s — use shell_background for long processes"
    except Exception as e:
        return f"ERROR: {e}"


def shell_background(cmd: str) -> str:
    try:
        kwargs = dict(shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(cmd, **kwargs)
        return f"started in background: {cmd}"
    except Exception as e:
        return f"ERROR: {e}"


def shell_wait(cmd: str, seconds: int) -> str:
    time.sleep(max(0, seconds))
    return shell(cmd)


def file_op(op: str, path: str, content: str = "") -> str:
    try:
        if op == "read":
            with open(path, "r", errors="replace") as f:
                return f.read()
        elif op == "write":
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"written: {path} ({len(content)} bytes)"
        elif op == "list":
            entries = []
            for root, dirs, files in os.walk(path or "."):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for file in files:
                    entries.append(os.path.join(root, file))
            return "\n".join(entries[:300]) or "(empty)"
    except Exception as e:
        return f"ERROR: {e}"
