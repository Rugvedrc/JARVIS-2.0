import os
import platform
import subprocess


def _run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""


def discover_environment() -> str:
    r = {}
    r["os"]       = f"{platform.system()} {platform.release()}"
    r["cwd"]      = os.getcwd()
    r["files"]    = _run("dir /b" if os.name == "nt" else "ls")

    python_cmd = None
    for cmd in ["py", "python", "python3"]:
        out = _run(f"{cmd} --version")
        r[f"cmd_{cmd}"] = out if (out and "Python" in out) else "NOT AVAILABLE"
        if out and "Python" in out and not python_cmd:
            python_cmd = cmd

    r["python_command"]    = python_cmd or "NONE FOUND"
    r["shell"]             = "PowerShell/cmd (Windows)" if os.name == "nt" else "bash (Unix)"
    r["command_chaining"]  = "use ';' not '&&'" if os.name == "nt" else "use '&&' or ';'"
    r["background_process"]= "use shell_background for servers / long processes"

    for tool in ["node", "npm", "git", "pip", "curl"]:
        out = _run(f"{tool} --version")
        r[tool] = out if (out and "not found" not in out.lower()) else "NOT AVAILABLE"

    lines = ["=== ENVIRONMENT (live, verified) ==="]
    for k, v in r.items():
        lines.append(f"  {k}: {v}")
    lines.append("=== END ENVIRONMENT ===")
    return "\n".join(lines)
