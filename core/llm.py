import json
import time
import requests
from config import NVIDIA_API_KEY, BASE_URL, MODEL, MAX_TOKENS, LLM_TEMPERATURE


def llm(system: str, messages: list[dict], print_fn=print) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
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
                timeout=90,
                stream=True,
            )
            r.raise_for_status()

            chunks = []
            print_fn("  [llm thinking", end="", flush=True)
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            chunks.append(delta)
                            print_fn(".", end="", flush=True)
                    except Exception:
                        pass
            print_fn("]")
            return "".join(chunks)

        except Exception as e:
            print_fn(f"\n  [llm error {attempt+1}/3]: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    return "[]"
