"""One small LLM client with three providers and a disk cache.

  anthropic:<model>   Anthropic API via the official SDK (pip install anthropic; needs ANTHROPIC_API_KEY)
  ollama:<model>      local Ollama server
  claude-cli:<model>  the `claude -p` command line, if installed and signed in

Pick with DME_LLM=provider:model. Default: Anthropic if a key is set, else Ollama.
Responses are cached on disk by a hash of the full request, so a re-run of the
same scenario replays instantly; pass use_cache=False for fresh calls."""
import hashlib
import json
import os
import re
import subprocess
import threading
import urllib.request
from pathlib import Path


def default_spec():
    if os.environ.get("DME_LLM"):
        return os.environ["DME_LLM"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic:claude-opus-5-5"
    return "ollama:qwen3.5:9b-q4_K_M"


class LLM:
    def __init__(self, spec=None, cache_dir=".llm_cache", use_cache=True):
        self.provider, self.model = (spec or default_spec()).split(":", 1)
        self.cache = Path(cache_dir)
        self.use_cache = use_cache
        self.calls = self.hits = 0
        self._lock = threading.Lock()

    def chat(self, system, messages, json_mode=False, max_tokens=220):
        req = [self.provider, self.model, system, messages, json_mode, max_tokens]
        path = self.cache / (hashlib.sha256(json.dumps(req).encode()).hexdigest()[:32] + ".json")
        if self.use_cache and path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text())["text"]
        text = getattr(self, "_" + self.provider.replace("-", "_"))(system, messages, json_mode, max_tokens)
        with self._lock:
            self.calls += 1
        self.cache.mkdir(exist_ok=True)
        path.write_text(json.dumps({"request": req, "text": text}, indent=1))
        return text

    def json(self, system, user, max_tokens=500):
        # One retry with a firmer instruction: a cached bad answer would otherwise repeat forever.
        for extra in ("", "\n\nYour previous answer was not valid JSON. Return one valid JSON object only."):
            text = self.chat(system, [{"role": "user", "content": user + extra}], json_mode=True, max_tokens=max_tokens)
            m = re.search(r"\{.*\}", text, re.S)
            try:
                out = json.loads(m.group(0) if m else text)
                if isinstance(out, dict):
                    return out
            except json.JSONDecodeError:
                pass
        raise ValueError("model did not return a JSON object")

    # --- providers -------------------------------------------------------
    def _post(self, url, body, headers=None):
        r = urllib.request.Request(url, json.dumps(body).encode(),
                                   {"content-type": "application/json", **(headers or {})})
        with urllib.request.urlopen(r, timeout=180) as resp:
            return json.loads(resp.read())

    def _ollama(self, system, messages, json_mode, max_tokens):
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}] + messages,
                "options": {"temperature": 0, "seed": 7, "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return self._post(host + "/api/chat", body)["message"]["content"].strip()

    def _anthropic(self, system, messages, json_mode, max_tokens):
        import anthropic                       # optional dependency, only for this provider
        if json_mode:
            system += "\nRespond with a single JSON object and nothing else."
        client = anthropic.Anthropic()
        # Current Claude models think before answering and that counts toward max_tokens,
        # so leave headroom and keep effort low: these are short, simple turns.
        args = dict(model=self.model, max_tokens=max(max_tokens, 4000), system=system, messages=messages,
                    output_config={"effort": "low"})
        resp = None
        if self.model.startswith(("claude-opus-5", "claude-fable-5")):
            try:    # if a safety classifier declines, let the API re-run on a fallback model
                resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default", **args)
            except anthropic.BadRequestError:
                resp = None
        if resp is None:
            resp = client.messages.create(**args)
        if resp.stop_reason == "refusal":
            raise RuntimeError("model declined the request")
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _claude_cli(self, system, messages, json_mode, max_tokens):
        if json_mode:
            system += "\nRespond with a single JSON object and nothing else."
        convo = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        prompt = convo + "\n\nWrite only the next [assistant] message, with no label."
        out = subprocess.run(["claude", "-p", prompt, "--system-prompt", system, "--model", self.model,
                              "--no-session-persistence"],
                             capture_output=True, text=True, timeout=180)
        if out.returncode:
            raise RuntimeError(out.stderr[:300])
        return re.sub(r"^\[assistant\]\s*", "", out.stdout.strip())      # the CLI sometimes echoes the role label
