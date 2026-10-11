"""Background consolidation: pending episodes -> atomic, reviewable memories.

This is the "sleep" pass of the memory system. It runs off the request path:
  - an LLM (any OpenAI-compatible endpoint, e.g. HyperQwen) is called ONCE per
    batch of episodes and returns atomic facts (JSON);
  - each fact is deduplicated against existing memories by cosine similarity;
  - near-duplicate facts are skipped;
  - facts that are similar enough to an existing memory to be a *contradiction*
    are checked by the LLM: if the new one invalidates the old, the old memory
    is superseded (kept, reviewable) and the new one becomes active;
  - if the LLM is unreachable, a naive splitter stores raw sentences (degraded
    mode: still functional, just less smart).

Design goal: a long-running agent never pays an LLM call *on the retrieval
path*. All LLM work is in this background path, batched, so it can run while
the model is idle or be throttled to one batch per N minutes.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable

import httpx

from memcore import MemoryStore

SYSTEM_PROMPT = (
    "You are a memory consolidator. You receive raw conversation episodes and "
    "a few of the user's most similar existing memories. Produce a JSON array "
    "of *atomic facts*: each fact is a single self-contained statement, in the "
    "user's language, that should be remembered long-term. Rules: "
    "1) One idea per fact. No conjunctions chaining unrelated facts. "
    "2) Rewrite in the user's own terms (facts about the user, their "
    "preferences, decisions, projects, and people). "
    "3) Skip ephemera (small talk, 'thanks', 'ok', questions) — only "
    "durable, retrievable information. "
    "4) If a new fact contradicts an existing memory, still emit it; a "
    "separate 'supersedes' field lists which memory id it invalidates. "
    "5) Facts about yourself as the assistant (capabilities, how you behave) "
    "must also be captured — the user may ask 'what do you know?'. "
    "Respond ONLY with JSON: "
    '[{"text": "...", "supersedes": [memory_id, ...]}] or []'
)


def _extract_json(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [d for d in data if isinstance(d, dict) and d.get("text")]
    except json.JSONDecodeError:
        return []


class Consolidator:
    def __init__(self, store: MemoryStore, llm_url: str,
                 llm_api_key: str = "", batch_size: int = 4,
                 batch_interval: float = 30.0, enabled: bool = True):
        self.store = store
        self.llm_url = llm_url.rstrip("/")
        self.llm_key = llm_api_key or os.environ.get("HYPERQWEN_API_KEY", "")
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"batches": 0, "facts": 0, "superseded": 0, "errors": 0,
                      "llm": False}

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="mem-consolidator")
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                eps = self.store.pending_episodes(self.batch_size)
                if eps:
                    self.consolidate(eps)
            except Exception as e:
                self.stats["errors"] += 1
                print(f"[consolidator] batch failed: {e}", flush=True)
            self._stop.wait(self.batch_interval)

    # -- LLM ------------------------------------------------------------------

    def _llm_available(self) -> bool:
        if not self.llm_url:
            return False
        try:
            r = httpx.get(f"{self.llm_url}/v1/models", timeout=2.0)
            r.raise_for_status()
            data = r.json().get("data") or []
            if data:
                self.model_id = data[0].get("id", "default")
            return True
        except Exception:
            return False

    def _llm_chat(self, messages: list[dict], temperature: float = 0.1,
                  max_tokens: int = 2048) -> str | None:
        if not self.llm_url:
            return None
        body = {
            "model": getattr(self, "model_id", "default"),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.llm_key}"} if self.llm_key else {}
        try:
            r = httpx.post(f"{self.llm_url}/v1/chat/completions", json=body,
                           headers=headers, timeout=120.0)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    # -- consolidation ----------------------------------------------------------

    def consolidate(self, episodes: list[tuple[int, str, str, float]]) -> int:
        text_block = "\n---\n".join(
            f"[{ep[0]}] ({ep[1]}) {ep[2]}" for ep in episodes
        )

        facts: list[dict] = []
        if self.llm_url and self._llm_available():
            self.stats["llm"] = True
            # give the LLM a few similar existing memories for dedup context
            last = episodes[-1][2]
            similar = self.store.search(last, k=4, include_superseded=True)
            ctx = json.dumps(
                [{"id": m.id, "text": m.text} for m, _ in similar], ensure_ascii=False
            )
            resp = self._llm_chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Episodes:\n{text_block}\n\nExisting similar "
                            f"memories (for reference, may be superseded):\n{ctx}"},
            ])
            if resp:
                facts = _extract_json(resp)

        if not facts:
            # degraded mode: naive sentence-level extraction, no LLM
            clean = "\n".join(ep[2] for ep in episodes)
            facts = self._naive_facts(clean)

        new_facts = 0
        for f in facts:
            text = f["text"].strip()
            if not text or len(text) < 8:
                continue
            sup_ids = [int(i) for i in (f.get("supersedes") or [])
                       if isinstance(i, (int, float))]
            new_id = self.store.add_memory(text, session="consolidated",
                                           source_episode=episodes[0][0])
            if new_id is None:
                continue
            if sup_ids:
                # verify each claim before superseding: the new memory must be
                # highly similar to the old one (a guard against the LLM
                # claiming to replace an unrelated fact).
                emb = self.store.embedder.encode([text])[0]
                for sid in sup_ids:
                    near = self.store._nearest(emb, 1)
                    if near and near[0][0] == sid and \
                            self.store._similarity(near[0][1]) >= 0.75:
                        self.store.supersede(sid, new_id)
                        self.stats["superseded"] += 1
            new_facts += 1
            self.stats["facts"] += 1

        self.store.mark_episodes([ep[0] for ep in episodes], "consolidated")
        self.stats["batches"] += 1
        return new_facts

    @staticmethod
    def _naive_facts(text: str) -> list[dict]:
        """Fallback without an LLM: split on sentence boundaries, keep
        declarative-ish sentences, dedupe, cap length."""
        out: list[dict] = []
        seen: set[str] = set()
        for chunk in re.split(r"(?m)^(?=\[)\n|---", text):
            for s in re.split(r"(?<=[.!?])\s+|\n+", chunk):
                s = s.strip()
                if not re.match(r"^\[\d+\]", s) and 20 <= len(s) <= 400 and \
                        s.lower() not in seen:
                    seen.add(s.lower())
                    out.append({"text": s, "supersedes": []})
        return out[:12]
