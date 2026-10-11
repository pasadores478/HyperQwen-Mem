"""HyperQwen-Mem — long-term memory for a local LLM agent, on one SQLite file.

Layers (design rationale in README.md):
  episodes   raw conversation turns / events, stored verbatim (provenance)
  memories   atomic facts consolidated from episodes (the "semantic" layer)
  vec index  sqlite-vec cosine index over memory embeddings (CPU, 384 dim)
  fts index  SQLite FTS5 (BM25) over memory text
  retrieval  vector + BM25 fused with RRF, recency decay + small access boost;
             superseded memories are excluded by default
  consolidation  background worker: batch episodes -> atomic facts via a local
             LLM (OpenAI-compatible endpoint, i.e. HyperQwen), or a naive
             sentence splitter when the LLM is unavailable. Contradictions
             supersede old memories instead of deleting them, so history is
             reviewable and the active set stays small.

Everything runs on CPU; the GPU stays free for the 27B model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_DIM = 384

class Embedder:
    """CPU sentence transformer with a hashing-bag-of-words fallback."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU
        self.model_name = model_name
        self.model = None
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:  # offline / no network: degrade, don't die
            print(f"[memcore] warning: sentence-transformers unavailable ({e}); "
                  f"using hashing fallback embedder", flush=True)
            self.model = None
            self.dim = EMBED_DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None:
            v = self.model.encode(texts, normalize_embeddings=True)
            return [list(map(float, row)) for row in v]
        return [self._hash_embed(t) for t in texts]

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        v = [0.0] * EMBED_DIM
        for tok in re.findall(r"[\w']+|[\u00C0-\u024F]+\w*", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % EMBED_DIM] += 1.0
            v[(h >> 8) % EMBED_DIM] += 0.5  # second weak signal, like a 2nd hash
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def to_blob(self, vec: list[float]) -> bytes:
        return struct.pack(f"<{len(vec)}f", *vec)

    @staticmethod
    def from_blob(blob: bytes) -> list[float]:
        n = len(blob) // 4
        return list(struct.unpack(f"<{n}f", blob))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: int
    text: str
    session: str
    status: str            # remembered | superseded | forgotten
    created_at: float
    superseded_by: int | None = None
    source_episode: int | None = None
    accesses: int = 0
    score: float = 0.0     # filled by search()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY,
  session TEXT,
  text TEXT,
  created_at REAL,
  status TEXT DEFAULT 'pending'        -- pending | consolidated | skipped
);
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  session TEXT,
  status TEXT DEFAULT 'remembered',
  superseded_by INTEGER,
  source_episode INTEGER,
  created_at REAL,
  accesses INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE VIRTUAL TABLE IF NOT EXISTS mem_vec USING vec0(embedding float[384] distance_metric=cosine);
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(text, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
  INSERT INTO mem_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
  INSERT INTO mem_fts(mem_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
  INSERT INTO mem_fts(mem_fts, rowid, text) VALUES('delete', old.id, old.text);
  INSERT INTO mem_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class MemoryStore:
    RRF_K = 60
    HALF_LIFE_DAYS = 365
    DEDUP_SIM = 0.93          # same fact said again -> don't store twice
    CONTRADICT_SIM = 0.85     # above this, ask the LLM if the new fact supersedes

    def __init__(self, path: str, embedder: Embedder | None = None):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(self.db)  # registers the vec0 module on THIS connection
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self._dim_ok = self._check_vec_dim(embedder or Embedder())
        self.embedder = embedder or Embedder()
        self.lock = threading.RLock()

    def _check_vec_dim(self, emb: Embedder) -> bool:
        # vec0 schema bakes in the dim at creation; if the existing table was
        # created with a different dim, rebuild it once.
        row = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE name='mem_vec'"
        ).fetchone()
        want = f"float[{emb.dim}]"
        if row and row[0] and want not in row[0]:
            n = self.db.execute("SELECT COUNT(*) FROM mem_vec").fetchone()[0]
            if n:
                print(f"[memcore] embedding dim changed ({row[0]}); "
                      f"rebuilding vec index for {n} rows", flush=True)
                self.db.execute("DROP TABLE mem_vec")
                self.db.execute(
                    "CREATE VIRTUAL TABLE mem_vec USING vec0("
                    f"embedding {want} distance_metric=cosine)"
                )
                self._reindex_vec()
                self.db.commit()
            else:
                self.db.execute("DROP TABLE mem_vec")
                self.db.execute(
                    "CREATE VIRTUAL TABLE mem_vec USING vec0("
                    f"embedding {want} distance_metric=cosine)"
                )
                self.db.commit()
        return True

    def _reindex_vec(self):
        emb = self.embedder
        rows = self.db.execute(
            "SELECT id, text FROM memories WHERE status != 'forgotten'"
        ).fetchall()
        texts = [t for _, t in rows]
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            vecs = emb.encode(chunk)
            ids = [r[0] for r in rows[i:i + 64]]
            self.db.executemany(
                "INSERT INTO mem_vec(rowid, embedding) VALUES (?, ?)",
                [(i_, emb.to_blob(v)) for i_, v in zip(ids, vecs)],
            )
        self.db.commit()

    # -- writes --------------------------------------------------------------

    def add_episode(self, text: str, session: str = "default",
                    when: float | None = None) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO episodes(session, text, created_at, status) VALUES (?,?,?,?)",
                (session, text, when or time.time(), "pending"),
            )
            self.db.commit()
            return cur.lastrowid

    def add_memory(self, text: str, session: str = "default",
                   source_episode: int | None = None) -> int | None:
        """Add an atomic fact. Returns new id, or None if it was a duplicate."""
        vec = self.embedder.encode([text])[0]
        with self.lock:
            dup = self._nearest(vec, 1)
            if dup and self._similarity(dup[0][1]) >= self.DEDUP_SIM:
                existing_id, _ = dup[0]
                # touch the existing memory instead of duplicating it
                self.db.execute(
                    "UPDATE memories SET accesses = accesses + 1 WHERE id=?",
                    (existing_id,),
                )
                self.db.commit()
                return None
            cur = self.db.execute(
                "INSERT INTO memories(text, session, source_episode, created_at)"
                " VALUES (?,?,?,?)",
                (text, session, source_episode, time.time()),
            )
            mid = cur.lastrowid
            self.db.execute(
                "INSERT INTO mem_vec(rowid, embedding) VALUES (?, ?)",
                (mid, self.embedder.to_blob(vec)),
            )
            self.db.commit()
            return mid

    def _nearest(self, vec: list[float], k: int) -> list[tuple[int, float]]:
        """Returns [(memory_id, cosine_distance), ...] over non-forgotten rows."""
        # NOTE: sqlite-vec only computes `distance` when ORDER BY distance is
        # present; a bare WHERE returns NULL distances, so always order.
        rows = self.db.execute(
            "SELECT rowid, distance FROM mem_vec "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (self.embedder.to_blob(vec), max(k * 8, 64)),
        ).fetchall()
        if not rows:
            return []
        live = {
            r[0]: r[1]
            for r in self.db.execute(
                "SELECT id, status FROM memories WHERE id IN ({})".format(
                    ",".join("?" * len(rows)))
                , [r[0] for r in rows]
            )
            if r[1] in ("remembered", "superseded")
        }
        out = [(i, d) for i, d in rows if i in live and d is not None]
        out.sort(key=lambda x: x[1])
        return out[:k]

    def _similarity(self, dist: float) -> float:
        # cosine metric: distance = 1 - cos  (clamped for safety)
        return max(0.0, min(1.0, 1.0 - dist))

    # -- retrieval -----------------------------------------------------------

    def search(self, query: str, k: int = 5, session: str | None = None,
               include_superseded: bool = False) -> list[tuple[Memory, float]]:
        t0 = time.time()
        qvec = self.embedder.encode([query])[0]

        # vector candidates
        rows = self.db.execute(
            "SELECT rowid, distance FROM mem_vec WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT 100",
            (self.embedder.to_blob(qvec),),
        ).fetchall()
        vec_rank = {rowid: i for i, (rowid, _) in enumerate(rows)}

        # bm25 candidates
        tokens = re.findall(r"[\w']+|[\u00C0-\u024F]+\w*", query.lower())
        fts_q = " OR ".join('"%s"' % t.replace('"', '""') for t in set(tokens))
        fts_rank: dict[int, int] = {}
        if fts_q:
            try:
                frows = self.db.execute(
                    "SELECT rowid, bm25(mem_fts) r FROM mem_fts "
                    "WHERE mem_fts MATCH ? ORDER BY r LIMIT 100",
                    (fts_q,),
                ).fetchall()
                fts_rank = {r[0]: i for i, r in enumerate(frows)}
            except sqlite3.OperationalError:
                pass

        candidate_ids = sorted(set(vec_rank) | set(fts_rank))
        if not candidate_ids:
            return []

        where = "WHERE id IN ({})".format(",".join("?" * len(candidate_ids)))
        if session:
            where += " AND session = ?"
            args = list(candidate_ids) + [session]
        else:
            args = list(candidate_ids)
        if not include_superseded:
            where += " AND status = 'remembered'"
        mrows = self.db.execute(
            f"SELECT id, text, session, status, created_at, superseded_by, "
            f"source_episode, accesses FROM memories {where} ORDER BY id",
            args,
        ).fetchall()

        now = time.time()
        out: list[tuple[Memory, float]] = []
        for r in mrows:
            mid = r[0]
            score = 0.0
            if mid in vec_rank:
                score += 1.0 / (self.RRF_K + vec_rank[mid])
            if mid in fts_rank:
                score += 1.0 / (self.RRF_K + fts_rank[mid])
            age_days = max(0.0, (now - r[4]) / 86400.0)
            score *= 0.5 ** (age_days / self.HALF_LIFE_DAYS)          # recency
            score *= 1.0 + 0.05 * min(r[7] or 0, 20)                  # reuse boost
            m = Memory(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
            out.append((m, score))
        out.sort(key=lambda p: -p[1])

        top = out[:k]
        if top:
            self.db.execute(
                f"UPDATE memories SET accesses = accesses + 1 WHERE id IN ({",".join('?'*len(top))})",
                [m.id for m, _ in top],
            )
            self.db.commit()
        for m, s in top:
            m.score = s
        top_debug = f"{len(top)} hits in {(time.time()-t0)*1000:.0f}ms" if top else "no hits"
        self._last_search_ms = (time.time() - t0) * 1000
        self._last_search_info = top_debug
        return top

    # -- supersede / review / prune -------------------------------------------

    def supersede(self, old_id: int, new_id: int) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE memories SET status='superseded', superseded_by=? WHERE id=?",
                (new_id, old_id),
            )
            self.db.commit()

    def list_memories(self, status: str = "remembered", limit: int = 100,
                      offset: int = 0) -> list[Memory]:
        if status == "any":
            rows = self.db.execute(
                "SELECT id, text, session, status, created_at, superseded_by, "
                "source_episode, accesses FROM memories "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, text, session, status, created_at, superseded_by, "
                "source_episode, accesses FROM memories WHERE status=? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        return [Memory(*r) for r in rows]

    def forget_old_superseded(self, older_than_days: float = 90) -> int:
        """Prune superseded facts past their review window (keeps active set small)."""
        cutoff = time.time() - older_than_days * 86400
        with self.lock:
            ids = [r[0] for r in self.db.execute(
                "SELECT id FROM memories WHERE status='superseded' AND created_at < ?",
                (cutoff,)).fetchall()]
            for i in ids:
                self.db.execute("DELETE FROM mem_vec WHERE rowid=?", (i,))
            if ids:
                self.db.execute(
                    "DELETE FROM memories WHERE id IN ({})".format(
                        ",".join("?" * len(ids))), ids)
                self.db.commit()
            return len(ids)

    def pending_episodes(self, limit: int = 5) -> list[tuple[int, str, str, float]]:
        return self.db.execute(
            "SELECT id, session, text, created_at FROM episodes "
            "WHERE status='pending' ORDER BY id LIMIT ?", (limit,)
        ).fetchall()

    def mark_episodes(self, ids: list[int], status: str) -> None:
        if not ids:
            return
        with self.lock:
            self.db.execute(
                f"UPDATE episodes SET status=? WHERE id IN ({",".join('?'*len(ids))})",
                [status] + ids,
            )
            self.db.commit()

    def stats(self) -> dict:
        counts = dict(self.db.execute(
            "SELECT status, COUNT(*) FROM memories GROUP BY status").fetchall())
        ecounts = dict(self.db.execute(
            "SELECT status, COUNT(*) FROM episodes GROUP BY status").fetchall())
        size = os.path.getsize(self.path)
        return {
            "path": self.path,
            "db_bytes": size,
            "memories": counts,
            "episodes": ecounts,
            "last_search_ms": getattr(self, "_last_search_ms", None),
            "last_search": getattr(self, "_last_search_info", None),
            "embedding": self.embedder.model_name if self.embedder.model
                        else "hashing-fallback",
        }
