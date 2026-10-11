# HyperQwen-Mem

Memoria de largo plazo para tu agente HyperQwen (vLLM + Qwen3.8-27B).
Un solo servicio local, un solo archivo SQLite, cero GPU.

## La investigación: qué sistemas existen

| Sistema | Enfoque | Pros | Contras (para tu caso) |
|---|---|---|---|
| **Mem0** (mem0ai) | Extrae "memories" atómicas con LLM, grafo opcional, dedup por embedding | Popular, SDK fácil, self-hostable | Cada write puede costar 1-3 LLM calls (extract + dedup/merge); en un 27B local eso es lento (segundos por hecho). El grafo usa Neo4j si se activa → pesado. |
| **Zep / Graphiti** | Knowledge graph temporal (Neo4j): nodos=entidades, aristas con `valid_at`/`invalid_at`; el tiempo es primera clase | El mejor para hechos que cambian (tu gato Luna→Rayo, "antes vivía en X, ahora en Y"); revisión temporal excelente | Requiere Neo4j (mín. ~2 GB RAM, su propio daemon) + LLM en el write path. Para un setup 1 PC + 27B local es la opción más pesada. |
| **Letta (MemGPT)** | Memoria "core" (system context) + "archival" (vector); el agente se auto-edita su contexto con tools | Arquitectónicamente elegante, agente maneja su propia memoria | MemGPT asume un agente con muchas turns y LLM rápido; con un 27B local las llamadas de self-edit son caras. Su archival store es basically un vector DB. |
| **HippoRAG (1 y 2)** | Inspirado en el hipocampo: grafo de OpenIE + Personalized PageRank; HippoRAG2 elimina LLMs en el retrieval y usa un grafo de palabras | Retrieval sin LLM en el request path (como lo pides); papers con buenos números en QA multi-hop | OpenIE (extracción de triples) es el paso caro y ruidoso; PageRank en cada query escala lineal con el grafo; sin consolidación temporal nativa. |
| **A-MEM** (agentic memory) | Notas con "atoms", enlaces de similitud auto-organizados, memoria que se reescribe con el tiempo (inspired en Zettelkasten) | Interesante para que la memoria "crezca" y se reorganice sola | Complejidad alta, sin benchmarks duros; más investigación que producto. |
| **Generative Agents (Park 2023)** | Stream de recuerdos + reflection (el LLM genera "insights" periódicamente) + decay de recency×importance×recency en el score | El patrón de scoring que casi todos copian (recency/importance/relevance) | Todo el retrieval pasa por un LLM (score de reflection) → no sirve en local sin optimizar. |

### Qué adopté y por qué (tu constraint: "no consuma mucho en long agent")

1. **Nada de LLM en el request path.** Retrieval = embedding 384-dim en CPU (~10 ms) + búsqueda vectorial (sqlite-vec) + FTS5/BM25, fusionados con RRF y decay de recencia. Costo por consulta medido: **~15-23 ms con 1000+ memorias**. El LLM (HyperQwen) solo se usa en el *write path*, en lotes, de fondo.
2. **Consolidación asíncrona** ("sleep"): un worker extrae hechos atómicos de los episodios (mensajes de conversación) en batches; cada batch = 1 LLM call. Si el LLM no está, degrada a un splitter de oraciones (funcional, menos inteligente).
3. **Supersede en vez de delete**: cuando un hecho nuevo contradice uno viejo (similitud alta + LLM confirma), el viejo se marca `superseded` — se puede revisar ("hazle un log de qué recuerdas, incluso lo que ya no vale") y se pruna después de 90 días. El set activo se mantiene pequeño y fresco → **no se degrada con el tiempo**.
4. **Dedup por embedding** al escribir: repetir un hecho no duplica la fila, solo la refresca.
5. **Todo en 1 archivo SQLite** (WAL): portables, revisable con cualquier tool, ~1.8 MB para ~500 memorias + 1000 bulk.

No usé Neo4j, ni FAISS, ni Chroma: para esta escala (miles-millones de memorias) sqlite-vec + FTS5 sobran y son más fáciles de revisar.

## Arquitectura

```
episodes (crudos) --worker LLM--> memories (hechos atómicos)
    + vec0 (cosine 384d) + FTS5 (BM25)
    retrieval: RRF(vector, bm25) × recency-decay × reuse-boost
    supersede: nuevo invalida al viejo (guardado, revisable)
```

- `memcore.py` — store: SQLite + sqlite-vec + FTS5, dedup, search, supersede, prune.
- `consolidator.py` — worker en background (batches → LLM → hechos atómicos; modo degradado sin LLM).
- `server.py` — HTTP: `POST /mcp` (JSON-RPC: memory_search / memory_remember / memory_list),
  `GET /` (página de **revisión** de recuerdos), `GET /stats`.
- `launch.py` — start/stop/status desatendido.
- `test_e2e.py` — test end-to-end (protocolo, dedup, búsqueda, consolidación con el LLM real, supersede, escala).

## Uso

```bash
source /tmp/hmenv/bin/activate        # (venv con sqlite-vec, sentence-transformers, httpx)
python launch.py start                # arranca en 127.0.0.1:8321
python launch.py status               # stats: cuántas memorias, latencia, consolidador
python test_e2e.py                    # test completo

# Revisar recuerdos:  http://127.0.0.1:8321/  (página web, busca y lista)
# Stats:             curl localhost:8321/stats
```

### Integración con HyperQwen (MCP)

Tu agente puede usar el endpoint MCP (`POST /mcp`, JSON-RPC 2.0, `protocolVersion 2025-03-26`):
- `memory_search {query, k}` → los N recuerdos más relevantes.
- `memory_remember {text}` → guardar un hecho atómico directamente (dedup automático).
- `memory_list {status, limit}` → revisar lo guardado (activos, obsoletos, olvidados).

Alternativa sin MCP: envía al agente dos system-hints:
1. Antes de responder, llama `GET /mcp tools/call memory_search` con el resumen de la conversación.
2. Después de conversaciones con información nueva, llama `memory_remember` con hechos atómicos.
El worker además consolida solos los episodios que inyectes con `POST /api/episode`.

### Configuración (`.env`)

| Var | Defecto | Notas |
|---|---|---|
| `MEM_DB` | `./data/memory.db` | ruta del SQLite |
| `MEM_PORT` | `8321` | |
| `HYPERQWEN_URL` | `http://127.0.0.1:18020` | endpoint vLLM para el consolidador |
| `CONSOLIDATE` | `1` | `0` = solo modo degradado (sin LLM) |
| `MEM_BATCH` / `MEM_BATCH_INTERVAL` | `4` / `30s` | tamaño de lote y frecuencia del worker |

### Costo real (tu hardware: Ryzen 9 5950X, GPU ocupada)

- Embeddings: `bge-small-en-v1.5` en CPU, ~10 ms/frase, ~100 MB RAM.
- Retrieval: ~15-23 ms con 1000+ memorias (medido en test).
- Escribir un hecho: 1 embed (~10 ms). Consolidar 4 episodios: 1 LLM call (varios segundos, en background, no bloquea nada).
- GPU: 0 bytes (forzado a CPU).
- DB: ~1.8 MB para ~500 memorias reales + 1000 de prueba.

### Límites conocidos / próximos pasos
- El embedder es bilingüe solo de forma parcial (bge-small); para español puro, `paraphrase-multilingual-MiniLM` es el drop-in (mismo dim 384… no: 384→ hay que migrar la DB: `rm -rf data`).
- Consolidación por LLM: el 27B a veces devuelve JSON imperfecto; el extractor es tolerante, pero si ves `facts=0` con `llm=yes`, el prompt necesita ajuste para tu modelo.
- Sin auth: solo loopback. Si lo expones, agrega una key.
- Pruning agresiva configurable; hoy borra `superseded` > 90 días.
