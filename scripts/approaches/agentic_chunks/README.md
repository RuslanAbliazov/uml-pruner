# Approach #3 — Agentic chunked selection

**Status:** stub. Public interface defined; implementation TBD.

## Idea

1. **Chunk the full diagram.**
   Split nodes into manageable groups (≤ a few dozen classes per chunk):
   - by package (default — packages are the natural cohesive unit),
   - by fixed size (round-robin / hash buckets),
   - by community detection (e.g. Louvain over the relation graph).
2. **Survey each chunk in parallel.**
   For each chunk, prompt the LLM:
   _"Given this query and these N classes (with their methods/relations),
    list classes you believe are relevant, with confidence."_
3. **Synthesize.**
   Aggregate every chunk's positive answers; ask a final LLM pass to
   deduplicate and produce the final REQUIRED / USEFUL split.

## Why this might beat the others
- The agent **always sees every class** (chunked, but no upstream filter
  drops anything). Recall ceiling is essentially the LLM's own.
- Embarrassingly parallel — chunks run concurrently.

## Why it might lose
- Cost is O(diagram_size) LLM calls. For 20k-node diagrams that's ~200+ calls
  per query unless we cache per-chunk answers across queries.
- Chunk boundaries can split tightly-related classes. Mitigation: include the
  immediate neighborhood of every chunk member as low-priority "context".

## Plan / TODOs
- [ ] Implement chunking strategies in
      `src/approaches/agentic_chunks/chunkers.py`:
      `package`, `size`, `louvain`.
- [ ] Per-chunk survey prompt (new file in `prompts/`, e.g.
      `chunk_survey_user.txt`).
- [ ] Synthesizer prompt (e.g. `chunk_synthesis_user.txt`).
- [ ] Async fan-out with concurrency cap.
- [ ] YAML config: `approaches.agentic_chunks.*`.

## Run (once implemented)

```bash
python scripts/approaches/agentic_chunks/run.py
# Equivalent to:
python scripts/benchmark.py --approach agentic_chunks
```

## Source
- `src/approaches/agentic_chunks/runner.py`
