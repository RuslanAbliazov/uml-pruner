# Approach #4 — Human-like agent

**Status:** stub.

## Idea (planned)

An agent picks N anchors via RAG, expands the neighborhood by graph
centrality (betweenness) and call-site frequency, then prunes via LLM.
Mimics how a human would explore a large codebase.
