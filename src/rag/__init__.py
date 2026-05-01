"""Local embedding-based retrieval as an optional alternative to LLM Stage 1.

Layout:
- node_to_text: serialize a UML node into a short document for embedding.
- encoder:      load a SentenceTransformer model, encode in batches.
- cache:        on-disk persistence for computed embeddings.
- retriever:    cosine-similarity top-K lookup.
"""
