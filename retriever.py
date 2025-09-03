def retrieve_top_k(query, model, index, chunks, k=3):
    """Return top-k chunks and distances for a user query."""
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append((chunks[idx], float(distances[0][rank])))
    return results
